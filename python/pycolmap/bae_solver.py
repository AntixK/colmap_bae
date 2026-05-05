"""BAE solver bridge for COLMAP bundle adjustment.

Called from C++ BaeBundleAdjuster::Solve() via pybind11 embedded Python.
Follows InstantSFM's TorchBA architecture: base LM optimizer with separate
parameter blocks and rotate_quat projection.
"""

import os
import sys
from functools import partial

# PYTHONHASHSEED only takes effect at interpreter startup, so writing it
# here does NOT alter the hash randomisation of the current process (the
# CLI binary embeds Python and has already initialised it before this
# module loads).  We set it anyway for two reasons:
#   1. Subprocesses spawned from this module would inherit a deterministic
#      seed (we don't spawn any today, but defense in depth).
#   2. Documents the intent — for full hash determinism, set
#      PYTHONHASHSEED=0 in the shell that launches colmap (run_benchmark.py
#      already does this via subprocess env).
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pypose as pp
import torch
import torch.nn as nn

from bae.autograd.function import TrackingTensor, map_transform
from bae.autograd.graph import jacobian
from bae.optim import LM
from bae.sparse.py_ops import diagonal_op_
from bae.utils.ba import rotate_quat
from bae.utils.pysolvers import PCG
from pypose.optim.kernel import Huber, Cauchy


def _log(msg):
    """Print a [BAE] message; flushed so it interleaves with C++ logs."""
    print(f"[BAE] {msg}", flush=True)
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _distort_and_project(points_cam, intrinsics):
    """Perspective divide + SIMPLE_RADIAL distortion + focal scaling."""
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., 0].unsqueeze(-1)
    k1 = intrinsics[..., 1].unsqueeze(-1)
    n = torch.sum(points_proj**2, dim=-1, keepdim=True)
    # BAE is currently restricted to COLMAP SIMPLE_RADIAL, which only has k1.
    # The third slot is kept in the buffer for bridge compatibility and must
    # remain semantically inert.
    r = 1 + k1 * n
    return points_proj * r * f


def _transform_points(extrinsics, points):
    """Apply row-major 3x4 world-to-camera extrinsics to 3D points."""
    rotations = extrinsics[..., :3]
    translations = extrinsics[..., 3]
    return torch.matmul(rotations, points.unsqueeze(-1)).squeeze(-1) + translations


def _rotate_points_xyzw(quaternions, points):
    """Rotate 3D points by xyzw quaternions using pure Torch ops."""
    q = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    twice_cross = 2.0 * torch.cross(q_xyz, points, dim=-1)
    return points + q_w * twice_cross + torch.cross(q_xyz, twice_cross, dim=-1)


def _transform_points_se3(extrinsics, points):
    """Apply PyPose SE3.data poses in COLMAP world-to-camera convention."""
    translations = extrinsics[..., :3]
    quaternions_xyzw = extrinsics[..., 3:7]
    rotated = _rotate_points_xyzw(quaternions_xyzw, points)
    return rotated + translations


@map_transform
def colmap_project(points, extrinsics, intrinsics):
    """Project using SE3 extrinsics converted to COLMAP matrices."""
    points_cam = _transform_points_se3(extrinsics, points)
    return _distort_and_project(points_cam, intrinsics)


@map_transform
def colmap_project_fixed_rot(points, translations, rotations, intrinsics):
    """Project with fixed rotations: only translations are optimized."""
    rotated = torch.matmul(rotations, points.unsqueeze(-1)).squeeze(-1)
    points_cam = rotated + translations
    return _distort_and_project(points_cam, intrinsics)


class ColmapReproj(nn.Module):
    """Full BA model (identical to TorchBA's ReprojectionModel)."""

    def __init__(self, extrinsics, intrinsics, points_3d):
        super().__init__()
        self.extrinsics = nn.Parameter(TrackingTensor(extrinsics))
        self.intrinsics = nn.Parameter(TrackingTensor(intrinsics))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.extrinsics.trim_SE3_grad = True

    def forward(self, points_2d, image_indices, camera_indices, point_indices):
        points_proj = colmap_project(
            self.points_3d[point_indices],
            self.extrinsics[image_indices],
            self.intrinsics[camera_indices],
        )
        return points_proj - points_2d

    def loss(self, input, target=None):
        if isinstance(input, dict):
            R = self.forward(**input)
        else:
            R = self.forward(input)
        return (R**2).sum() / 2


class ColmapReprojFixedRot(nn.Module):
    """Fixed-rotation BA model."""

    def __init__(self, translations, intrinsics, points_3d):
        super().__init__()
        self.translations = nn.Parameter(TrackingTensor(translations))
        self.intrinsics = nn.Parameter(TrackingTensor(intrinsics))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))

    def forward(self, points_2d, image_indices, camera_indices,
                point_indices, rotations):
        points_proj = colmap_project_fixed_rot(
            self.points_3d[point_indices],
            self.translations[image_indices],
            rotations[image_indices],
            self.intrinsics[camera_indices],
        )
        return points_proj - points_2d

    def loss(self, input, target=None):
        if isinstance(input, dict):
            R = self.forward(**input)
        else:
            R = self.forward(input)
        return (R**2).sum() / 2


def _compact_remap(indices, total):
    used_mask = np.zeros(total, dtype=bool)
    used_mask[indices] = True
    new2old = np.where(used_mask)[0]
    old2new = np.full(total, -1, dtype=np.int64)
    old2new[new2old] = np.arange(len(new2old))
    return used_mask, old2new, new2old


def _compute_reprojection_errors(
    extrinsics, intrinsics, points_3d, points_2d,
    image_indices, camera_indices, point_indices,
):
    """Compute per-observation reprojection error (in pixels).

    Vectorised on GPU.  Adapted from InstantSFM's
    FilterTracksByReprojectionNormalized: transforms world points into
    camera space via the SE3 pose, applies COLMAP's radial projection,
    and returns the L2 distance to the (already-centred) 2D observation.
    """
    pts = points_3d[point_indices]                    # (N, 3)
    ext = extrinsics[image_indices]                   # (N, 3, 4)
    intr = intrinsics[camera_indices]                 # (N, 3)

    # Apply COLMAP's world-to-camera extrinsics directly in matrix form.
    pts_cam = _transform_points(ext, pts)             # (N, 3)

    # Perspective divide
    z = pts_cam[:, 2:3].clamp(min=1e-8)
    uv = pts_cam[:, :2] / z                           # (N, 2)

    # SIMPLE_RADIAL distortion
    f  = intr[:, 0:1]
    k1 = intr[:, 1:2]
    r2 = (uv * uv).sum(dim=-1, keepdim=True)
    dist = 1.0 + k1 * r2
    proj = uv * dist * f                              # (N, 2)

    return (proj - points_2d).norm(dim=-1)            # (N,)


def _filter_observations_by_reproj(
    extrinsics, intrinsics, points_3d, points_2d,
    image_indices, camera_indices, point_indices,
    max_error,
):
    """Remove observations whose reprojection error exceeds *max_error* px.

    Adapted from InstantSFM ``FilterTracksByReprojectionNormalized``.
    Returns the boolean keep-mask over observations.
    """
    with torch.no_grad():
        errs = _compute_reprojection_errors(
            extrinsics, intrinsics, points_3d, points_2d,
            image_indices, camera_indices, point_indices,
        )
    # Also reject points behind the camera (z <= 0).
    pts_cam_z = _transform_points(
        extrinsics[image_indices], points_3d[point_indices])[:, 2]
    keep = (errs < max_error) & (pts_cam_z > 0)
    return keep


def _log_probe_errors(
    tag,
    extrinsics,
    intrinsics,
    points_3d,
    probe_image_indices_np,
    probe_camera_indices_np,
    probe_point_indices_np,
    probe_points_2d_np,
    probe_labels,
):
    if probe_image_indices_np is None or len(probe_image_indices_np) == 0:
        _log(f"probe {tag}: no probes configured")
        return

    device = extrinsics.device
    probe_img = torch.as_tensor(
        probe_image_indices_np, dtype=torch.long, device=device)
    probe_cam = torch.as_tensor(
        probe_camera_indices_np, dtype=torch.long, device=device)
    probe_pt = torch.as_tensor(
        probe_point_indices_np, dtype=torch.long, device=device)
    probe_obs = torch.as_tensor(
        probe_points_2d_np, dtype=torch.float64, device=device)

    with torch.no_grad():
        errs = _compute_reprojection_errors(
            extrinsics, intrinsics, points_3d, probe_obs,
            probe_img, probe_cam, probe_pt,
        )
        pts_cam = _transform_points(
            extrinsics[probe_img], points_3d[probe_pt])
        depths = pts_cam[:, 2]

    errs_np = errs.detach().cpu().numpy()
    depths_np = depths.detach().cpu().numpy()
    if len(errs_np) == 0:
        _log(f"probe {tag}: no valid probe residuals")
        return

    p50, p90, p100 = np.percentile(errs_np, [50, 90, 100])
    _log(
        f"probe {tag}: n={len(errs_np)} p50={p50:.3f} "
        f"p90={p90:.3f} max={p100:.3f}"
    )
    for i, label in enumerate(probe_labels[:len(errs_np)]):
        _log(
            f"probe {tag} #{i}: {label} "
            f"err_px={errs_np[i]:.3f} depth={depths_np[i]:.6f}"
        )


class _LoggingPCG:
    """Solver wrapper that logs the LM normal-equation relative residual.

    Wraps a `PCG` solver to expose the directly observable convergence
    signal: after the solver returns step `x` from inputs `(A, b)`, we
    compute `|A x - b| / |b|`. If the underlying PCG terminated on its
    tolerance, this ratio is at most the configured `tol`. If PCG hit
    a max-iteration cap or stagnated, the ratio is larger and quantifies
    the inaccuracy of the LM step.

    Iteration count is not reported here because the underlying
    `bae.utils.pysolvers.PCG` does not expose it through the call API,
    and modifying that module is out of scope. The relative residual
    alone tells us whether each LM solve is producing the
    Gauss-Newton step the LM model expects, which is the question
    raised by §3.12 in info.md.
    """

    def __init__(self, base):
        self._base = base

    def __call__(self, A, b):
        x = self._base(A, b)
        with torch.no_grad():
            x_col = x.reshape(-1, 1) if x.dim() == 1 else x
            b_col = b.reshape(-1, 1) if b.dim() == 1 else b
            Ax = A @ x_col
            r = Ax - b_col
            rnorm = float(r.norm())
            bnorm = float(b_col.norm())
            rel = rnorm / bnorm if bnorm > 1e-30 else float("inf")
        _log(
            f"pcg solve: |b|={bnorm:.3e} |Ax-b|={rnorm:.3e} "
            f"rel={rel:.3e}"
        )
        return x


def _log_se3_projection_consistency(
    tag,
    extrinsics,
    intrinsics,
    points_3d,
    probe_image_indices_np,
    probe_camera_indices_np,
    probe_point_indices_np,
    probe_labels,
):
    if probe_image_indices_np is None or len(probe_image_indices_np) == 0:
        _log(f"se3 projector {tag}: no probes configured")
        return

    device = extrinsics.device
    probe_img = torch.as_tensor(
        probe_image_indices_np, dtype=torch.long, device=device)
    probe_cam = torch.as_tensor(
        probe_camera_indices_np, dtype=torch.long, device=device)
    probe_pt = torch.as_tensor(
        probe_point_indices_np, dtype=torch.long, device=device)

    with torch.no_grad():
        pts = points_3d[probe_pt]
        ext = extrinsics[probe_img]
        intr = intrinsics[probe_cam]
        points_cam_quat = rotate_quat(pts, ext)
        points_cam_matrix = _transform_points_se3(ext, pts)
        proj_quat = _distort_and_project(points_cam_quat, intr)
        proj_matrix = _distort_and_project(points_cam_matrix, intr)
        cam_diffs = (points_cam_quat - points_cam_matrix).norm(dim=-1)
        proj_diffs = (proj_quat - proj_matrix).norm(dim=-1)

    cam_diffs_np = cam_diffs.cpu().numpy()
    proj_diffs_np = proj_diffs.cpu().numpy()
    if len(cam_diffs_np) == 0:
        _log(f"se3 projector {tag}: no valid probe residuals")
        return

    cam_p50, cam_p90, cam_max = np.percentile(cam_diffs_np, [50, 90, 100])
    proj_p50, proj_p90, proj_max = np.percentile(proj_diffs_np, [50, 90, 100])
    _log(
        f"se3 projector {tag}: "
        f"cam_diff p50={cam_p50:.6e} p90={cam_p90:.6e} max={cam_max:.6e}  "
        f"proj_diff p50={proj_p50:.6e} p90={proj_p90:.6e} max={proj_max:.6e}"
    )
    for i, label in enumerate(probe_labels[:len(proj_diffs_np)]):
        _log(
            f"se3 projector {tag} #{i}: {label} "
            f"cam_diff={cam_diffs_np[i]:.6e} "
            f"proj_diff={proj_diffs_np[i]:.6e}"
        )


def _log_parameter_drifts(
    extrinsics_before,
    extrinsics_after,
    intrinsics_before,
    intrinsics_after,
    points_before,
    points_after,
    constant_pose_mask_np,
    constant_point_mask_np,
    refine_focal_length,
    refine_extra_params,
):
    constant_pose_mask = constant_pose_mask_np.astype(bool)
    if constant_pose_mask.any():
        trans_before = extrinsics_before[:, :, 3]
        trans_after = extrinsics_after[:, :, 3]
        trans_delta = np.linalg.norm(trans_after - trans_before, axis=1)
        rot_delta = np.linalg.norm(
            extrinsics_after[:, :, :3] - extrinsics_before[:, :, :3],
            axis=(1, 2),
        )
        trans_const = trans_delta[constant_pose_mask]
        rot_const = rot_delta[constant_pose_mask]
        _log(
            "const pose drift: "
            f"n={len(trans_const)} "
            f"t_p50={np.percentile(trans_const, 50):.6e} "
            f"t_max={np.max(trans_const):.6e} "
            f"Rfro_p50={np.percentile(rot_const, 50):.6e} "
            f"Rfro_max={np.max(rot_const):.6e}"
        )

    constant_point_mask = constant_point_mask_np.astype(bool)
    if constant_point_mask.any():
        point_delta = np.linalg.norm(points_after - points_before, axis=1)
        point_const = point_delta[constant_point_mask]
        _log(
            "const point drift: "
            f"n={len(point_const)} "
            f"p50={np.percentile(point_const, 50):.6e} "
            f"max={np.max(point_const):.6e}"
        )

    focal_delta = np.abs(intrinsics_after[:, 0] - intrinsics_before[:, 0])
    extra_delta = np.abs(intrinsics_after[:, 1:] - intrinsics_before[:, 1:])
    if not refine_focal_length:
        _log(
            "focal drift while disabled: "
            f"p50={np.percentile(focal_delta, 50):.6e} "
            f"max={np.max(focal_delta):.6e}"
        )
    if not refine_extra_params:
        extra_norm = np.linalg.norm(extra_delta, axis=1)
        _log(
            "extra-param drift while disabled: "
            f"p50={np.percentile(extra_norm, 50):.6e} "
            f"max={np.max(extra_norm):.6e}"
        )


def _quat_angle_deg_xyzw(quat_before, quat_after):
    q0 = quat_before / np.clip(
        np.linalg.norm(quat_before, axis=1, keepdims=True), 1e-12, None)
    q1 = quat_after / np.clip(
        np.linalg.norm(quat_after, axis=1, keepdims=True), 1e-12, None)
    dots = np.abs(np.sum(q0 * q1, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _log_active_dofs(
    tag,
    constant_rotation,
    intrinsics_before,
    image_indices_active_np,
    camera_indices_active_np,
    point_indices_active_np,
    constant_pose_mask_np,
    constant_point_mask_np,
    refine_focal_length,
    refine_extra_params,
):
    pose_mask = constant_pose_mask_np.astype(bool)
    point_mask = constant_point_mask_np.astype(bool)
    active_img_ids = np.asarray(image_indices_active_np, dtype=np.int64)
    active_cam_ids = np.asarray(camera_indices_active_np, dtype=np.int64)
    active_pt_ids = np.asarray(point_indices_active_np, dtype=np.int64)
    uniq_imgs = np.unique(active_img_ids)
    uniq_cams = np.unique(active_cam_ids)
    uniq_pts = np.unique(active_pt_ids)
    n_imgs = len(uniq_imgs)
    n_pts = len(uniq_pts)
    n_cams = len(uniq_cams)
    variable_imgs = uniq_imgs[~pose_mask[uniq_imgs]]
    variable_pts = uniq_pts[~point_mask[uniq_pts]]
    active_pose_dofs = len(variable_imgs) * (3 if constant_rotation else 6)
    active_intr_dofs = 0
    if refine_focal_length:
        active_intr_dofs += len(uniq_cams)
    if refine_extra_params:
        active_intr_dofs += len(uniq_cams)
    active_point_dofs = len(variable_pts) * 3
    _log(
        f"{tag}: opt_imgs={n_imgs} variable_imgs={len(variable_imgs)} "
        f"opt_cams={n_cams} opt_pts={n_pts} variable_pts={len(variable_pts)} "
        f"pose_dofs={active_pose_dofs} intr_dofs={active_intr_dofs} "
        f"point_dofs={active_point_dofs} "
        f"constant_rotation={constant_rotation} "
        f"refine_focal={refine_focal_length} "
        f"refine_extra={refine_extra_params}"
    )


def _log_parameter_update_stats(
    tag,
    constant_rotation,
    extrinsics_before,
    extrinsics_after,
    intrinsics_before,
    intrinsics_after,
    points_before,
    points_after,
    image_indices_active_np,
    camera_indices_active_np,
    point_indices_active_np,
    constant_pose_mask_np,
    constant_point_mask_np,
    thresholds=(1e-10, 1e-8, 1e-6),
):
    def _fmt(label, values):
        if len(values) == 0:
            return f"{label}: none"
        values = np.asarray(values)
        counts = " ".join(
            f">{thr:.0e}={int((values > thr).sum())}/{len(values)}"
            for thr in thresholds
        )
        return (
            f"{label}: p50={np.percentile(values, 50):.6e} "
            f"p90={np.percentile(values, 90):.6e} "
            f"max={np.max(values):.6e} {counts}"
        )

    uniq_imgs = np.unique(np.asarray(image_indices_active_np, dtype=np.int64))
    uniq_cams = np.unique(np.asarray(camera_indices_active_np, dtype=np.int64))
    uniq_pts = np.unique(np.asarray(point_indices_active_np, dtype=np.int64))
    pose_mask = constant_pose_mask_np.astype(bool)
    point_mask = constant_point_mask_np.astype(bool)
    variable_imgs = uniq_imgs[~pose_mask[uniq_imgs]]
    variable_pts = uniq_pts[~point_mask[uniq_pts]]

    t_before = extrinsics_before[variable_imgs, :, 3]
    t_after = extrinsics_after[variable_imgs, :, 3]
    trans_delta = np.linalg.norm(t_after - t_before, axis=1)
    _log(f"{tag} " + _fmt("translation", trans_delta))

    if not constant_rotation:
        q_before = _extrinsics_matrices_to_se3_data_numpy(
            extrinsics_before[variable_imgs])[:, 3:7]
        q_after = _extrinsics_matrices_to_se3_data_numpy(
            extrinsics_after[variable_imgs])[:, 3:7]
        rot_delta_deg = _quat_angle_deg_xyzw(q_before, q_after)
        _log(f"{tag} " + _fmt("rotation_deg", rot_delta_deg))

    focal_delta = np.abs(
        intrinsics_after[uniq_cams, 0] - intrinsics_before[uniq_cams, 0])
    k1_delta = np.abs(
        intrinsics_after[uniq_cams, 1] - intrinsics_before[uniq_cams, 1])
    _log(f"{tag} " + _fmt("focal", focal_delta))
    _log(f"{tag} " + _fmt("k1", k1_delta))

    point_delta = np.linalg.norm(
        points_after[variable_pts] - points_before[variable_pts], axis=1)
    _log(f"{tag} " + _fmt("point3D", point_delta))


def _parameter_block_numels(params):
    numels = []
    for param in params:
        if not param.requires_grad:
            numels.append(0)
        elif getattr(param, "trim_SE3_grad", False):
            numels.append(int(np.prod(param.shape[:-1])) * (param.shape[-1] - 1))
        else:
            numels.append(param.numel())
    return numels


def _split_step_vector(step, numels):
    splits = []
    offset = 0
    flat = step.reshape(-1)
    for n in numels:
        splits.append(flat[offset:offset + n])
        offset += n
    return splits


def _log_lm_iteration_stats(tag, J_blocks, residual, step_vec, params):
    numels = _parameter_block_numels(params)
    step_blocks = _split_step_vector(step_vec, numels)
    residual_col = residual.view(-1, 1)
    rhs_blocks = []
    for block in J_blocks:
        rhs = -(block.to_sparse_csr().mT @ residual_col).reshape(-1)
        rhs_blocks.append(rhs)

    labels = []
    if len(params) == 3:
        labels = ["pose", "intr", "points"]
    else:
        labels = [f"block{i}" for i in range(len(params))]

    summaries = []
    for label, rhs, step in zip(labels, rhs_blocks, step_blocks):
        rhs_norm = rhs.norm().item() if rhs.numel() > 0 else 0.0
        step_norm = step.norm().item() if step.numel() > 0 else 0.0
        step_max = step.abs().max().item() if step.numel() > 0 else 0.0
        summaries.append(
            f"{label}:|JTr|={rhs_norm:.3e}|D|={step_norm:.3e}|D|max={step_max:.3e}"
        )
    _log(f"{tag} " + "  ".join(summaries))


def _compute_quality(J_blocks, D_blocks, residual, last_loss, new_loss):
    jd = None
    for block, d_block in zip(J_blocks, D_blocks):
        contrib = block.to_sparse_coo() @ d_block.reshape(-1, 1)
        jd = contrib if jd is None else jd + contrib
    residual_col = residual.reshape(-1, 1)
    denom = -(jd.mT @ (2 * residual_col + jd)).reshape(())
    denom_val = float(denom.item()) if torch.is_tensor(denom) else float(denom)
    if abs(denom_val) < 1e-20:
        return float("nan")
    return float((last_loss - new_loss) / denom_val)


def _compute_column_scaling(
    J_coo, eps=1e-12, min_scale=1e-6, max_scale=1e3
):
    J_coo = J_coo.coalesce()
    cols = J_coo.indices()[1]
    vals = J_coo.values()
    diag = torch.zeros(
        J_coo.shape[1], dtype=vals.dtype, device=vals.device)
    diag.scatter_add_(0, cols, vals * vals)
    raw_scale = torch.rsqrt(torch.clamp(diag, min=eps))
    scale = torch.clamp(raw_scale, min=min_scale, max=max_scale)
    return scale, raw_scale


def _apply_column_scaling(J_coo, scale):
    J_coo = J_coo.coalesce()
    rows, cols = J_coo.indices()
    vals = J_coo.values() * scale[cols]
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, J_coo.shape, device=vals.device
    ).coalesce()


def _extrinsics_matrices_to_se3_data_numpy(extrinsics_np):
    se3 = pp.mat2SE3(torch.as_tensor(extrinsics_np, dtype=torch.float64))
    return np.ascontiguousarray(se3.data.cpu().numpy())


def solve(
    extrinsics_np, intrinsics_np, points_3d_np, points_2d_np,
    image_indices_np, camera_indices_np, point_indices_np,
    constant_pose_mask_np, constant_point_mask_np, options_dict,
):
    if not torch.cuda.is_available():
        raise RuntimeError("BAE requires CUDA")
    gpu_index = options_dict.get("gpu_index", "0")
    device = f"cuda:{gpu_index}"
    torch.cuda.empty_cache()

    max_iterations = options_dict.get("max_num_iterations", 100)
    constant_rotation = options_dict.get(
        "constant_rig_from_world_rotation", False,
    )

    extrinsics_full = extrinsics_np.reshape(-1, 3, 4)
    intrinsics_full = intrinsics_np.reshape(-1, 3)
    points_full = points_3d_np.reshape(-1, 3)
    # SIMPLE_RADIAL only: keep the compatibility slot semantically zero.
    intrinsics_full[:, 2] = 0.0
    extrinsics_before = extrinsics_full.copy()
    intrinsics_before = intrinsics_full.copy()
    points_before = points_full.copy()
    refine_focal_length = options_dict.get("refine_focal_length", True)
    refine_extra_params = options_dict.get("refine_extra_params", True)

    probe_image_indices_np = options_dict.get("probe_image_indices")
    probe_camera_indices_np = options_dict.get("probe_camera_indices")
    probe_point_indices_np = options_dict.get("probe_point_indices")
    probe_points_2d_np = options_dict.get("probe_points_2d")
    probe_labels = list(options_dict.get("probe_labels", []))

    n_imgs_orig = extrinsics_full.shape[0]
    n_cams_orig = intrinsics_full.shape[0]
    n_pts_orig = points_full.shape[0]

    # DEBUG: confirm indices arrays from C++ have correct distinct values.
    _log(
        f"DEBUG indices: img_idx shape={image_indices_np.shape} "
        f"strides={image_indices_np.strides} dtype={image_indices_np.dtype} "
        f"unique={len(np.unique(image_indices_np))} "
        f"min={int(image_indices_np.min())} "
        f"max={int(image_indices_np.max())}"
    )
    _log(
        f"DEBUG indices: pt_idx  shape={point_indices_np.shape} "
        f"strides={point_indices_np.strides} dtype={point_indices_np.dtype} "
        f"unique={len(np.unique(point_indices_np))} "
        f"min={int(point_indices_np.min())} "
        f"max={int(point_indices_np.max())}"
    )
    _log(
        f"DEBUG indices: first 10 img_idx={image_indices_np[:10].tolist()}  "
        f"pt_idx={point_indices_np[:10].tolist()}"
    )

    # DEBUG workaround: if strides are wrong, re-read raw buffer with
    # explicit stride.  Will raise if the underlying memory truly is zero
    # (i.e., the bug is not stride-related).
    try:
        raw = np.frombuffer(memoryview(image_indices_np).tobytes(),
                            dtype=np.int32)
        _log(
            f"DEBUG raw img_idx via frombuffer: len={len(raw)} "
            f"unique={len(np.unique(raw))} "
            f"min={int(raw.min())} max={int(raw.max())}  "
            f"first10={raw[:10].tolist()}"
        )
    except Exception as e:
        _log(f"DEBUG frombuffer failed: {e!r}")

    # ------------------------------------------------------------------
    # Pre-BA outlier filtering (adapted from InstantSFM global_mapper
    # line 133: FilterTracksByAngle + FilterTracksByReprojectionNormalized).
    #
    # Compute initial reprojection errors and discard observations with
    # error above a generous threshold.  This removes the gross outliers
    # that would otherwise dominate the least-squares cost and prevent
    # convergence — the single biggest difference between InstantSFM
    # (which filters before BA) and vanilla COLMAP (which does not).
    # ------------------------------------------------------------------
    image_indices_cur = image_indices_np.copy()
    camera_indices_cur = camera_indices_np.copy()
    point_indices_cur = point_indices_np.copy()
    points_2d_cur = points_2d_np.reshape(-1, 2).copy()

    # Compute initial reprojection errors on GPU.
    _ext_t = torch.tensor(extrinsics_full, dtype=torch.float64, device=device)
    _intr_t = torch.tensor(intrinsics_full, dtype=torch.float64, device=device)
    _pts3_t = torch.tensor(points_full, dtype=torch.float64, device=device)
    _pts2_t = torch.tensor(points_2d_cur, dtype=torch.float64, device=device)
    _img_idx = torch.tensor(image_indices_cur, dtype=torch.long, device=device)
    _cam_idx = torch.tensor(camera_indices_cur, dtype=torch.long, device=device)
    _pt_idx = torch.tensor(point_indices_cur, dtype=torch.long, device=device)

    with torch.no_grad():
        all_errs = _compute_reprojection_errors(
            _ext_t, _intr_t, _pts3_t, _pts2_t,
            _img_idx, _cam_idx, _pt_idx,
        )
        pts_z = _transform_points(_ext_t[_img_idx], _pts3_t[_pt_idx])[:, 2]
        valid = pts_z > 0
        # Per-observation focal length (column 0 of intrinsics).  Used to
        # convert pixel residuals to normalized image-plane units —
        # directly comparable to COLMAP's max_normalized_reproj_error
        # (default 1e-2 in global_mapper.h).
        focals = _intr_t[_cam_idx, 0]
        all_norm_np = (all_errs / focals.clamp(min=1e-8)).cpu().numpy()
        all_errs_np = all_errs.cpu().numpy()
        valid_np = valid.cpu().numpy()

    # Log error distribution in both pixel and normalized units so we can
    # compare against COLMAP's filter thresholds (which are normalized).
    valid_errs = all_errs_np[valid_np]
    valid_norm = all_norm_np[valid_np]
    median_err_px = 1.0
    if len(valid_errs) > 0:
        pcts = np.percentile(valid_errs, [10, 25, 50, 75, 90, 95, 99])
        npcts = np.percentile(valid_norm, [10, 25, 50, 75, 90, 95, 99])
        median_err_px = float(pcts[2])
        _log(
            f"init err [px]   p10={pcts[0]:.2f} p25={pcts[1]:.2f} "
            f"p50={pcts[2]:.2f} p75={pcts[3]:.2f} p90={pcts[4]:.2f} "
            f"p95={pcts[5]:.2f} p99={pcts[6]:.2f}  "
            f"behind_cam={int((~valid_np).sum())}  n={len(valid_errs)}"
        )
        _log(
            f"init err [norm] p10={npcts[0]:.2e} p25={npcts[1]:.2e} "
            f"p50={npcts[2]:.2e} p75={npcts[3]:.2e} p90={npcts[4]:.2e} "
            f"p95={npcts[5]:.2e} p99={npcts[6]:.2e}  "
            f"(colmap_filter=1.00e-02)"
        )
    _log_probe_errors(
        "pre_opt",
        _ext_t,
        _intr_t,
        _pts3_t,
        probe_image_indices_np,
        probe_camera_indices_np,
        probe_point_indices_np,
        probe_points_2d_np,
        probe_labels,
    )

    # Huber kernel: fixed delta = 1.0 px to match Ceres' HuberLoss(1.0)
    # used by COLMAP's global mapper.  Earlier we used delta = 2*median
    # (adaptive), but the four-dataset benchmark showed BAE's mean reproj
    # error consistently exceeds Ceres' (1.03x ignatius, 1.38x bridge,
    # 2.05x mihama, 3.24x soil) which causes ~50% point loss on
    # bridge/soil through COLMAP's downstream filter.  Matching Ceres'
    # kernel exactly is the principled apples-to-apples choice.
    kernel_delta = 1.0
    _log(f"kernel: Huber(delta={kernel_delta:.2f} px)")

    # Pre-filter in NORMALIZED image-plane units (residual / focal).
    # Matches COLMAP's filter convention (max_normalized_reproj_error,
    # default 1e-2 in global_mapper.h, applied at {3,2,1}*1e-2 across the
    # 3 outer BA rounds).  0.10 is 10x COLMAP's iter-0 cutoff: generous
    # enough to retain the inlier core, tight enough to reject gross
    # outliers regardless of camera focal length.
    keep_mask = valid_np & (all_norm_np < 0.10)

    n_before = len(image_indices_cur)
    image_indices_cur = image_indices_cur[keep_mask]
    camera_indices_cur = camera_indices_cur[keep_mask]
    point_indices_cur = point_indices_cur[keep_mask]
    points_2d_cur = points_2d_cur[keep_mask]
    _log(
        f"pre-filter: kept {len(image_indices_cur)} / {n_before} "
        f"observations (threshold 1.00e-01 norm)"
    )

    # Track-length distribution log (post norm-filter).  Diagnostic only.
    counts_np = np.bincount(point_indices_cur, minlength=n_pts_orig)
    present_lengths = counts_np[counts_np > 0]
    if len(present_lengths) > 0:
        lpcts = np.percentile(present_lengths, [10, 25, 50, 75, 90, 95, 99])
        lp = lpcts.astype(int)
        _log(
            f"track lengths: p10={lp[0]} p25={lp[1]} p50={lp[2]} "
            f"p75={lp[3]} p90={lp[4]} p95={lp[5]} p99={lp[6]}  "
            f"n_tracks={len(present_lengths)}"
        )

    # Note: tri-angle filter removed (was 0.5° max-from-mean cone drop).
    # Four-dataset benchmark showed it had no measurable effect on point
    # retention (the 1.5°→0.5° change moved soil retention 47.4%→46.7%),
    # confirming it was not the cause of point loss.  Ceres BA has no
    # analogous pre-filter inside the solver — it relies on the Huber
    # kernel for down-weighting and COLMAP's downstream filter for hard
    # rejection.  We now match that architecture.

    del _ext_t, _intr_t, _pts3_t, _pts2_t, _img_idx, _cam_idx, _pt_idx
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Helper: build compact arrays, model, optimizer, and input_data
    # from the current observation arrays.
    # ------------------------------------------------------------------
    def _build_problem(img_idx_np, cam_idx_np, pt_idx_np, pts2d_np):
        """Return (model, optimizer, input_data, remap_info)."""
        ig_u, ig_o2n, ig_n2o = _compact_remap(img_idx_np, n_imgs_orig)
        cm_u, cm_o2n, cm_n2o = _compact_remap(cam_idx_np, n_cams_orig)
        pt_u, pt_o2n, pt_n2o = _compact_remap(pt_idx_np, n_pts_orig)

        ext_c = extrinsics_full[ig_n2o]
        intr_c = intrinsics_full[cm_n2o]
        pts_c = points_full[pt_n2o]
        intr_c[:, 2] = 0.0

        ii = ig_o2n[img_idx_np]
        ci = cm_o2n[cam_idx_np]
        pi = pt_o2n[pt_idx_np]

        n_i, n_c, n_p, n_o = len(ig_n2o), len(cm_n2o), len(pt_n2o), len(ii)
        _log(
            f"problem: {n_i} imgs, {n_c} cams, {n_p} pts, {n_o} obs, "
            f"const_rot={constant_rotation}"
        )
        if n_o == 0:
            return None, None, None, (ig_n2o, cm_n2o, pt_n2o)

        # Build the optimizer's intrinsics as a 2-column tensor (f, k1)
        # for SIMPLE_RADIAL.  The C++-side buffer is 3-wide for forward
        # compatibility with RADIAL (f, k1, k2), but BAE's projection
        # currently only references k1.  Carrying a third column through
        # `nn.Parameter` produces a permanently-zero Jacobian column,
        # which: (a) clamps `_compute_column_scaling`'s `raw_max` at the
        # `eps`-floor (~1e6) and forces the `max_scale=1e3` cap to bind
        # every iter, (b) puts a structural zero into JᵀJ at that DOF.
        # Excluding the slot from the optimizer eliminates both.
        intr_t = torch.tensor(
            intr_c[:, :2], dtype=torch.float64, device=device)
        p3_t = torch.tensor(pts_c, dtype=torch.float64, device=device)
        p2_t = torch.tensor(pts2d_np, dtype=torch.float64, device=device)
        ii_t = torch.tensor(ii, dtype=torch.int32, device=device)
        ci_t = torch.tensor(ci, dtype=torch.int32, device=device)
        pi_t = torch.tensor(pi, dtype=torch.int32, device=device)

        if constant_rotation:
            tr_t = torch.tensor(
                ext_c[:, :, 3], dtype=torch.float64, device=device)
            ro_t = torch.tensor(
                ext_c[:, :, :3], dtype=torch.float64, device=device)
            mdl = ColmapReprojFixedRot(tr_t, intr_t, p3_t).to(device)
            inp = {"points_2d": p2_t, "image_indices": ii_t,
                   "camera_indices": ci_t, "point_indices": pi_t,
                   "rotations": ro_t}
            # Fixed-rotation BA has been stable with a large trust-region
            # radius / low initial damping.
            strat = pp.optim.strategy.TrustRegion(
                radius=1e4, max=1e10, up=2.0, down=0.5**4)
        else:
            ex_t = pp.mat2SE3(
                torch.tensor(ext_c, dtype=torch.float64, device=device))
            mdl = ColmapReproj(ex_t, intr_t, p3_t).to(device)
            inp = {"points_2d": p2_t, "image_indices": ii_t,
                   "camera_indices": ci_t, "point_indices": pi_t}
            # Full BA's first joint step is consistently over-aggressive:
            # the logs show several catastrophic rejects before damping
            # reaches ~3.3 and the step becomes acceptable. Start closer to
            # that regime and adapt less violently than the default schedule.
            strat = pp.optim.strategy.TrustRegion(
                radius=0.3, min=1e-6, max=1e10, up=2.0, down=0.5)
        # Wrap PCG so each LM normal-equation solve emits its achieved
        # |Ax - b| / |b|.  See `_LoggingPCG` for rationale.
        slvr = _LoggingPCG(PCG(tol=1e-5))
        kernel = Huber(delta=kernel_delta)
        opt = LM(mdl, strategy=strat, solver=slvr, kernel=kernel, reject=30)
        return mdl, opt, inp, (ig_n2o, cm_n2o, pt_n2o)

    # ------------------------------------------------------------------
    # Helper: run one round of LM optimisation (InstantSFM convergence).
    # ------------------------------------------------------------------
    def _run_ba(mdl, opt, inp, max_iters):
        window_size = 4
        func_tol = 5e-4
        loss_hist = []
        n_it = 0

        @torch.no_grad()
        def _debug_step():
            for pg in opt.param_groups:
                residual = list(opt.model(inp))[0]
                j_blocks = jacobian(residual, pg["params"])
                if isinstance(residual, TrackingTensor):
                    residual = residual.tensor()
                residual = residual.detach()
                j_blocks = [j.detach().to_sparse_coo() for j in j_blocks]
                J_coo_unscaled = torch.cat(j_blocks, dim=-1).coalesce()
                J_coo = J_coo_unscaled
                scale = None
                if not constant_rotation:
                    scale, raw_scale = _compute_column_scaling(J_coo)
                    J_coo = _apply_column_scaling(J_coo, scale)
                    scale_np = scale.cpu().numpy()
                    raw_scale_np = raw_scale.cpu().numpy()
                    _log(
                        "lm scaling: "
                        f"raw_p90={np.percentile(raw_scale_np, 90):.3e} "
                        f"raw_max={np.max(raw_scale_np):.3e} "
                        f"p10={np.percentile(scale_np, 10):.3e} "
                        f"p50={np.percentile(scale_np, 50):.3e} "
                        f"p90={np.percentile(scale_np, 90):.3e} "
                        f"max={np.max(scale_np):.3e}"
                    )
                J = J_coo.to_sparse_csr()
                J_T = J.mT.to_sparse_csr()
                J_unscaled = J_coo_unscaled.to_sparse_csr()

                last_loss = (
                    opt.loss if hasattr(opt, "loss")
                    else opt.model.loss(inp, None)
                )
                last_loss = last_loss.detach() if torch.is_tensor(last_loss) else last_loss
                opt.last = opt.loss = last_loss
                opt.reject_count = 0

                A = opt.mm(J_T, J)
                diagonal_op_(A, op=partial(torch.clamp_, min=pg["min"], max=pg["max"]))

                attempt = 0
                accepted = False
                while opt.last <= opt.loss:
                    attempt += 1
                    damping_before = float(pg["damping"])
                    diagonal_op_(A, op=partial(torch.mul, other=1 + pg["damping"]))
                    rhs = -(J_T @ residual.view(-1, 1))
                    try:
                        step = opt.solver(A, rhs)
                        step = step[:, None]
                        if scale is not None:
                            step = step * scale.reshape(-1, 1)
                    except Exception as e:
                        _log(
                            f"lm iter {n_it + 1:3d} attempt {attempt}: "
                            f"linear solver failed: {e!r}"
                        )
                        break

                    _log_lm_iteration_stats(
                        f"lm iter {n_it + 1:3d} attempt {attempt}",
                        j_blocks,
                        residual,
                        step,
                        pg["params"],
                    )

                    opt.update_parameter(pg["params"], step)
                    new_loss = opt.model.loss(inp, None)
                    new_loss = new_loss.detach() if torch.is_tensor(new_loss) else new_loss
                    d_blocks = _split_step_vector(
                        step, _parameter_block_numels(pg["params"]))
                    quality = _compute_quality(
                        j_blocks, d_blocks, residual.view(-1, 1),
                        float(opt.last), float(new_loss))
                    opt.loss = new_loss
                    opt.strategy.update(
                        pg, last=opt.last, loss=opt.loss, J=J_unscaled,
                        D=step, R=residual.view(-1, 1))
                    rejected = bool(opt.last < opt.loss and opt.reject_count < opt.reject)
                    damping_after = float(pg["damping"])
                    _log(
                        f"lm iter {n_it + 1:3d} attempt {attempt}: "
                        f"last={float(opt.last):.6f} "
                        f"new={float(new_loss):.6f} "
                        f"quality={quality:.3e} "
                        f"damping={damping_before:.3e}->{damping_after:.3e} "
                        f"reject_count={opt.reject_count} "
                        f"accepted={not rejected}"
                    )
                    if rejected:
                        opt.update_parameter(params=pg["params"], step=-step)
                        opt.loss = opt.last
                        opt.reject_count += 1
                    else:
                        accepted = True
                        break
                if not accepted and attempt > 0:
                    _log(
                        f"lm iter {n_it + 1:3d}: no accepted step after "
                        f"{attempt} attempt(s)"
                    )
            return opt.loss

        for _ in range(max_iters):
            loss = _debug_step()
            n_it += 1
            loss_hist.append(loss.item())
            _log(f"iter {n_it:3d}  cost={loss_hist[-1]:.6f}")
            if len(loss_hist) >= 2 * window_size:
                avg_r = sum(loss_hist[-window_size:]) / window_size
                avg_p = sum(
                    loss_hist[-2*window_size:-window_size]) / window_size
                imp = (avg_p - avg_r) / avg_p
                if abs(imp) < func_tol:
                    break
                if loss_hist[-1] == loss_hist[-2]:
                    break
        return n_it, loss_hist

    # ------------------------------------------------------------------
    # Iterative optimise → filter loop (InstantSFM global_mapper L147-151).
    #
    #   for iter in range(3):
    #       ba_engine.Solve(...)
    #       FilterTracksByReprojectionNormalized(
    #           ..., max_reproj_error * max(1, 3 - iter))
    #
    # Single BA round on the pre-filtered data.  COLMAP's outer loop
    # already handles iterative filter→re-optimise (3 iterations with
    # decreasing thresholds), so we don't duplicate that here.
    # ------------------------------------------------------------------
    model, optimizer, input_data, remap = _build_problem(
        image_indices_cur, camera_indices_cur,
        point_indices_cur, points_2d_cur,
    )
    if model is None:
        _log("no observations after pre-filter, skipping.")
        return {
            "extrinsics": extrinsics_full,
            "extrinsics_se3_data": (
                None if constant_rotation else
                _extrinsics_matrices_to_se3_data_numpy(extrinsics_full)
            ),
            "intrinsics": intrinsics_full,
            "points_3d": points_full,
            "num_iterations": 0,
            "initial_cost": 0.0,
            "final_cost": 0.0,
            "converged": True,
        }

    _log_active_dofs(
        "active dofs",
        constant_rotation,
        intrinsics_before,
        image_indices_cur,
        camera_indices_cur,
        point_indices_cur,
        constant_pose_mask_np,
        constant_point_mask_np,
        refine_focal_length,
        refine_extra_params,
    )
    initial_cost = model.loss(input_data, None).item()
    _log(
        f"initial cost={initial_cost:.6f}, obs={len(image_indices_cur)}"
    )

    n_it, loss_hist = _run_ba(model, optimizer, input_data, max_iterations)

    # Write optimised params back.
    ig_n2o, cm_n2o, pt_n2o = remap
    optimized_extrinsics_se3_full = None
    if constant_rotation:
        extrinsics_full[ig_n2o, :, 3] = (
            model.translations.data.cpu().numpy())
    else:
        assert model.extrinsics.data.shape[-1] == 7, (
            "Expected PyPose SE3 parameters to have trailing dimension 7, "
            f"got {tuple(model.extrinsics.data.shape)}")
        optimized_extrinsics_se3 = model.extrinsics.data.contiguous()
        # `matrix()[:, :3, :]` is a strided view. Make it contiguous before
        # converting to NumPy so the C++ side can safely memcpy the result.
        optimized_extrinsics = (
            pp.SE3(optimized_extrinsics_se3).matrix()[:, :3, :].contiguous())
        extrinsics_full[ig_n2o] = optimized_extrinsics.cpu().numpy()
        optimized_extrinsics_se3_full = (
            _extrinsics_matrices_to_se3_data_numpy(extrinsics_full))
        optimized_extrinsics_se3_full[ig_n2o] = np.ascontiguousarray(
            optimized_extrinsics_se3.cpu().numpy())
    # `model.intrinsics` is now (n_cams_compact, 2) holding (f, k1).
    # Slice into the 3-column shared buffer that the C++ side reads;
    # the third column is left at the SIMPLE_RADIAL-compatible zero.
    intrinsics_full[cm_n2o, :2] = model.intrinsics.data.cpu().numpy()
    intrinsics_full[:, 2] = 0.0
    points_full[pt_n2o] = model.points_3d.data.cpu().numpy()

    final_cost = loss_hist[-1] if loss_hist else initial_cost
    _log(
        f"finished: {n_it} iters, cost {initial_cost:.6f} -> "
        f"{final_cost:.6f}"
    )
    _log_parameter_update_stats(
        "update stats",
        constant_rotation,
        extrinsics_before,
        extrinsics_full,
        intrinsics_before,
        intrinsics_full,
        points_before,
        points_full,
        image_indices_cur,
        camera_indices_cur,
        point_indices_cur,
        constant_pose_mask_np,
        constant_point_mask_np,
    )

    # Recompute residual distribution after BA so we can see how much
    # the optimizer actually moved residuals (in pixels and normalized).
    post_ext = torch.tensor(
        extrinsics_full, dtype=torch.float64, device=device)
    post_intr = torch.tensor(
        intrinsics_full, dtype=torch.float64, device=device)
    post_pts3 = torch.tensor(
        points_full, dtype=torch.float64, device=device)
    post_pts2 = torch.tensor(
        points_2d_cur, dtype=torch.float64, device=device)
    post_ii = torch.tensor(
        image_indices_cur, dtype=torch.long, device=device)
    post_ci = torch.tensor(
        camera_indices_cur, dtype=torch.long, device=device)
    post_pi = torch.tensor(
        point_indices_cur, dtype=torch.long, device=device)
    with torch.no_grad():
        post_errs = _compute_reprojection_errors(
            post_ext, post_intr, post_pts3, post_pts2,
            post_ii, post_ci, post_pi,
        )
        post_focals = post_intr[post_ci, 0].clamp(min=1e-8)
        post_norm = (post_errs / post_focals).cpu().numpy()
        post_errs_np = post_errs.cpu().numpy()
    if len(post_errs_np) > 0:
        ppx = np.percentile(post_errs_np, [50, 90, 99])
        pn = np.percentile(post_norm, [50, 90, 99])
        _log(
            f"post err [px]   p50={ppx[0]:.2f} p90={ppx[1]:.2f} "
            f"p99={ppx[2]:.2f}"
        )
        _log(
            f"post err [norm] p50={pn[0]:.2e} p90={pn[1]:.2e} "
            f"p99={pn[2]:.2e}  (colmap_filter=1.00e-02)"
        )
    _log_probe_errors(
        "post_opt",
        post_ext,
        post_intr,
        post_pts3,
        probe_image_indices_np,
        probe_camera_indices_np,
        probe_point_indices_np,
        probe_points_2d_np,
        probe_labels,
    )
    if not constant_rotation:
        _log_se3_projection_consistency(
            "post_opt",
            model.extrinsics.data,
            model.intrinsics.data,
            model.points_3d.data,
            probe_image_indices_np,
            probe_camera_indices_np,
            probe_point_indices_np,
            probe_labels,
        )
    _log_parameter_drifts(
        extrinsics_before,
        extrinsics_full,
        intrinsics_before,
        intrinsics_full,
        points_before,
        points_full,
        constant_pose_mask_np,
        constant_point_mask_np,
        refine_focal_length,
        refine_extra_params,
    )
    del post_ext, post_intr, post_pts3, post_pts2, post_ii, post_ci, post_pi
    torch.cuda.empty_cache()

    return {
        "extrinsics": np.ascontiguousarray(extrinsics_full),
        "extrinsics_se3_data": (
            optimized_extrinsics_se3_full
        ),
        "intrinsics": np.ascontiguousarray(intrinsics_full),
        "points_3d": np.ascontiguousarray(points_full),
        "num_iterations": n_it,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "converged": True,
    }
