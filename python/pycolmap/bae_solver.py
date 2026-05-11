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


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


_FULL_BA_SOLVE_COUNT = 0
_SE3_TANGENT_DIAG_LOGGED = False


def _make_trust_region_strategy(constant_rotation):
    global _FULL_BA_SOLVE_COUNT

    if constant_rotation:
        radius = _env_float("COLMAP_BAE_FIXED_ROT_RADIUS", 1e4)
        max_radius = _env_float("COLMAP_BAE_FIXED_ROT_MAX_RADIUS", 1e10)
        up = _env_float("COLMAP_BAE_FIXED_ROT_TR_UP", 2.0)
        down = _env_float("COLMAP_BAE_FIXED_ROT_TR_DOWN", 0.5**4)
        _log(
            "trust-region fixed_rot: "
            f"radius={radius:.3e} max={max_radius:.3e} "
            f"up={up:.3e} down={down:.3e}"
        )
        return pp.optim.strategy.TrustRegion(
            radius=radius, max=max_radius, up=up, down=down
        )

    solve_idx = _FULL_BA_SOLVE_COUNT
    _FULL_BA_SOLVE_COUNT += 1
    apply_overrides = True
    if _env_bool("COLMAP_BAE_FULL_BA_OVERRIDE_FIRST_ONLY", False):
        apply_overrides = solve_idx == 0

    default_radius = 0.3
    default_min_radius = 1e-6
    default_max_radius = 1e10
    default_up = 2.0
    default_down = 0.5

    if apply_overrides:
        radius = _env_float("COLMAP_BAE_FULL_BA_RADIUS", default_radius)
        min_radius = _env_float(
            "COLMAP_BAE_FULL_BA_MIN_RADIUS", default_min_radius
        )
        max_radius = _env_float(
            "COLMAP_BAE_FULL_BA_MAX_RADIUS", default_max_radius
        )
        up = _env_float("COLMAP_BAE_FULL_BA_TR_UP", default_up)
        down = _env_float("COLMAP_BAE_FULL_BA_TR_DOWN", default_down)
    else:
        radius = default_radius
        min_radius = default_min_radius
        max_radius = default_max_radius
        up = default_up
        down = default_down

    _log(
        "trust-region full_ba: "
        f"solve_idx={solve_idx} apply_overrides={apply_overrides} "
        f"radius={radius:.3e} min={min_radius:.3e} max={max_radius:.3e} "
        f"up={up:.3e} down={down:.3e}"
    )
    return pp.optim.strategy.TrustRegion(
        radius=radius, min=min_radius, max=max_radius, up=up, down=down
    )


def _log_se3_tangent_layout_once():
    """Log which pp.se3 tangent coordinates move translation vs rotation.

    The stationary gauge fix freezes tangent-space DoFs, so we want a
    ground-truth log of which tangent indices correspond to translation
    in the specific PyPose build we're running against.
    """
    global _SE3_TANGENT_DIAG_LOGGED
    if _SE3_TANGENT_DIAG_LOGGED:
        return
    _SE3_TANGENT_DIAG_LOGGED = True

    eps = 1e-4
    base = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                        dtype=torch.float64)
    lines = []
    for i in range(6):
        step = torch.zeros((1, 6), dtype=torch.float64)
        step[0, i] = eps
        moved = pp.SE3(base.clone()).add_(pp.se3(step))
        matrix = moved.matrix()[0, :3, :]
        translation = matrix[:, 3].cpu().numpy()
        rotation = matrix[:, :3].cpu().numpy()
        trace = float(np.trace(rotation))
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        rot_deg = float(np.degrees(np.arccos(cos_theta)))
        lines.append(
            f"d{i}: t=[{translation[0]:+.3e}, {translation[1]:+.3e}, "
            f"{translation[2]:+.3e}] rot_deg={rot_deg:.3e}"
        )
    _log("se3 tangent layout: " + " | ".join(lines))


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


def _build_compact_constraint_masks(
    ig_n2o,
    cm_n2o,
    pt_n2o,
    constant_pose_mask_np,
    constant_point_mask_np,
    refine_focal_length,
    refine_extra_params,
    constant_rotation,
    options_dict,
):
    pose_dofs_per_img = 3 if constant_rotation else 6
    pose_active = np.ones(len(ig_n2o) * pose_dofs_per_img, dtype=bool)
    intr_active = np.ones(len(cm_n2o) * 2, dtype=bool)
    point_active = np.ones(len(pt_n2o) * 3, dtype=bool)

    compact_img_map = {
        int(orig_idx): int(local_idx) for local_idx, orig_idx in enumerate(ig_n2o)
    }
    compact_pt_map = {
        int(orig_idx): int(local_idx) for local_idx, orig_idx in enumerate(pt_n2o)
    }

    compact_const_pose = constant_pose_mask_np[np.asarray(ig_n2o, dtype=np.int64)]
    for local_idx, is_const in enumerate(compact_const_pose.astype(bool)):
        if not is_const:
            continue
        start = local_idx * pose_dofs_per_img
        pose_active[start:start + pose_dofs_per_img] = False

    compact_const_point = constant_point_mask_np[np.asarray(pt_n2o, dtype=np.int64)]
    for local_idx, is_const in enumerate(compact_const_point.astype(bool)):
        if not is_const:
            continue
        start = local_idx * 3
        point_active[start:start + 3] = False

    if not refine_focal_length:
        intr_active[0::2] = False
    if not refine_extra_params:
        intr_active[1::2] = False

    gauge_mode = options_dict.get("gauge_mode", "none")
    gauge_already_fixed = bool(options_dict.get("gauge_already_fixed", False))
    gauge_summary = [f"mode={gauge_mode}"]

    if gauge_already_fixed:
        gauge_summary.append("already_fixed=True")
        _log("gauge constraints: " + " ".join(gauge_summary))
        return {
            "pose_active": pose_active,
            "intr_active": intr_active,
            "point_active": point_active,
        }

    gauge_mode_norm = str(gauge_mode).strip().lower()

    if gauge_mode_norm == "two_cams_from_world":
        anchor_img_orig = int(options_dict.get("gauge_anchor_image_idx", -1))
        second_img_orig = int(options_dict.get("gauge_second_image_idx", -1))
        anchor_image_id = int(options_dict.get("gauge_anchor_image_id", -1))
        second_image_id = int(options_dict.get("gauge_second_image_id", -1))
        anchor_frame_id = int(options_dict.get("gauge_anchor_frame_id", -1))
        second_frame_id = int(options_dict.get("gauge_second_frame_id", -1))
        second_dim = int(options_dict.get("gauge_second_translation_dim", -1))
        baseline_norm = float(options_dict.get("gauge_baseline_norm", 0.0))
        baseline_locked_component = float(
            options_dict.get("gauge_baseline_locked_component", 0.0)
        )

        anchor_local = compact_img_map.get(anchor_img_orig, -1)
        if anchor_local >= 0:
            start = anchor_local * pose_dofs_per_img
            pose_active[start:start + pose_dofs_per_img] = False
            gauge_summary.append(
                f"anchor_img_orig={anchor_img_orig} anchor_local={anchor_local} "
                f"anchor_image_id={anchor_image_id} anchor_frame_id={anchor_frame_id}"
            )
        else:
            gauge_summary.append(
                f"anchor_img_orig={anchor_img_orig} anchor_local=skip "
                f"anchor_image_id={anchor_image_id} anchor_frame_id={anchor_frame_id}"
            )

        second_local = compact_img_map.get(second_img_orig, -1)
        if second_local >= 0 and 0 <= second_dim < 3:
            pose_active[second_local * pose_dofs_per_img + second_dim] = False
            gauge_summary.append(
                f"second_img_orig={second_img_orig} second_local={second_local} "
                f"second_image_id={second_image_id} second_frame_id={second_frame_id} "
                f"fixed_t_dim={second_dim} baseline_norm={baseline_norm:.6g} "
                f"locked_component_abs={baseline_locked_component:.6g}"
            )
        else:
            gauge_summary.append(
                f"second_img_orig={second_img_orig} second_local=skip "
                f"second_image_id={second_image_id} second_frame_id={second_frame_id} "
                f"fixed_t_dim={second_dim} baseline_norm={baseline_norm:.6g} "
                f"locked_component_abs={baseline_locked_component:.6g}"
            )
    elif gauge_mode_norm == "three_points":
        point_indices = np.asarray(
            options_dict.get("gauge_point_indices", []), dtype=np.int64
        )
        fixed_local = []
        for point_orig in point_indices.tolist():
            point_local = compact_pt_map.get(int(point_orig), -1)
            if point_local >= 0:
                start = point_local * 3
                point_active[start:start + 3] = False
                fixed_local.append(point_local)
        gauge_summary.append(
            f"fixed_points_orig={point_indices.tolist()} fixed_points_local={fixed_local}"
        )
    else:
        gauge_summary.append("no_extra_gauge_constraints")

    _log("gauge constraints: " + " ".join(gauge_summary))

    return {
        "pose_active": pose_active,
        "intr_active": intr_active,
        "point_active": point_active,
    }


def _compress_sparse_block_columns(block, active_mask):
    active_mask = torch.as_tensor(
        active_mask, dtype=torch.bool, device=block.device).reshape(-1)
    ncols = int(block.shape[1])
    if active_mask.numel() != ncols:
        raise ValueError(
            f"Active-mask length {active_mask.numel()} != block cols {ncols}"
        )
    active_count = int(active_mask.sum().item())
    if active_count == ncols:
        return block.coalesce(), active_mask, None

    block = block.coalesce()
    indices = block.indices()
    rows = indices[0]
    cols = indices[1]
    keep_nz = active_mask[cols]
    if keep_nz.any():
        active_pos = torch.cumsum(active_mask.to(torch.int64), dim=0) - 1
        new_cols = active_pos[cols[keep_nz]]
        new_indices = torch.stack([rows[keep_nz], new_cols])
        new_values = block.values()[keep_nz]
    else:
        new_indices = torch.empty(
            (2, 0), dtype=torch.long, device=block.device)
        new_values = torch.empty(
            (0,), dtype=block.values().dtype, device=block.device)
    compressed = torch.sparse_coo_tensor(
        new_indices,
        new_values,
        (block.shape[0], active_count),
        device=block.device,
        dtype=block.dtype,
    ).coalesce()
    return compressed, active_mask, active_count


def _expand_step_blocks_to_full(step_blocks_reduced, block_specs):
    expanded = []
    for step_block, spec in zip(step_blocks_reduced, block_specs):
        active_mask = spec["active_mask"]
        full_numel = int(active_mask.numel())
        full_block = torch.zeros(
            full_numel, dtype=step_block.dtype, device=step_block.device)
        if spec["active_count"] > 0:
            full_block[active_mask] = step_block.reshape(-1)
        expanded.append(full_block)
    return expanded


def _cat_sparse_blocks(blocks):
    if len(blocks) == 1:
        return blocks[0].coalesce()
    return torch.cat([block.coalesce() for block in blocks], dim=-1).coalesce()


def _log_lm_iteration_stats(
    tag, J_blocks, residual, step_vec, params, block_numels=None
):
    if block_numels is None:
        block_numels = _parameter_block_numels(params)
    step_blocks = _split_step_vector(step_vec, block_numels)
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


def _log_jtj_diag_stats(tag, J_blocks, params):
    if len(params) == 3:
        labels = ["pose", "intr", "points"]
    else:
        labels = [f"block{i}" for i in range(len(params))]

    summaries = []
    for label, block in zip(labels, J_blocks):
        block = block.coalesce()
        if block.shape[1] == 0 or block._nnz() == 0:
            summaries.append(f"{label}:diag(JTJ)=empty")
            continue
        cols = block.indices()[1]
        vals = block.values()
        diag = torch.zeros(
            block.shape[1], dtype=vals.dtype, device=vals.device
        )
        diag.scatter_add_(0, cols, vals * vals)
        diag_np = diag.detach().cpu().numpy()
        summaries.append(
            f"{label}:diag(JTJ)"
            f"[p10={np.percentile(diag_np, 10):.3e}"
            f" p50={np.percentile(diag_np, 50):.3e}"
            f" p90={np.percentile(diag_np, 90):.3e}"
            f" max={np.max(diag_np):.3e}]"
        )
    _log(f"{tag} " + "  ".join(summaries))


def _compute_quality_terms(J_blocks, D_blocks, residual, last_loss, new_loss):
    jd = None
    for block, d_block in zip(J_blocks, D_blocks):
        contrib = block.to_sparse_coo() @ d_block.reshape(-1, 1)
        jd = contrib if jd is None else jd + contrib
    residual_col = residual.reshape(-1, 1)
    denom = -(jd.mT @ (2 * residual_col + jd)).reshape(())
    denom_val = float(denom.item()) if torch.is_tensor(denom) else float(denom)
    actual_reduction = float(last_loss - new_loss)
    if abs(denom_val) < 1e-20:
        quality = float("nan")
    else:
        quality = float(actual_reduction / denom_val)
    return {
        "actual_reduction": actual_reduction,
        "predicted_reduction": denom_val,
        "quality": quality,
    }


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


def _huber_rho_and_weight(s, delta):
    """Per-observation Huber rho(s) and sqrt-of-derivative weight.

    s: (N,) squared residual norms.
    delta: Huber threshold.

    Returns rho_per_obs (N,) and weight (N,):
      inlier  s <= delta^2: rho = s,                weight = 1
      outlier s >  delta^2: rho = 2*delta*sqrt(s) - delta^2,
                            weight = sqrt(delta / sqrt(s))
    """
    d2 = delta * delta
    inlier = s <= d2
    sqrt_s = torch.sqrt(torch.clamp(s, min=1e-30))
    rho_per_obs = torch.where(inlier, s, 2.0 * delta * sqrt_s - d2)
    rho_prime = torch.where(inlier, torch.ones_like(s), delta / sqrt_s)
    weight = torch.sqrt(rho_prime)
    return rho_per_obs, weight


def _apply_huber_correction(residual_2d, j_blocks, delta):
    """Triggs FastTriggs (square-rooted kernel) correction for (R, J).

    The bae library's `LM.step()` silently drops the configured robust
    kernel when assembling the GN normal equations: raw R and J flow
    into PCG unweighted. On bridge, kushimoto, mihama (residuals span
    1–10 px) this makes the L2 step target outlier reduction while
    the inlier p50 stays flat. Applying the IRLS reweighting Ceres
    uses internally for HuberLoss to (R, J) — *not* to the model.loss
    used for accept/reject, which pypose's RobustModel wrapper already
    handles kernel-side — fixes the step direction. See info.md §3.31.

    Args:
        residual_2d: (N, 2) 2D reprojection residuals.
        j_blocks: list of sparse COO Jacobians, each (2N, n_dofs_block).
        delta: Huber threshold.

    Returns:
        residual_weighted: (N, 2) — kernel-weighted residual.
        j_blocks_weighted: list of sparse COO — kernel-weighted Jacobians.
    """
    s = (residual_2d * residual_2d).sum(dim=-1)  # (N,)
    _, w = _huber_rho_and_weight(s, delta)

    residual_weighted = residual_2d * w.unsqueeze(-1)  # (N, 2)

    # Each 2D residual r_i corresponds to two rows of J (rows 2i, 2i+1).
    # Both rows get the same weight w_i.
    w_per_row = w.unsqueeze(-1).expand(-1, 2).reshape(-1)  # (2N,)
    j_blocks_weighted = []
    for j in j_blocks:
        j = j.coalesce()
        rows = j.indices()[0]
        scaled_values = j.values() * w_per_row[rows]
        j_weighted = torch.sparse_coo_tensor(
            j.indices(),
            scaled_values,
            j.shape,
            device=j.device,
            dtype=j.dtype,
        ).coalesce()
        j_blocks_weighted.append(j_weighted)

    return residual_weighted, j_blocks_weighted


def solve(
    extrinsics_np, intrinsics_np, points_3d_np, points_2d_np,
    image_indices_np, camera_indices_np, point_indices_np,
    constant_pose_mask_np, constant_point_mask_np, options_dict,
):
    _log_se3_tangent_layout_once()
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
            return (
                None,
                None,
                None,
                (ig_n2o, cm_n2o, pt_n2o),
                {
                    "pose_active": np.zeros(
                        n_i * (3 if constant_rotation else 6), dtype=bool),
                    "intr_active": np.zeros(n_c * 2, dtype=bool),
                    "point_active": np.zeros(n_p * 3, dtype=bool),
                },
            )

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
        else:
            ex_t = pp.mat2SE3(
                torch.tensor(ext_c, dtype=torch.float64, device=device))
            mdl = ColmapReproj(ex_t, intr_t, p3_t).to(device)
            inp = {"points_2d": p2_t, "image_indices": ii_t,
                   "camera_indices": ci_t, "point_indices": pi_t}
        constraint_masks = _build_compact_constraint_masks(
            ig_n2o,
            cm_n2o,
            pt_n2o,
            constant_pose_mask_np,
            constant_point_mask_np,
            refine_focal_length,
            refine_extra_params,
            constant_rotation,
            options_dict,
        )

        strat = _make_trust_region_strategy(constant_rotation)
        # Wrap PCG so each LM normal-equation solve emits its achieved
        # |Ax - b| / |b|.  See `_LoggingPCG` for rationale.
        slvr = _LoggingPCG(PCG(tol=1e-5))
        kernel = Huber(delta=kernel_delta)
        reject_cap = _env_int("COLMAP_BAE_LM_REJECT_CAP", 30)
        _log(f"lm reject cap: {reject_cap}")
        opt = LM(
            mdl, strategy=strat, solver=slvr, kernel=kernel, reject=reject_cap
        )
        return mdl, opt, inp, (ig_n2o, cm_n2o, pt_n2o), constraint_masks

    # ------------------------------------------------------------------
    # Helper: run one round of LM optimisation (InstantSFM convergence).
    # ------------------------------------------------------------------
    def _run_ba(mdl, opt, inp, max_iters, constraint_masks):
        window_size = 4
        # Tightened 5e-4 -> 5e-5: on kushimoto every BA call exited at 8-18
        # iters via the previous threshold with cost still descending at
        # ~3-5e-4 per window, while Ceres on the same input kept descending
        # for ~3 more orders of magnitude in cost. Bridge was iter-cap-bound
        # so this change is a no-op there; kushimoto/mihama benefit. See
        # info.md §3.31 / kushimoto run analysis.
        func_tol = 5e-5
        loss_hist = []
        n_it = 0
        accepted_tiny_streak = 0
        tiny_step_norm_thresh = 1e-5
        tiny_step_max_thresh = 1e-6
        low_quality_thresh = 1e-2
        damping_saturation_thresh = 1e4

        @torch.no_grad()
        def _debug_step():
            nonlocal accepted_tiny_streak
            for pg in opt.param_groups:
                residual = list(opt.model(inp))[0]
                j_blocks = jacobian(residual, pg["params"])
                if isinstance(residual, TrackingTensor):
                    residual = residual.tensor()
                residual = residual.detach()
                j_blocks = [j.detach().to_sparse_coo() for j in j_blocks]
                # Apply Triggs FastTriggs (square-rooted Huber) correction to
                # (R, J). The bae library's `LM.step()` drops the kernel
                # when assembling J^T J and J^T r for the PCG solve;
                # without this the GN step descends pure L2 and outlier-
                # heavy datasets stall (info.md §3.31). The `model.loss`
                # used for accept/reject is already kerneled via pypose's
                # `RobustModel` wrapper, so we only need to weight R and J.
                residual, j_blocks = _apply_huber_correction(
                    residual, j_blocks, kernel_delta
                )
                raw_block_masks = [
                    constraint_masks["pose_active"],
                    constraint_masks["intr_active"],
                    constraint_masks["point_active"],
                ]
                compressed_blocks = []
                block_specs = []
                for block, active_mask_np in zip(j_blocks, raw_block_masks):
                    active_mask = torch.as_tensor(
                        active_mask_np, dtype=torch.bool, device=block.device)
                    compressed_block, active_mask, active_count = (
                        _compress_sparse_block_columns(block, active_mask)
                    )
                    compressed_blocks.append(compressed_block)
                    block_specs.append({
                        "active_mask": active_mask,
                        "active_count": (
                            int(active_mask.sum().item())
                            if active_count is None else int(active_count)
                        ),
                    })

                J_coo_unscaled = _cat_sparse_blocks(compressed_blocks)
                if J_coo_unscaled.shape[1] == 0:
                    _log(
                        f"lm iter {n_it + 1:3d}: no active parameter DoFs; "
                        "skipping LM step"
                    )
                    opt.last = opt.loss = (
                        opt.loss if hasattr(opt, "loss")
                        else opt.model.loss(inp, None)
                    )
                    continue
                J_coo = J_coo_unscaled
                scale = None
                _log_jtj_diag_stats(
                    f"lm iter {n_it + 1:3d} raw",
                    compressed_blocks,
                    pg["params"],
                )
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
                    scaled_blocks = []
                    offset = 0
                    for block, spec in zip(compressed_blocks, block_specs):
                        active_count = spec["active_count"]
                        if active_count == 0:
                            scaled_blocks.append(block)
                        else:
                            block_scale = scale[offset:offset + active_count]
                            scaled_blocks.append(
                                _apply_column_scaling(block, block_scale)
                            )
                        offset += active_count
                    _log_jtj_diag_stats(
                        f"lm iter {n_it + 1:3d} scaled",
                        scaled_blocks,
                        pg["params"],
                    )
                J = J_coo.to_sparse_csr()
                J_T = J.mT.to_sparse_csr()
                J_unscaled = J_coo_unscaled.to_sparse_csr()

                # opt.model is pypose's RobustModel wrapper, whose .loss
                # already applies the kernel — so this comparison is
                # Huber-correct without further work here. The matching
                # GN step direction is what `_apply_huber_correction`
                # above fixes. See info.md §3.31.
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
                        step_reduced = opt.solver(A, rhs)
                        step_reduced = step_reduced[:, None]
                        if scale is not None:
                            step_reduced = step_reduced * scale.reshape(-1, 1)
                    except Exception as e:
                        _log(
                            f"lm iter {n_it + 1:3d} attempt {attempt}: "
                            f"linear solver failed: {e!r}"
                        )
                        break

                    step_blocks_reduced = _split_step_vector(
                        step_reduced,
                        [spec["active_count"] for spec in block_specs],
                    )
                    step_blocks_full = _expand_step_blocks_to_full(
                        step_blocks_reduced, block_specs)
                    step_full = torch.cat(step_blocks_full).reshape(-1, 1)

                    _log_lm_iteration_stats(
                        f"lm iter {n_it + 1:3d} attempt {attempt}",
                        compressed_blocks,
                        residual,
                        step_reduced,
                        pg["params"],
                        block_numels=[
                            spec["active_count"] for spec in block_specs
                        ],
                    )

                    opt.update_parameter(pg["params"], step_full)
                    new_loss = opt.model.loss(inp, None)
                    new_loss = new_loss.detach() if torch.is_tensor(new_loss) else new_loss
                    quality_terms = _compute_quality_terms(
                        compressed_blocks,
                        step_blocks_reduced,
                        residual.view(-1, 1),
                        float(opt.last), float(new_loss))
                    quality = quality_terms["quality"]
                    actual_reduction = quality_terms["actual_reduction"]
                    predicted_reduction = quality_terms["predicted_reduction"]
                    step_reduced_norm = float(step_reduced.norm().item())
                    step_reduced_max = float(step_reduced.abs().max().item())
                    accepted_zero_step = (
                        step_reduced_norm < 1e-12 or step_reduced_max < 1e-12
                    )
                    tiny_step = (
                        step_reduced_norm < tiny_step_norm_thresh
                        or step_reduced_max < tiny_step_max_thresh
                    )
                    opt.loss = new_loss
                    opt.strategy.update(
                        pg, last=opt.last, loss=opt.loss, J=J_unscaled,
                        D=step_reduced, R=residual.view(-1, 1))
                    rejected = bool(opt.last < opt.loss and opt.reject_count < opt.reject)
                    damping_after = float(pg["damping"])
                    accepted_tiny_step = tiny_step and not rejected
                    damping_saturation = damping_after >= damping_saturation_thresh
                    _log(
                        f"lm iter {n_it + 1:3d} attempt {attempt}: "
                        f"last={float(opt.last):.6f} "
                        f"new={float(new_loss):.6f} "
                        f"actual_reduction={actual_reduction:.6e} "
                        f"predicted_reduction={predicted_reduction:.6e} "
                        f"quality={quality:.3e} "
                        f"step_norm={step_reduced_norm:.3e} "
                        f"step_max={step_reduced_max:.3e} "
                        f"damping={damping_before:.3e}->{damping_after:.3e} "
                        f"reject_count={opt.reject_count} "
                        f"accepted={not rejected} "
                        f"accepted_zero_step={accepted_zero_step and not rejected} "
                        f"accepted_tiny_step={accepted_tiny_step} "
                        f"damping_saturation={damping_saturation}"
                    )
                    if rejected:
                        accepted_tiny_streak = 0
                        opt.update_parameter(params=pg["params"], step=-step_full)
                        opt.loss = opt.last
                        opt.reject_count += 1
                    else:
                        if accepted_tiny_step and quality < low_quality_thresh:
                            accepted_tiny_streak += 1
                            _log(
                                f"lm iter {n_it + 1:3d} attempt {attempt}: "
                                "accepted tiny step with low quality; "
                                f"streak={accepted_tiny_streak} "
                                f"(tiny if step_norm<{tiny_step_norm_thresh:.0e} "
                                f"or step_max<{tiny_step_max_thresh:.0e}, "
                                f"low_quality<{low_quality_thresh:.0e})"
                            )
                            if accepted_tiny_streak >= 3:
                                _log(
                                    f"lm iter {n_it + 1:3d}: possible LM stagnation "
                                    f"({accepted_tiny_streak} consecutive tiny "
                                    "accepted steps)"
                                )
                        else:
                            accepted_tiny_streak = 0
                        accepted = True
                        break
                if not accepted and attempt > 0:
                    accepted_tiny_streak = 0
                    _log(
                        f"lm iter {n_it + 1:3d}: no accepted step after "
                        f"{attempt} attempt(s)"
                    )
            return opt.loss

        # Track which stop condition triggers the exit. Logged once at
        # the end so we can correlate "BA call did N iters and stopped
        # because X" with the per-stage residual percentile changes.
        exit_reason = "max_iter"
        windowed_imp_at_exit = float("nan")
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
                    exit_reason = "func_tol"
                    windowed_imp_at_exit = imp
                    break
                if loss_hist[-1] == loss_hist[-2]:
                    exit_reason = "loss_repeat"
                    windowed_imp_at_exit = imp
                    break
        # Diagnostic: report what stopped the LM so we can tell whether
        # raising max_num_iterations would help (max_iter exit) or not
        # (func_tol / loss_repeat exit).
        if len(loss_hist) >= 2:
            cost_drop_total = (
                (loss_hist[0] - loss_hist[-1]) / max(loss_hist[0], 1e-30)
            )
        else:
            cost_drop_total = 0.0
        _log(
            f"_run_ba exit: reason={exit_reason} n_it={n_it}/{max_iters} "
            f"cost_first={loss_hist[0] if loss_hist else float('nan'):.6f} "
            f"cost_last={loss_hist[-1] if loss_hist else float('nan'):.6f} "
            f"cost_drop_total={cost_drop_total:.4e} "
            f"windowed_imp={windowed_imp_at_exit:.4e}"
        )
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
    model, optimizer, input_data, remap, constraint_masks = _build_problem(
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
    # Use optimizer.model.loss (RobustModel) so initial_cost is in the same
    # kernel-weighted units as loss_hist[0] produced by _debug_step.
    initial_cost = optimizer.model.loss(input_data, None).item()
    _log(
        f"initial cost={initial_cost:.6f}, obs={len(image_indices_cur)}"
    )

    n_it, loss_hist = _run_ba(
        model, optimizer, input_data, max_iterations, constraint_masks)

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
