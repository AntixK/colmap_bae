"""BAE solver bridge for COLMAP bundle adjustment.

Called from C++ BaeBundleAdjuster::Solve() via pybind11 embedded Python.
Follows InstantSFM's TorchBA architecture: base LM optimizer with separate
parameter blocks and rotate_quat projection.
"""

import os
import sys

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
from bae.optim import LM
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

        intr_t = torch.tensor(intr_c, dtype=torch.float64, device=device)
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

        strat = pp.optim.strategy.TrustRegion(
            radius=1e4, max=1e10, up=2.0, down=0.5**4)
        slvr = PCG(tol=1e-5)
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
        for _ in range(max_iters):
            loss = opt.step(inp)
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
    intrinsics_full[cm_n2o] = model.intrinsics.data.cpu().numpy()
    intrinsics_full[:, 2] = 0.0
    points_full[pt_n2o] = model.points_3d.data.cpu().numpy()

    final_cost = loss_hist[-1] if loss_hist else initial_cost
    _log(
        f"finished: {n_it} iters, cost {initial_cost:.6f} -> "
        f"{final_cost:.6f}"
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
        options_dict.get("refine_focal_length", True),
        options_dict.get("refine_extra_params", True),
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
