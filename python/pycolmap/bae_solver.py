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
    """Perspective divide + radial distortion + focal scaling."""
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., 0].unsqueeze(-1)
    k1 = intrinsics[..., 1].unsqueeze(-1)
    k2 = intrinsics[..., 2].unsqueeze(-1)
    n = torch.sum(points_proj**2, dim=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    return points_proj * r * f


@map_transform
def colmap_project(points, extrinsics, intrinsics):
    """Project using SE3 extrinsics (same as TorchBA's rotate_quat)."""
    points_cam = rotate_quat(points, extrinsics)
    return _distort_and_project(points_cam, intrinsics)


@map_transform
def colmap_project_fixed_rot(points, translations, rotations, intrinsics):
    """Project with fixed rotations: only translations are optimized."""
    rotated = pp.SO3(rotations).Act(points)
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
    ext = extrinsics[image_indices]                   # (N, 7)
    intr = intrinsics[camera_indices]                 # (N, 3)

    # R * p + t  via pypose SE3
    pts_cam = pp.SE3(ext).Act(pts)                    # (N, 3)

    # Perspective divide
    z = pts_cam[:, 2:3].clamp(min=1e-8)
    uv = pts_cam[:, :2] / z                           # (N, 2)

    # Radial distortion
    f  = intr[:, 0:1]
    k1 = intr[:, 1:2]
    k2 = intr[:, 2:3]
    r2 = (uv * uv).sum(dim=-1, keepdim=True)
    dist = 1.0 + k1 * r2 + k2 * r2 * r2
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
    pts_cam_z = pp.SE3(extrinsics[image_indices]).Act(
        points_3d[point_indices])[:, 2]
    keep = (errs < max_error) & (pts_cam_z > 0)
    return keep


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

    extrinsics_full = extrinsics_np.reshape(-1, 7)
    intrinsics_full = intrinsics_np.reshape(-1, 3)
    points_full = points_3d_np.reshape(-1, 3)

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
        pts_z = pp.SE3(_ext_t[_img_idx]).Act(_pts3_t[_pt_idx])[:, 2]
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

    # Robust kernel scale (#1): Huber transition at 2 * median pixel
    # residual.  With a bad initialization (median ~500 px) Huber(1.0)
    # puts every observation in the linear regime, so all gradients become
    # constant-magnitude and the LM step direction is dominated by sign
    # noise rather than residual structure.  Scaling the kernel to the
    # actual residual distribution keeps inliers in the quadratic regime.
    kernel_delta = max(2.0 * median_err_px, 1.0)
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

    # Track-length distribution log (post norm-filter).  Useful for tuning
    # any future min-track filter — sequential matching with overlap=N
    # produces tracks of length 5..N+ typically; isolated short tracks
    # (length 2) survive only when matching gaps are large.
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

    # Triangulation-angle filter (#A): drop observations whose track has
    # near-degenerate parallax.  A track with all rays within a tight cone
    # is poorly conditioned: tiny perturbations of the 3D point change
    # depth dramatically without changing the projected residual, so BA
    # can't find a descent direction.  These are the observations that
    # survive a residual filter (small residual) but pin BA at a
    # high-residual plateau (~6e-2 norm in our case).
    #
    # We compute, per point, the maximum angle between any of its rays
    # and the mean ray (a fast O(N_obs) proxy for max pairwise angle —
    # max-pairwise is bounded by 2x max-from-mean, so this is at worst
    # 2x stricter than the true pairwise threshold).  COLMAP's
    # incremental mapper uses a 1.5deg minimum (incremental_mapper.h:126).
    TRI_ANGLE_DEG = 1.5
    cos_thresh = float(np.cos(np.radians(TRI_ANGLE_DEG)))

    img_idx_t = torch.tensor(image_indices_cur, dtype=torch.long, device=device)
    pt_idx_t = torch.tensor(point_indices_cur, dtype=torch.long, device=device)
    ext_t = torch.tensor(extrinsics_full, dtype=torch.float64, device=device)
    pts3_t = torch.tensor(points_full, dtype=torch.float64, device=device)

    # Camera centers in world frame.  Existing convention: pp.SO3(q).Act(p)
    # rotates world -> cam, then + t gives cam-frame point (lines 51-52,
    # 92-98).  Therefore cam_center_world = -SO3(q).Inv().Act(t).
    quats = ext_t[:, 3:7]
    trans = ext_t[:, :3]
    cam_centers = -pp.SO3(quats).Inv().Act(trans)

    # Per-obs unit ray from camera center to 3D point in world frame.
    rays = pts3_t[pt_idx_t] - cam_centers[img_idx_t]
    rays_unit = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Per-point mean ray (sum of unit rays, then normalize).
    sum_ray = torch.zeros(n_pts_orig, 3, dtype=torch.float64, device=device)
    sum_ray.scatter_add_(
        0, pt_idx_t.unsqueeze(-1).expand(-1, 3), rays_unit)
    mean_ray = sum_ray / sum_ray.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Per-obs dot to its track's mean ray.  Per-point min dot = max angle
    # from mean.  Initialize to >1 sentinel so empty tracks don't pass.
    dots = (rays_unit * mean_ray[pt_idx_t]).sum(dim=-1)
    min_dot_per_pt = torch.full(
        (n_pts_orig,), 1.0 + 1e-9, dtype=torch.float64, device=device)
    min_dot_per_pt.scatter_reduce_(0, pt_idx_t, dots, reduce="amin")

    # Log angle distribution (only over points present in our obs set).
    present_mask = torch.zeros(n_pts_orig, dtype=torch.bool, device=device)
    present_mask[pt_idx_t] = True
    min_angles_deg = torch.rad2deg(
        torch.arccos(min_dot_per_pt.clamp(-1.0, 1.0)))
    present_angles = min_angles_deg[present_mask].cpu().numpy()
    if len(present_angles) > 0:
        apcts = np.percentile(
            present_angles, [10, 25, 50, 75, 90, 95, 99])
        _log(
            f"track angle [deg]: p10={apcts[0]:.2f} p25={apcts[1]:.2f} "
            f"p50={apcts[2]:.2f} p75={apcts[3]:.2f} p90={apcts[4]:.2f} "
            f"p95={apcts[5]:.2f} p99={apcts[6]:.2f}"
        )

    # Drop obs whose point is inside the degenerate cone.
    track_keep_pt = (min_dot_per_pt < cos_thresh).cpu().numpy()
    obs_keep = track_keep_pt[point_indices_cur]
    n_before_tri = len(image_indices_cur)
    image_indices_cur = image_indices_cur[obs_keep]
    camera_indices_cur = camera_indices_cur[obs_keep]
    point_indices_cur = point_indices_cur[obs_keep]
    points_2d_cur = points_2d_cur[obs_keep]
    _log(
        f"tri-angle filter: kept {len(image_indices_cur)} / "
        f"{n_before_tri} observations "
        f"(max-from-mean >= {TRI_ANGLE_DEG:.1f} deg)"
    )

    del _ext_t, _intr_t, _pts3_t, _pts2_t, _img_idx, _cam_idx, _pt_idx
    del img_idx_t, pt_idx_t, ext_t, pts3_t, cam_centers
    del rays, rays_unit, sum_ray, mean_ray, dots, min_dot_per_pt
    del min_angles_deg, present_mask
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
                ext_c[:, :3], dtype=torch.float64, device=device)
            ro_t = torch.tensor(
                ext_c[:, 3:7], dtype=torch.float64, device=device)
            mdl = ColmapReprojFixedRot(tr_t, intr_t, p3_t).to(device)
            inp = {"points_2d": p2_t, "image_indices": ii_t,
                   "camera_indices": ci_t, "point_indices": pi_t,
                   "rotations": ro_t}
        else:
            ex_t = torch.tensor(ext_c, dtype=torch.float64, device=device)
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
    if constant_rotation:
        extrinsics_full[ig_n2o, :3] = (
            model.translations.data.cpu().numpy())
    else:
        extrinsics_full[ig_n2o] = model.extrinsics.data.cpu().numpy()
    intrinsics_full[cm_n2o] = model.intrinsics.data.cpu().numpy()
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
    del post_ext, post_intr, post_pts3, post_pts2, post_ii, post_ci, post_pi
    torch.cuda.empty_cache()

    return {
        "extrinsics": extrinsics_full,
        "intrinsics": intrinsics_full,
        "points_3d": points_full,
        "num_iterations": n_it,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "converged": True,
    }
