"""BAE solver bridge for COLMAP bundle adjustment.

Called from C++ BaeBundleAdjuster::Solve() via pybind11 embedded Python.
Follows InstantSFM's TorchBA architecture: base LM optimizer with separate
parameter blocks and rotate_quat projection.
"""

import logging
import numpy as np
import pypose as pp
import torch
import torch.nn as nn

from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.utils.ba import rotate_quat
from bae.utils.pysolvers import PCG
from pypose.optim.kernel import Huber

logger = logging.getLogger("colmap.bae")


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

    # Coarse filter on the raw data (before compact remap) using GPU.
    _ext_t = torch.tensor(extrinsics_full, dtype=torch.float64, device=device)
    _intr_t = torch.tensor(intrinsics_full, dtype=torch.float64, device=device)
    _pts3_t = torch.tensor(points_full, dtype=torch.float64, device=device)
    _pts2_t = torch.tensor(points_2d_cur, dtype=torch.float64, device=device)
    _img_idx = torch.tensor(image_indices_cur, dtype=torch.long, device=device)
    _cam_idx = torch.tensor(camera_indices_cur, dtype=torch.long, device=device)
    _pt_idx = torch.tensor(point_indices_cur, dtype=torch.long, device=device)

    # Use a generous initial threshold (like InstantSFM iter-0:
    # max_reprojection_error * 3).  We use absolute pixels since our
    # observations are already centred.
    initial_filter_px = options_dict.get("initial_filter_px", 100.0)
    keep = _filter_observations_by_reproj(
        _ext_t, _intr_t, _pts3_t, _pts2_t,
        _img_idx, _cam_idx, _pt_idx,
        max_error=initial_filter_px,
    )
    keep_np = keep.cpu().numpy()
    n_before = len(image_indices_cur)
    image_indices_cur = image_indices_cur[keep_np]
    camera_indices_cur = camera_indices_cur[keep_np]
    point_indices_cur = point_indices_cur[keep_np]
    points_2d_cur = points_2d_cur[keep_np]
    logger.info(
        "BAE pre-filter: kept %d / %d observations (threshold %.1f px)",
        len(image_indices_cur), n_before, initial_filter_px,
    )
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

        ii = ig_o2n[img_idx_np]
        ci = cm_o2n[cam_idx_np]
        pi = pt_o2n[pt_idx_np]

        n_i, n_c, n_p, n_o = len(ig_n2o), len(cm_n2o), len(pt_n2o), len(ii)
        logger.info(
            "BAE problem: %d imgs, %d cams, %d pts, %d obs, const_rot=%s",
            n_i, n_c, n_p, n_o, constant_rotation,
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
        hub = Huber(1.0)  # matching InstantSFM exactly
        opt = LM(mdl, strategy=strat, solver=slvr, kernel=hub, reject=30)
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
            logger.info("BAE iter %3d  cost=%.6f", n_it, loss_hist[-1])
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
    # Decreasing thresholds: optimise then filter, progressively
    # cleaning the data.
    # ------------------------------------------------------------------
    filter_thresholds = [100.0, 50.0, 20.0]
    total_iterations = 0
    initial_cost = None

    for rnd, thresh in enumerate(filter_thresholds):
        model, optimizer, input_data, remap = _build_problem(
            image_indices_cur, camera_indices_cur,
            point_indices_cur, points_2d_cur,
        )
        if model is None:
            logger.info("BAE round %d: no observations, skipping.", rnd)
            break

        cost0 = model.loss(input_data, None).item()
        if initial_cost is None:
            initial_cost = cost0
        logger.info("BAE round %d: initial cost=%.6f", rnd, cost0)

        n_it, loss_hist = _run_ba(model, optimizer, input_data, max_iterations)
        total_iterations += n_it

        # Write optimised params back into the full-size numpy arrays
        # so the next round (and the C++ caller) sees them.
        ig_n2o, cm_n2o, pt_n2o = remap
        if constant_rotation:
            extrinsics_full[ig_n2o, :3] = (
                model.translations.data.cpu().numpy())
        else:
            extrinsics_full[ig_n2o] = model.extrinsics.data.cpu().numpy()
        intrinsics_full[cm_n2o] = model.intrinsics.data.cpu().numpy()
        points_full[pt_n2o] = model.points_3d.data.cpu().numpy()

        # Filter observations by reprojection error for the next round
        # (adapted from InstantSFM FilterTracksByReprojectionNormalized).
        ext_t = torch.tensor(extrinsics_full, dtype=torch.float64, device=device)
        intr_t = torch.tensor(intrinsics_full, dtype=torch.float64, device=device)
        p3_t = torch.tensor(points_full, dtype=torch.float64, device=device)
        p2_t = torch.tensor(points_2d_cur, dtype=torch.float64, device=device)
        ii_t = torch.tensor(image_indices_cur, dtype=torch.long, device=device)
        ci_t = torch.tensor(camera_indices_cur, dtype=torch.long, device=device)
        pi_t = torch.tensor(point_indices_cur, dtype=torch.long, device=device)

        keep = _filter_observations_by_reproj(
            ext_t, intr_t, p3_t, p2_t, ii_t, ci_t, pi_t,
            max_error=thresh,
        )
        keep_np = keep.cpu().numpy()
        n_before = len(image_indices_cur)
        image_indices_cur = image_indices_cur[keep_np]
        camera_indices_cur = camera_indices_cur[keep_np]
        point_indices_cur = point_indices_cur[keep_np]
        points_2d_cur = points_2d_cur[keep_np]
        logger.info(
            "BAE round %d filter: kept %d / %d obs (threshold %.1f px)",
            rnd, len(image_indices_cur), n_before, thresh,
        )
        del ext_t, intr_t, p3_t, p2_t, ii_t, ci_t, pi_t
        torch.cuda.empty_cache()

    final_cost = loss_hist[-1] if loss_hist else (initial_cost or 0.0)
    logger.info(
        "BAE finished: %d total iters, cost %.6f -> %.6f",
        total_iterations, initial_cost or 0.0, final_cost,
    )

    return {
        "extrinsics": extrinsics_full,
        "intrinsics": intrinsics_full,
        "points_3d": points_full,
        "num_iterations": total_iterations,
        "initial_cost": initial_cost or 0.0,
        "final_cost": final_cost,
        "converged": True,
    }
