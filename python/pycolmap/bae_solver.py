"""BAE solver bridge for COLMAP bundle adjustment.

Called from C++ BaeBundleAdjuster::Solve() via pybind11 embedded Python.
Follows InstantSFM's TorchBA architecture: base LM optimizer with separate
parameter blocks.

When constant_rig_from_world_rotation=True (fixed-rotation BA stage),
translations are separated from rotations following TorchBA's
ReprojectionMultiRigModelFixedRel pattern: rotations become fixed input
data while only translations are optimized as parameters.
"""

import logging
import numpy as np
import pypose as pp
import torch
import torch.nn as nn

from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.utils.pysolvers import PCG

logger = logging.getLogger("colmap.bae")


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

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
def colmap_project_se3(points, se3_poses, intrinsics):
    """Project using se3 (Lie algebra) pose parameterization.

    Implements the SO3 exponential map (Rodrigues) manually using only
    standard torch ops — no pypose Exp (which uses boolean masking
    incompatible with vmap/jacrev).

    se3_poses: [..., 6] = [tx, ty, tz, wx, wy, wz]
    """
    t = se3_poses[..., :3]
    w = se3_poses[..., 3:6]

    # Rodrigues: R*p = p + sin(θ)*(k×p) + (1-cos(θ))*(k×(k×p))
    # Use eps to avoid division by zero (vmap-safe, no boolean mask).
    theta_sq = (w * w).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq.clamp(min=1e-24))
    k = w / theta  # unit axis

    kxp = torch.linalg.cross(k, points, dim=-1)
    kxkxp = torch.linalg.cross(k, kxp, dim=-1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    rotated = points + sin_t * kxp + (1.0 - cos_t) * kxkxp

    points_cam = rotated + t
    return _distort_and_project(points_cam, intrinsics)


@map_transform
def colmap_project_fixed_rot(points, translations, rotations, intrinsics):
    """Project with fixed rotations (se3 not needed here)."""
    rotated = pp.SO3(rotations).Act(points)
    points_cam = rotated + translations
    return _distort_and_project(points_cam, intrinsics)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class ColmapReproj(nn.Module):
    """Full BA with se3 pose parameterization.

    Poses are stored as se3 (6D Lie algebra).  The exponential map
    converts to SE3 inside forward(), so the Jacobian is naturally in
    se3 space and the LM update is plain Euclidean addition.
    """

    def __init__(self, se3_poses, intrinsics, points_3d):
        super().__init__()
        self.se3_poses = nn.Parameter(TrackingTensor(se3_poses))
        self.intrinsics = nn.Parameter(TrackingTensor(intrinsics))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        # No trim_SE3_grad — se3 is already 6D, Euclidean update is correct.

    def forward(self, points_2d, image_indices, camera_indices, point_indices):
        points_proj = colmap_project_se3(
            self.points_3d[point_indices],
            self.se3_poses[image_indices],
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
    """Fixed-rotation BA: rotations are frozen, only translations optimized.

    Follows TorchBA's ReprojectionMultiRigModelFixedRel pattern:
    rotations are passed via forward() as fixed input, not as Parameters.
    """

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compact_remap(indices, total):
    """Build a compact remapping, filtering out unobserved parameters."""
    used_mask = np.zeros(total, dtype=bool)
    used_mask[indices] = True
    new2old = np.where(used_mask)[0]
    old2new = np.full(total, -1, dtype=np.int64)
    old2new[new2old] = np.arange(len(new2old))
    return used_mask, old2new, new2old


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def solve(
    extrinsics_np,
    intrinsics_np,
    points_3d_np,
    points_2d_np,
    image_indices_np,
    camera_indices_np,
    point_indices_np,
    constant_pose_mask_np,
    constant_point_mask_np,
    options_dict,
):
    """Entry point called from C++ BaeBundleAdjuster::Solve()."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "BAE backend requires CUDA, but torch.cuda.is_available() is False."
        )
    gpu_index = options_dict.get("gpu_index", "0")
    device = f"cuda:{gpu_index}"
    torch.cuda.empty_cache()

    max_iterations = options_dict.get("max_num_iterations", 100)
    refine_focal_length = options_dict.get("refine_focal_length", True)
    refine_extra_params = options_dict.get("refine_extra_params", True)
    constant_rotation = options_dict.get(
        "constant_rig_from_world_rotation", False,
    )


    extrinsics_full = extrinsics_np.reshape(-1, 7)
    intrinsics_full = intrinsics_np.reshape(-1, 3)
    points_full = points_3d_np.reshape(-1, 3)

    n_imgs_orig = extrinsics_full.shape[0]
    n_cams_orig = intrinsics_full.shape[0]
    n_pts_orig = points_full.shape[0]

    # ---- Filter to only parameters referenced by observations ----
    img_used, img_old2new, img_new2old = _compact_remap(
        image_indices_np, n_imgs_orig,
    )
    cam_used, cam_old2new, cam_new2old = _compact_remap(
        camera_indices_np, n_cams_orig,
    )
    pt_used, pt_old2new, pt_new2old = _compact_remap(
        point_indices_np, n_pts_orig,
    )

    extrinsics_compact = extrinsics_full[img_new2old]
    intrinsics_compact = intrinsics_full[cam_new2old]
    points_compact = points_full[pt_new2old]

    image_indices_compact = img_old2new[image_indices_np]
    camera_indices_compact = cam_old2new[camera_indices_np]
    point_indices_compact = pt_old2new[point_indices_np]

    n_imgs = len(img_new2old)
    n_cams = len(cam_new2old)
    n_pts = len(pt_new2old)
    n_obs = image_indices_compact.size

    logger.info(
        "BAE solver: %d images, %d cameras, %d points, %d observations, "
        "device=%s, const_rot=%s, refine_f=%s, refine_k=%s",
        n_imgs, n_cams, n_pts, n_obs, device,
        constant_rotation, refine_focal_length, refine_extra_params,
    )

    # ---- Build tensors ----
    intrinsics_t = torch.tensor(
        intrinsics_compact, dtype=torch.float64, device=device,
    )
    points_3d_t = torch.tensor(
        points_compact, dtype=torch.float64, device=device,
    )
    points_2d_t = torch.tensor(
        points_2d_np, dtype=torch.float64, device=device,
    ).reshape(-1, 2)
    image_indices_t = torch.tensor(
        image_indices_compact, dtype=torch.int32, device=device,
    )
    camera_indices_t = torch.tensor(
        camera_indices_compact, dtype=torch.int32, device=device,
    )
    point_indices_t = torch.tensor(
        point_indices_compact, dtype=torch.int32, device=device,
    )

    # ---- Build model (architecture depends on constant_rotation) ----
    if constant_rotation:
        # Fixed-rotation mode: separate translations (Parameter) from
        # rotations (plain tensor passed via input_data).
        translations_t = torch.tensor(
            extrinsics_compact[:, :3], dtype=torch.float64, device=device,
        )
        rotations_t = torch.tensor(
            extrinsics_compact[:, 3:7], dtype=torch.float64, device=device,
        )
        model = ColmapReprojFixedRot(
            translations_t, intrinsics_t, points_3d_t,
        ).to(device)
        input_data = {
            "points_2d": points_2d_t,
            "image_indices": image_indices_t,
            "camera_indices": camera_indices_t,
            "point_indices": point_indices_t,
            "rotations": rotations_t,
        }
    else:
        # Full mode: convert SE3 (7D) → se3 (6D) via logarithm map.
        extrinsics_t = torch.tensor(
            extrinsics_compact, dtype=torch.float64, device=device,
        )
        se3_poses = pp.Log(pp.SE3(extrinsics_t)).tensor()
        model = ColmapReproj(
            se3_poses, intrinsics_t, points_3d_t,
        ).to(device)
        input_data = {
            "points_2d": points_2d_t,
            "image_indices": image_indices_t,
            "camera_indices": camera_indices_t,
            "point_indices": point_indices_t,
        }

    # ---- Build optimizer (base LM, like TorchBA) ----
    # No Huber kernel: BAE's LM tracks accept/reject via model.loss()
    # which returns unweighted cost.  Kernel creates a mismatch that
    # rejects all steps when initial errors are large (global SfM).
    #
    # Trust region tuned for global SfM: small initial radius keeps
    # damping high (LM regime) to avoid premature Gauss-Newton steps
    # that stall on this highly non-linear problem.
    strategy = pp.optim.strategy.TrustRegion(
        radius=1e2, max=1e6, up=1.5, down=0.5,
    )
    solver = PCG(tol=1e-5)
    optimizer = LM(
        model,
        strategy=strategy,
        solver=solver,
        reject=100,
    )

    # Compute initial cost.
    initial_cost = model.loss(input_data, None).item()
    logger.info("BAE initial cost: %.6f", initial_cost)

    # ---- Optimization loop ----
    prev_cost = initial_cost
    num_iterations = 0
    final_cost = initial_cost
    converged = False

    # Snapshot intrinsic components we should NOT refine.
    frozen_focal = None
    if not refine_focal_length:
        frozen_focal = model.intrinsics.data[:, 0].clone()
    frozen_distortion = None
    if not refine_extra_params:
        frozen_distortion = model.intrinsics.data[:, 1:].clone()

    # Windowed convergence check (matching InstantSFM's _run_optimization).
    window_size = 4
    function_tolerance = 5e-4
    loss_history = []

    for _ in range(max_iterations):
        loss = optimizer.step(input_data)

        # Restore intrinsic components that shouldn't change.
        with torch.no_grad():
            if frozen_focal is not None:
                model.intrinsics.data[:, 0] = frozen_focal
            if frozen_distortion is not None:
                model.intrinsics.data[:, 1:] = frozen_distortion

        num_iterations += 1
        final_cost = loss.item()
        loss_history.append(final_cost)
        logger.info(
            "BAE iter %3d  cost=%.6f",
            num_iterations, final_cost,
        )

        # InstantSFM convergence: compare windowed averages.
        if len(loss_history) >= 2 * window_size:
            avg_recent = sum(loss_history[-window_size:]) / window_size
            avg_prev = sum(loss_history[-2*window_size:-window_size]) / window_size
            improvement = (avg_prev - avg_recent) / avg_prev
            if abs(improvement) < function_tolerance:
                converged = True
                break
            if loss_history[-1] == loss_history[-2]:
                converged = True
                break

    logger.info(
        "BAE finished: %d iterations, cost %.6f -> %.6f, converged=%s",
        num_iterations, initial_cost, final_cost, converged,
    )

    # ---- Expand compact results back to original-sized arrays ----
    if constant_rotation:
        # Reconstruct full extrinsics from optimized translations + fixed rots.
        opt_trans = model.translations.detach().cpu().numpy()
        extr_compact = extrinsics_compact.copy()
        extr_compact[:, :3] = opt_trans
    else:
        # Convert optimized se3 (6D) back to SE3 (7D) [tx,ty,tz,qx,qy,qz,qw].
        se3_opt = model.se3_poses.detach().cpu()
        t_out = se3_opt[:, :3]
        w_out = se3_opt[:, 3:6]
        theta = w_out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        half_t = theta / 2.0
        k = w_out / theta
        qw = torch.cos(half_t)
        qxyz = k * torch.sin(half_t)
        extr_compact = torch.cat([t_out, qxyz, qw], dim=-1).numpy()

    intr_compact = model.intrinsics.detach().cpu().numpy()
    pts_compact = model.points_3d.detach().cpu().numpy()

    extr_out = extrinsics_full.copy()
    extr_out[img_new2old] = extr_compact

    intr_out = intrinsics_full.copy()
    intr_out[cam_new2old] = intr_compact

    pts_out = points_full.copy()
    pts_out[pt_new2old] = pts_compact

    return {
        "extrinsics": extr_out,
        "intrinsics": intr_out,
        "points_3d": pts_out,
        "num_iterations": num_iterations,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "converged": converged,
    }
