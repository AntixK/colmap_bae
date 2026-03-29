"""BAE solver bridge for COLMAP bundle adjustment.

Called from C++ BaeBundleAdjuster::Solve() via pybind11 embedded Python.
Provides ColmapReproj (COLMAP-compatible projection model) and a solve()
entry point that runs BAE's Levenberg-Marquardt optimizer.

Following InstantSFM's architecture, extrinsics (per-image SE3 poses) and
intrinsics (per-camera [f, k1, k2]) are stored as separate nn.Parameters.
This produces sparser Jacobian blocks in BAE and reduces GPU memory usage.
"""

import logging
import math

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


@map_transform
def colmap_project(points, extrinsics, intrinsics):
    """Project 3D points using COLMAP's SIMPLERADIAL/Radial camera model.

    Unlike BAE's default project() which negates the projection
    (BAL dataset convention: -X/Z), this follows COLMAP's standard
    pinhole model: u = X/Z, v = Y/Z (no negation).

    Args:
        points:     (N, 3) world-space 3D points
        extrinsics: (N, 7) per-observation [tx,ty,tz, qx,qy,qz,qw]
        intrinsics: (N, 3) per-observation [f, k1, k2]

    Observed 2D points are pre-centered on the C++ side as
    (obs_x - cx, obs_y - cy), so cx,cy do not appear here.

    The projection is:
      1. p_cam = SE3(pose) * p_world        (rotate + translate)
      2. u = p_cam.x / p_cam.z              (perspective division)
      3. r^2 = u^2 + v^2                    (radial distance squared)
      4. d = 1 + k1*r^2 + k2*r^4            (radial distortion)
      5. proj = f * d * [u, v]               (focal length scaling)

    This matches COLMAP's SimpleRadial/Radial camera models exactly.
    """
    # SE3 transform: p_cam = R * p_world + t
    points_proj = rotate_quat(points, extrinsics)
    # Perspective division WITHOUT negation (COLMAP convention).
    points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)
    # Radial distortion: d = 1 + k1*r^2 + k2*r^4
    f = intrinsics[..., 0].unsqueeze(-1)
    k1 = intrinsics[..., 1].unsqueeze(-1)
    k2 = intrinsics[..., 2].unsqueeze(-1)
    n = torch.sum(points_proj**2, dim=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    points_proj = points_proj * r * f
    return points_proj


class ColmapReproj(nn.Module):
    """COLMAP-compatible reprojection model for BAE optimizer.

    Following InstantSFM's architecture, parameters are stored as
    three separate nn.Parameter blocks for sparser Jacobians:
      extrinsics: (N_imgs, 7)  [tx,ty,tz, qx,qy,qz,qw]
      intrinsics: (N_cams, 3)  [f, k1, k2]
      points_3d:  (N_pts, 3)   [x, y, z]
    """

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
        """Compute 0.5 * ||residual||^2 (standard nonlinear least squares)."""
        if isinstance(input, dict):
            R = self.forward(**input)
        else:
            R = self.forward(input)
        return (R**2).sum() / 2


class ColmapLM(LM):
    """LM optimizer with support for freezing individual cameras/points.

    Handles 3 separate parameter blocks (identified by identity):
      extrinsics: SE3 manifold update, frozen by constant_pose_mask
      intrinsics: Euclidean update, controlled by refine_focal_length/extra_params
      points_3d:  Euclidean update, frozen by constant_point_mask
    """

    def __init__(
        self,
        model,
        constant_pose_mask,
        constant_point_mask,
        refine_focal_length=True,
        refine_extra_params=True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.constant_pose_mask = constant_pose_mask
        self.constant_point_mask = constant_point_mask
        self.refine_focal_length = refine_focal_length
        self.refine_extra_params = refine_extra_params

    def update_parameter(self, params, step):
        numels = []
        for param in params:
            if param.requires_grad:
                if getattr(param, "trim_SE3_grad", False):
                    numels.append(
                        math.prod(param.shape[:-1]) * (param.shape[-1] - 1)
                    )
                else:
                    numels.append(param.numel())
        steps = step.split(numels)
        for param, d in zip(params, steps):
            if not param.requires_grad:
                continue
            if param is self.model.extrinsics:
                # Extrinsics: d shape (N_imgs * 6,) -> (N_imgs, 6)
                d = d.view(param.shape[0], -1).clone()
                d[self.constant_pose_mask] = 0
                param[..., :7] = pp.SE3(param[..., :7]).add_(
                    pp.se3(d[..., :6])
                )
            elif param is self.model.intrinsics:
                # Intrinsics: d shape (N_cams * 3,) -> (N_cams, 3)
                d = d.view(param.shape).clone()
                if not self.refine_focal_length:
                    d[:, 0] = 0
                if not self.refine_extra_params:
                    d[:, 1:] = 0
                param.add_(d)
            else:
                # Points: d shape (N_pts * 3,) -> (N_pts, 3)
                d = d.view(param.shape).clone()
                d[self.constant_point_mask] = 0
                param.add_(d)


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
    """Entry point called from C++ BaeBundleAdjuster::Solve().

    All array inputs are numpy arrays (flat or shaped). Returns a dict:
      extrinsics:     optimized (N_imgs, 7) numpy array
      intrinsics:     optimized (N_cams, 3) numpy array
      points_3d:      optimized (N_pts, 3)  numpy array
      num_iterations: int
      initial_cost:   float
      final_cost:     float
      converged:      bool
    """
    # Determine device.
    use_gpu = options_dict.get("use_gpu", True)
    gpu_index = options_dict.get("gpu_index", "0")
    if use_gpu and torch.cuda.is_available():
        device = f"cuda:{gpu_index}"
        torch.cuda.empty_cache()
    else:
        device = "cpu"

    max_iterations = options_dict.get("max_num_iterations", 100)
    refine_focal_length = options_dict.get("refine_focal_length", True)
    refine_extra_params = options_dict.get("refine_extra_params", True)
    loss_function_scale = options_dict.get("loss_function_scale", 1.0)

    n_imgs = extrinsics_np.size // 7
    n_cams = intrinsics_np.size // 3
    n_pts = points_3d_np.size // 3
    n_obs = image_indices_np.size
    logger.info(
        "BAE solver: %d images, %d cameras, %d points, %d observations, "
        "device=%s",
        n_imgs, n_cams, n_pts, n_obs, device,
    )

    def _build(device):
        """Allocate tensors, model, and optimizer on the given device."""
        extrinsics = torch.tensor(
            extrinsics_np, dtype=torch.float64, device=device
        ).reshape(-1, 7)
        intrinsics = torch.tensor(
            intrinsics_np, dtype=torch.float64, device=device
        ).reshape(-1, 3)
        points_3d = torch.tensor(
            points_3d_np, dtype=torch.float64, device=device
        ).reshape(-1, 3)
        points_2d = torch.tensor(
            points_2d_np, dtype=torch.float64, device=device
        ).reshape(-1, 2)
        image_indices = torch.tensor(
            image_indices_np, dtype=torch.long, device=device
        )
        camera_indices = torch.tensor(
            camera_indices_np, dtype=torch.long, device=device
        )
        point_indices = torch.tensor(
            point_indices_np, dtype=torch.long, device=device
        )
        constant_pose_mask = torch.tensor(
            constant_pose_mask_np, dtype=torch.bool, device=device
        )
        constant_point_mask = torch.tensor(
            constant_point_mask_np, dtype=torch.bool, device=device
        )

        model = ColmapReproj(extrinsics, intrinsics, points_3d).to(device)
        strategy = pp.optim.strategy.TrustRegion(
            radius=1e4, max=1e10, up=2.0, down=0.5**4,
        )
        solver = PCG(tol=1e-5)
        huber_kernel = Huber(loss_function_scale)
        optimizer = ColmapLM(
            model,
            constant_pose_mask=constant_pose_mask,
            constant_point_mask=constant_point_mask,
            refine_focal_length=refine_focal_length,
            refine_extra_params=refine_extra_params,
            strategy=strategy,
            solver=solver,
            kernel=huber_kernel,
            reject=30,
        )

        input_data = {
            "points_2d": points_2d,
            "image_indices": image_indices,
            "camera_indices": camera_indices,
            "point_indices": point_indices,
        }
        return model, optimizer, input_data

    try:
        model, optimizer, input_data = _build(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "CUDA out of memory on %s (%d obs), falling back to CPU",
            device, n_obs,
        )
        torch.cuda.empty_cache()
        device = "cpu"
        model, optimizer, input_data = _build(device)

    # Compute initial cost.
    initial_cost = model.loss(input_data, None).item()
    logger.info("BAE initial cost: %.6f", initial_cost)

    # Run optimization loop.
    num_iterations = 0
    final_cost = initial_cost
    converged = False
    prev_cost = initial_cost

    def _run_loop(model, optimizer, input_data, prev_cost):
        num_iterations = 0
        final_cost = prev_cost
        converged = False
        for _ in range(max_iterations):
            loss = optimizer.step(input_data)
            num_iterations += 1
            final_cost = loss.item()
            logger.info(
                "BAE iter %3d  cost=%.6f  rel_change=%.2e",
                num_iterations,
                final_cost,
                abs(prev_cost - final_cost) / max(prev_cost, 1e-12),
            )
            rel_change = abs(prev_cost - final_cost) / max(prev_cost, 1e-12)
            if rel_change < 1e-6:
                converged = True
                break
            prev_cost = final_cost
        return num_iterations, final_cost, converged

    try:
        num_iterations, final_cost, converged = _run_loop(
            model, optimizer, input_data, prev_cost,
        )
    except torch.cuda.OutOfMemoryError:
        if device == "cpu":
            raise
        logger.warning(
            "CUDA out of memory during optimization, falling back to CPU",
        )
        torch.cuda.empty_cache()
        device = "cpu"
        model, optimizer, input_data = _build(device)
        initial_cost = model.loss(input_data, None).item()
        prev_cost = initial_cost
        num_iterations, final_cost, converged = _run_loop(
            model, optimizer, input_data, prev_cost,
        )

    logger.info(
        "BAE finished: %d iterations, cost %.6f -> %.6f, converged=%s",
        num_iterations, initial_cost, final_cost, converged,
    )

    return {
        "extrinsics": model.extrinsics.detach().cpu().numpy(),
        "intrinsics": model.intrinsics.detach().cpu().numpy(),
        "points_3d": model.points_3d.detach().cpu().numpy(),
        "num_iterations": num_iterations,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "converged": converged,
    }
