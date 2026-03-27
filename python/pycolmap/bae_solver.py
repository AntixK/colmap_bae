"""BAE solver bridge for COLMAP bundle adjustment.

Called from C++ BaeBundleAdjuster::Solve() via pybind11 embedded Python.
Provides ColmapReproj (COLMAP-compatible projection model) and a solve()
entry point that runs BAE's Levenberg-Marquardt optimizer.
"""

import math

import numpy as np
import pypose as pp
import torch
import torch.nn as nn
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.utils.ba import rotate_quat
from bae.utils.pysolvers import PCG


@map_transform
def colmap_project(points, camera_params):
    """Project 3D points using COLMAP's SIMPLERADIAL camera model.

    Unlike BAE's default project() which negates the projection
    (BAL dataset convention: -X/Z), this follows COLMAP's standard
    pinhole model: u = X/Z, v = Y/Z (no negation).

    Camera params layout per image: [tx,ty,tz, qx,qy,qz,qw, f, k1, k2]

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
    points_proj = rotate_quat(points, camera_params[..., :7])
    # Perspective division WITHOUT negation (COLMAP convention).
    points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)
    # Radial distortion: d = 1 + k1*r^2 + k2*r^4
    f = camera_params[..., -3].unsqueeze(-1)
    k1 = camera_params[..., -2].unsqueeze(-1)
    k2 = camera_params[..., -1].unsqueeze(-1)
    n = torch.sum(points_proj**2, dim=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    points_proj = points_proj * r * f
    return points_proj


class ColmapReproj(nn.Module):
    """COLMAP-compatible reprojection model for BAE optimizer.

    Unlike BAE's default Reproj (ba_helpers.py), this uses COLMAP's
    projection convention (no negation in perspective division).

    Parameters are stored as:
      pose:      (N_imgs, 10) [tx,ty,tz, qx,qy,qz,qw, f, k1, k2]
      points_3d: (N_pts, 3)   [x, y, z]
    """

    def __init__(self, camera_params, points_3d):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(camera_params))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.pose.trim_SE3_grad = True

    def forward(self, points_2d, camera_indices, point_indices):
        points_proj = colmap_project(
            self.points_3d[point_indices],
            self.pose[camera_indices],
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

    Overrides update_parameter to zero out updates for:
    - Cameras marked constant by constant_camera_mask (pose + intrinsics)
    - Focal length globally (when refine_focal_length=False)
    - Distortion params globally (when refine_extra_params=False)
    - Points marked constant by constant_point_mask
    """

    def __init__(
        self,
        model,
        constant_camera_mask,
        constant_point_mask,
        refine_focal_length=True,
        refine_extra_params=True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.constant_camera_mask = constant_camera_mask
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
            if getattr(param, "trim_SE3_grad", False):
                # Camera update: d shape (N_imgs * 9,) -> (N_imgs, 9)
                # Layout after SE3 trimming: [se3(6), df, dk1, dk2]
                d = d.view(param.shape[0], -1).clone()
                # Freeze pose + intrinsics for constant cameras.
                d[self.constant_camera_mask] = 0
                # Freeze focal length globally if not refining.
                if not self.refine_focal_length:
                    d[:, 6] = 0
                # Freeze distortion params globally if not refining.
                if not self.refine_extra_params:
                    d[:, 7:] = 0
                param[..., :7] = pp.SE3(param[..., :7]).add_(
                    pp.se3(d[..., :6])
                )
                if param.shape[-1] > 7:
                    param[:, 7:] += d[:, 6:]
            else:
                # Point update: d shape (N_pts * 3,) -> (N_pts, 3)
                d = d.view(param.shape).clone()
                d[self.constant_point_mask] = 0
                param.add_(d)


def solve(
    camera_params_np,
    points_3d_np,
    points_2d_np,
    camera_indices_np,
    point_indices_np,
    constant_camera_mask_np,
    constant_point_mask_np,
    options_dict,
):
    """Entry point called from C++ BaeBundleAdjuster::Solve().

    All array inputs are numpy arrays (flat or shaped). Returns a dict:
      camera_params:  optimized (N_imgs, 10) numpy array
      points_3d:      optimized (N_pts, 3)   numpy array
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
    else:
        device = "cpu"

    max_iterations = options_dict.get("max_num_iterations", 100)
    refine_focal_length = options_dict.get("refine_focal_length", True)
    refine_extra_params = options_dict.get("refine_extra_params", True)

    # Convert numpy arrays to tensors and reshape.
    camera_params = torch.tensor(
        camera_params_np, dtype=torch.float64, device=device
    ).reshape(-1, 10)
    points_3d = torch.tensor(
        points_3d_np, dtype=torch.float64, device=device
    ).reshape(-1, 3)
    points_2d = torch.tensor(
        points_2d_np, dtype=torch.float64, device=device
    ).reshape(-1, 2)
    camera_indices = torch.tensor(
        camera_indices_np, dtype=torch.long, device=device
    )
    point_indices = torch.tensor(
        point_indices_np, dtype=torch.long, device=device
    )
    constant_camera_mask = torch.tensor(
        constant_camera_mask_np, dtype=torch.bool, device=device
    )
    constant_point_mask = torch.tensor(
        constant_point_mask_np, dtype=torch.bool, device=device
    )

    # Build model and optimizer.
    model = ColmapReproj(camera_params, points_3d).to(device)
    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    solver = PCG(tol=1e-4, maxiter=250)
    optimizer = ColmapLM(
        model,
        constant_camera_mask=constant_camera_mask,
        constant_point_mask=constant_point_mask,
        refine_focal_length=refine_focal_length,
        refine_extra_params=refine_extra_params,
        strategy=strategy,
        solver=solver,
        reject=30,
    )

    input_data = {
        "points_2d": points_2d,
        "camera_indices": camera_indices,
        "point_indices": point_indices,
    }

    # Compute initial cost.
    initial_cost = model.loss(input_data, None).item()

    # Run optimization loop.
    num_iterations = 0
    final_cost = initial_cost
    converged = False
    prev_cost = initial_cost
    for _ in range(max_iterations):
        loss = optimizer.step(input_data)
        num_iterations += 1
        final_cost = loss.item()
        rel_change = abs(prev_cost - final_cost) / max(prev_cost, 1e-12)
        if rel_change < 1e-6:
            converged = True
            break
        prev_cost = final_cost

    return {
        "camera_params": model.pose.detach().cpu().numpy(),
        "points_3d": model.points_3d.detach().cpu().numpy(),
        "num_iterations": num_iterations,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "converged": converged,
    }