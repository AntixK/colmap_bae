# Skill — Authoring BAE compute graphs

Use this when defining or modifying BAE forward models in `bae_solver.py`. Pulled from `ref.md` + direct reading of `sair-lab-bae-8a5edab282632443 (1).txt`.

## Mental model

The BAE sparse autograd (`bae.autograd.graph`) classifies every traced op into one of three kinds:

- **`index`** — sets sparse block-column layout. Use tensor indexing like `self.pose[camera_idx]`.
- **`map`** — sets Jacobian block values. Use `@map_transform` for vectorized residual functions.
- **`cat(dim=0)`** — structural op that splits and routes upstream Jacobians. Used for gauge fixes / fixed-state splits.

Only these three kinds are supported as the final op of the residual trace.

## Authoring recipe

1. Wrap every optimizable state as `nn.Parameter(TrackingTensor(data))`.
2. Mark SE(3) parameters with `param.trim_SE3_grad = True` if the stored layout starts with a 7-D quaternion pose.
3. Define per-factor residual functions with `@map_transform`.
4. In `forward()`, gather participating states by tensor indexing.
5. Combine factor groups with `torch.cat(..., dim=0)` if needed.
6. Return the residual tensor; the bae library will derive the sparse Jacobian.

## Canonical BAL residual

```python
class ColmapReproj(nn.Module):
    def __init__(self, extrinsics, intrinsics, points_3d):
        super().__init__()
        self.extrinsics = nn.Parameter(TrackingTensor(extrinsics))
        self.intrinsics = nn.Parameter(TrackingTensor(intrinsics))
        self.points_3d  = nn.Parameter(TrackingTensor(points_3d))
        self.extrinsics.trim_SE3_grad = True  # 7→6 columns per pose

    def forward(self, points_2d, image_indices, camera_indices, point_indices):
        points_proj = colmap_project(
            self.points_3d[point_indices],   # index op
            self.extrinsics[image_indices],  # index op
            self.intrinsics[camera_indices], # index op
        )
        return points_proj - points_2d        # map op (subtraction is whitelisted)
```

## Hard constraints (from `bae/autograd/graph.py` and `function.py`)

- The final residual must end in `map`, `index`, or `cat(dim=0)`.
- `map_transform` functions must be compatible with `jacrev` and effectively batch-vectorized via `vmap` — write with trailing-dim indexing (`[..., :2]`, `dim=-1`).
- Only `torch.cat(dim=0)` is supported; other concat dims fail.
- A parameter that never appears in observations produces empty Jacobian block-columns → solver failure.
- `trim_SE3_grad = True` shrinks 7-D stored params to 6-D optimized columns (or 10-D to 9-D for pose+3-intrinsics).

## Validation checklist (run after every modification)

- [ ] Residual returns shape `(N, 2)` for BA, `(N, 6)` for PGO, etc.
- [ ] Every returned Jacobian block is `torch.sparse_bsr`.
- [ ] `col_indices()` matches observation connectivity exactly.
- [ ] No empty parameter block-columns when every variable is supposed to be constrained.
- [ ] `diag(JᵀJ)` is strictly positive for every constrained column.

## Gauge-fixed BAL pattern (from `ref.md`)

If the first camera pose is fixed, build a `cat(dim=0)` graph:

```python
def forward(self, points_2d, camera_indices, point_indices, camera_fixed):
    camera_se3 = torch.cat([camera_fixed, self.pose_rest], dim=0)
    pred = project_with_se3_and_intrinsics(
        self.points_3d[point_indices],
        camera_se3[camera_indices],
        self.intrinsics[camera_indices],
    )
    return pred - points_2d
```

Backward routes Jacobian columns for camera 0 into the fixed branch (no update) and cameras 1..N-1 into `pose_rest` (optimized).

## Read these to extend the system

- `ref.md` — BAL standard / gauge-fixed / split-state patterns
- `ref.md` — PGO pattern
- `bae/autograd/function.py:map_transform` — the actual decorator
- `bae/autograd/graph.py:jacobian` — the backward Jacobian builder
