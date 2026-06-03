# Skill — Reading InstantSfM as a data point (not an oracle)

InstantSfM (`InstantSfM/instantsfm/`, full source dump in `cre185-instantsfm-8a5edab282632443.txt`) is the closest existing project to ours. Same `bae` library, same pypose LM. Useful to grep when answering "is this the canonical way?" — but **see Rule 07**: it is not an oracle.

## The TorchBA setup — what to copy literally

`InstantSfM/instantsfm/processors/bundle_adjustment.py:TorchBA.SolveSingle`:

```python
# data prep (relevant fields):
image_extrs = pp.mat2SE3(images.world2cams[idx]).tensor()  # SE3 quat layout
camera_intrs_list = [torch.tensor(camera.params) for ...]
camera_pps = camera_intrs[..., pp_indices]                  # principal points
camera_intrs = camera_intrs[..., remaining_indices]         # remove pp from optimized intrinsics

# Model
model = ReprojectionModel(image_extrs, camera_intrs, points_3d, cost_fn,
                          optimize_intrinsics=optimize_intrinsics)

# Optimizer
strategy = pp.optim.strategy.TrustRegion(radius=1e4, max=1e10, up=2.0, down=0.5**4)
sparse_solver = PCG(tol=1e-5)
huber_kernel = Huber(BUNDLE_ADJUSTER_OPTIONS['thres_loss_function'])
optimizer = LM(model, strategy=strategy, solver=sparse_solver,
               kernel=huber_kernel, reject=30)
```

`InstantSfM/instantsfm/utils/optimization_models.py:ReprojectionModel.forward`:

```python
loss = self.cost_fn(
    self.points_3d[point_indices],
    self.extrinsics[image_indices],
    self.intrinsics[camera_indices],
    camera_pps[camera_indices],  # principal points passed as buffer, not optimized
)
loss = loss - points_2d
return loss
```

`InstantSfM/instantsfm/utils/cost_function.py:reproject_simple_radial_no_depth`:

```python
points_proj = rotate_quat(points, extrinsics)            # bae.utils.ba.rotate_quat
points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)
f = intrinsics[..., -2].unsqueeze(-1)
k = intrinsics[..., -1].unsqueeze(-1)
r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
points_proj = points_proj * (1 + k * r2) * f + pp        # ← principal point added back here
return points_proj
```

## Things this tells you

1. **They split (f, k1) from (cx, cy)**: principal points are passed as a separate non-optimized buffer. We do the equivalent on the C++ side by pre-centering observations around `(cx, cy)`. Mathematically equivalent.
2. **They use `radius=1e4`**: starts with very low damping. Our default in `bae_solver.py` is `0.3` for full BA (set in §3.14 to avoid first-step instability). Both have measured failure modes.
3. **They use `Huber(BUNDLE_ADJUSTER_OPTIONS['thres_loss_function'])`**: a configurable kernel scale. Their default config has `thres_loss_function: 1.0` matching our `kernel_delta = 1.0` (matching Ceres's `HuberLoss(1.0)`).
4. **They use `reject=30`**: matches ours.
5. **They use `up=2.0, down=0.5**4 = 0.0625`**: faster radius shrink on reject than our `down=0.5`. Worth measuring; not necessarily worth copying — see Quirk 5 in `skills/bae-library-quirks.md`.

## Things this hides

1. **Same kernel-drop bug**: their `LM.step()` is the same `bae` library override. The `Huber(delta=1.0)` is dead code in their GN path too. (Rule 07.)
2. **No fixed_rotation stage**: they call `ba_engine.Solve(...)` directly in a 3-iter loop (`global_mapper.py:147-152`) with no fixed-rot pre-stage. **Falsified as a fix on its own** for our datasets in info.md §3.32 (Change 1 reverted).
3. **No probe diagnostics**: they don't verify cross-language data agreement. If they had a `pybind11` stride bug like our §1.1, they'd hit the same months-long debugging trap.
4. **No condition-number measurement**: they accept the textbook κ-narrative implicitly. `ceres.md §13.10` documents the gap between theory and our measurements.

## Filter cadence (`controllers/global_mapper.py:147-151`)

```python
for iter in range(3):
    ba_engine.Solve(cameras, images, tracks, ...)
    UndistortImages(cameras, images)
    FilterTracksByReprojectionNormalized(
        cameras, images, tracks,
        config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'] * max(1, 3 - iter))
```

- Iter 0: filter at `3 · 1e-2 = 3e-2` normalized
- Iter 1: filter at `2 · 1e-2`
- Iter 2: filter at `1 · 1e-2`

`max_reprojection_error = 1e-2` in normalized units (`config/colmap.py:13`). COLMAP's own global mapper uses the same `3x/2x/1x` schedule via `IterativeBundleAdjustment` (`global_mapper.cc:572-583`).

So our pipeline already does what InstantSfM does between BA outer iters. The `bae_solver.py` pre-BA filter at `norm < 0.10` is *additional* and is a no-op on most of our datasets (info.md §3.34).

## Bad

```
"InstantSfM uses radius=1e4 and that works for them, so let's set ours to 1e4."
```

## Good

```
"InstantSfM uses radius=1e4. We tested radius=1e4 → 0.3 in info.md §3.14
because radius=1e4 caused first-step instability on bridge. The instability
was symptomatic of the kernel-drop bug (§3.31). Now that kernel is corrected,
re-testing radius=1e4 is reasonable — but we still need a measurement, not
a copy."
```
