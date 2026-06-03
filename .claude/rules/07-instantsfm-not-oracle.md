# Rule 07 — InstantSfM is one data point, not an oracle

InstantSfM (`InstantSfM/instantsfm/` and `cre185-instantsfm-8a5edab282632443.txt`) uses the **same `bae` library** and the **same `pypose` LM** that we do. It has the same dataset-fragility we do. It is not a "working baseline" that proves anything by itself.

## What InstantSfM actually does

From `InstantSfM/instantsfm/processors/bundle_adjustment.py` and `controllers/global_mapper.py`:

```python
# instantsfm/processors/bundle_adjustment.py:TorchBA.SolveSingle
strategy = pp.optim.strategy.TrustRegion(radius=1e4, max=1e10, up=2.0, down=0.5**4)
sparse_solver = PCG(tol=1e-5)
huber_kernel = Huber(BUNDLE_ADJUSTER_OPTIONS['thres_loss_function'])
optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)
```

```python
# instantsfm/controllers/global_mapper.py:147-151
for iter in range(3):
    ba_engine.Solve(...)
    UndistortImages(cameras, images)
    FilterTracksByReprojectionNormalized(
        cameras, images, tracks,
        config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'] * max(1, 3 - iter))
```

```python
# instantsfm/config/colmap.py:13
'max_reprojection_error': 1e-2  # normalized units
```

Key facts:
- **No pre-BA filter inside the BA call**; filters at `3·1e-2 / 2·1e-2 / 1·1e-2` *between* BA iters.
- **No fixed_rotation BA stage**; just full BA, 3 times.
- **The kernel is passed but never applied** to the GN normal equations — same dead-code bug as our `bae_solver.py` had before the §3.31 fix. The `bae` library's `LM.step()` overrides pypose's `_step_dense` and drops the corrector. `instantsfm` inherits this.

## What this means

"InstantSfM doesn't do X" is **one piece of evidence**, not proof X is unnecessary. They may not do X because:

- They have the same bug we have, and X would fix both.
- They tested without X on a dataset where it doesn't matter.
- They have a different cost function and X isn't applicable.

When you cite InstantSfM, pair it with an independent argument:

## Bad

```
"InstantSfM doesn't have column scaling, so we should remove it."
```

## Good

```
"InstantSfM doesn't have column scaling. Independently, our scaled diag(JᵀJ)
logs show post-scaling spread ≤ 11.16 across all 8 datasets, so the scaling
*is* doing measurable work. Removing it is premature without verifying that
the unscaled spread doesn't hurt PCG convergence on kushimoto / mihama."
```

## The "Ceres works on this dataset" comparison is stronger

Ceres is the actual benchmark target. If you have to choose between
"InstantSfM does X" and "Ceres does X" for evidence, Ceres wins every time.
Ceres is documented (`ceres.md`, especially §13) and its source is in
`ceres-solver-ceres-solver-8a5edab282632443.txt`.
