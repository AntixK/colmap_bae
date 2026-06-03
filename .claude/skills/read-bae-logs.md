# Skill — Reading BAE run logs

The single most efficient way to diagnose a BAE failure is to grep the run log for the right lines. This file maps log signatures to diagnoses.

## Stage residual trajectory

These come from `global_mapper.cc:LogReprojectionResiduals`. They're the outer-loop view of optimizer progress.

```
[reproj iter_ba.start]            — post-global-positioning, pre-iter_BA
[reproj iter1.fixed_rot]          — after first iter_BA's fixed-rotation stage
[reproj iter1.full]               — after first iter_BA's full BA stage
[reproj iter2.fixed_rot]
[reproj iter2.full]
[reproj iter_ba.final_filter]     — after the post-iter_BA filter pass
[reproj iter_ba.final]
[reproj retri.post_triangulate]   — after retriangulation, before refinement
[reproj retri.refinement_done]    — after IterativeGlobalRefinement
[reproj retri.final_ba]           — final BA pass
[reproj retri.final_filtered]     — final state
```

Each prints percentiles in px and normalized units. Compare `iter1.full p50` between BAE and Ceres for the first useful quality signal.

## LM trajectory inside one BA call

From `bae_solver.py:_debug_step`:

```
[BAE] lm iter   1 raw pose:diag(JTJ)[p10=... p50=... p90=... max=... spread=...]
                       intr:diag(JTJ)[...]
                       points:diag(JTJ)[...]
[BAE] lm scaling: raw_p90=... raw_max=... p10=... p50=... p90=... max=...
[BAE] lm iter   1 scaled pose:diag(JTJ)[...]  ← should be ~1.0 if scaling is working
[BAE] lm iter   1 huber: inliers=N% w[min=... p10=... p50=... p90=... max=...] robust_cost=...
[BAE] pcg solve: |b|=... |Ax-b|=... rel=...  ← PCG residual
[BAE] lm iter   1 attempt 1 pose:n=N|JTr|=...|D|=...|D|max=... intr:... points:...
[BAE] lm iter   1 attempt 1: last=... new=... actual_reduction=... predicted_reduction=...
       quality=... step_norm=... step_max=... damping=A->B reject_count=N
       accepted=True/False accepted_zero_step=... accepted_tiny_step=... damping_saturation=...
[BAE] iter   1  cost=...
```

## Exit reasons

```
[BAE] _run_ba exit: reason={func_tol|loss_repeat|max_iter} n_it=N/MAX
      cost_first=... cost_last=... cost_drop_total=... windowed_imp=...
```

Decoding:

- `reason=max_iter` with `cost_drop_total > 0` → iter-budget-bound; cost was still descending. Raising `max_num_iterations` may help.
- `reason=func_tol` with `windowed_imp ≈ func_tol` (5e-4 default) → plateau. Tightening `func_tol` will not help (info.md §3.33).
- `reason=loss_repeat` → numerical underflow or damping saturated; check `damping_saturation=True` in the last few attempts.

## Probe diagnostics

From `bundle_adjustment_bae.cc:LogProbe*`. Same observations traced across boundaries:

```
[BAE probe pre_python_arrays]                  ← C++ flat buffers, BAE projection
[BAE probe pre_python_arrays_colmap_camera]    ← C++ flat buffers, COLMAP Camera::ImgFromCam
[BAE probe pre_python_reconstruction]          ← Reconstruction object via image.ProjectPoint

[BAE probe post_python_arrays]                 ← post-solve, BAE view
[BAE probe post_python_arrays_colmap_camera]   ← post-solve, COLMAP view (post-writeback would-be)
[BAE probe post_extrinsics_writeback]          ← Reconstruction after extrinsics writeback
[BAE probe post_intrinsics_writeback]          ← after intrinsics writeback
[BAE probe post_points_writeback]              ← after points writeback (final state)
```

**The invariant** (must hold post §3.9 fix):

```
post_python_arrays  ==  post_python_arrays_colmap_camera  ==  post_points_writeback
```

If they disagree, the bug is in the data transfer / projection convention, **not the optimizer**. See `rules/05-cross-language-boundary.md`.

## What good iter1.full looks like (ignatius, healthy)

- `lm iter 1 attempt 1: quality > 0.5, accepted=True, damping doesn't grow`
- Cost drops by 50%+ in first 5 iters.
- Exit reason `func_tol` with `windowed_imp < 5e-5` (real convergence).
- iter1.full p50 drops by ≥30% from iter1.fixed_rot p50.

## What broken iter1.full looks like (kushimoto, current state)

- `lm iter 1 attempt 1: quality ≈ 0.05-0.2, accepted=True, damping fixed`
- Quality decays exponentially: `0.2 → 0.1 → 0.05 → 0.02 → 0.002 → reject` (info.md §3.32).
- Exit reason `func_tol` with `windowed_imp ≈ 4-5e-5` (plateau, not convergence).
- iter1.full p50 unchanged from iter1.fixed_rot p50 (3.677 vs 3.666).
- Inlier fraction stable (no observations promoted from outlier to inlier).

## Useful grep patterns

```bash
# iter1.full trajectory
grep -n "iter_ba.start\|iter1.fixed_rot\|iter1.full\|iter_ba.final" run.log

# Exit reasons
grep -n "_run_ba exit" run.log

# LM trajectory inside iter1.full (assumes the call is after iter1.fixed_rot)
awk '/active dofs.*constant_rotation=False/{found=1} found && /lm iter/' run.log

# Probe equality check
grep -n "post_python_arrays\|post_python_arrays_colmap_camera\|post_points_writeback" run.log

# Catastrophic damping
grep -n "damping_saturation=True\|reject_count=29\|reject_count=30" run.log
```

## Don't skip the probe check

If a kushimoto-like failure appears on a **new** dataset, the first move is **not** "tune the LM." It is "grep for probe equality." A new boundary bug (like §1.1 stride or §3.9 fake-k2) would look identical to a convergence failure from the outside.
