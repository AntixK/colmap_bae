# Skill — Don't regress the load-bearing fixes

These are the fixes that took the longest to find and have the biggest effect. They are all in the editable file set; "simplifying" any of them without measured evidence will regress the project.

## 1. Huber kernel correction in `_debug_step` (info.md §3.31)

**Files**: `bae_solver.py:_apply_huber_correction`, `_debug_step`

**What it does**: applies Triggs FastTriggs `w_i = sqrt(ρ'(s_i))` to `(R, J)` before the GN normal equations. The bae library's `LM.step()` overrides pypose's `_step_dense` and drops the corrector — that override is the dead-code bug.

**Without it**: BAE's GN step descends pure L2; accept/reject uses kerneled loss via `RobustModel.loss`. Inconsistent on outlier-heavy distributions. Kushimoto, mihama, bridge stall. This is the single biggest find in the project.

**Don't remove**. Don't simplify the `_apply_huber_correction` math. The FastTriggs formula is specifically Ceres-compatible.

## 2. `A_base.clone()` per LM attempt (info.md §3.31 follow-up)

**File**: `bae_solver.py:_debug_step`, lines computing the damped LHS.

**What it does**: each rejected attempt rebuilds `A_attempt` from a fresh clone of the clamped `JᵀJ`, then applies `(1 + pg["damping"])` to its diagonal — so the damping is **per attempt**, not cumulative across attempts.

**Without it**: rejects multiply the diagonal by cumulative `∏(1 + λᵢ)` instead of `(1 + λₖ)`. After 3 rejects, the effective damping is multiplicatively larger than intended → step collapses → step norm underflows → spurious "damping saturation."

The catastrophic camera-only polish failure (step_norm `5.5e-4 → 1.9e-5 → 8.7e-8 → 2.5e-11` across rejects) was this bug. Now fixed.

## 3. Pre-centering observations on the C++ side (info.md §1.5, §3.9)

**File**: `bundle_adjustment_bae.cc:SetupProblem`, where 2D observations are pushed onto `points_2d_`.

**What it does**: subtracts `(cx, cy)` from each 2D observation, so the Python projection function `_distort_and_project` doesn't need to know the principal point. Mathematically equivalent to InstantSfM's `+ pp` in `reproject_simple_radial_no_depth`.

**Don't change** unless you also update the Python projection. The two halves must agree.

## 4. SIMPLE_RADIAL `k2 = 0` clamp (info.md §3.9, §3.19)

**Files**: `bundle_adjustment_bae.cc` (extraction and writeback both force `ip[2] = 0` for `SIMPLE_RADIAL`); `bae_solver.py` (projection function uses only `(f, k1)`, optimizer's intrinsics is `(n_cams_compact, 2)`).

**Without it**: BAE optimizes a phantom `k2` that COLMAP's `SIMPLE_RADIAL` model can't store, then writeback discards it → BAE's view and Reconstruction's view diverge. Originally manifested as bridge/soil 50% point retention loss.

## 5. Pybind11 1D-array stride fix (info.md §1.1)

**File**: `bundle_adjustment_bae.cc:make_1d` lambda.

**What it does**: passes explicit strides via `py::buffer_info(ptr, sizeof(T), ..., {N}, {sizeof(T)})`. The default `py::array_t<T>(ShapeContainer{N}, ptr)` constructor in this pybind11 version emits `strides[0] = 0`, broadcasting scalar — every index Python-side reads as zero.

**Without it**: every BAE experiment optimizes a rank-1 problem (all observations point at image 0 / point 0). Months of speculation invalidated.

**Sanity check** the probe diagnostics each run for this regression — the C++↔Python boundary is the highest-leverage place to validate.

## 6. Stationary gauge fix (info.md §3.20, §3.24)

**Files**: `bundle_adjustment_bae.cc:SelectGaugeConstraints` (chooses anchor + second image with largest-baseline component), passes the choice to Python via the options dict. `bae_solver.py:_build_compact_constraint_masks` compresses the corresponding columns out of `J`.

**What it does**: removes the 7 gauge DoFs (6 from anchor pose, 1 translation component from second image along the dominant baseline axis). Matches Ceres's `FixGaugeWithTwoCamsFromWorld` semantics.

**The fragile bit**: anchor-pair selection. Picking the **largest-baseline pair** (O(N²) sweep) is critical — picking the **first valid pair** (Ceres's literal C++ approach, fine for direct solvers, bad for PCG) gives microscopic locked baselines on sequential datasets and lets scale drift.

**Don't replace** with first-valid-pair selection without measuring on bridge / kushimoto.

## 7. iter_BA filter loop honoring backend (info.md §1.2)

**File**: `global_mapper.cc:IterativeRetriangulateAndRefine` — `custom_ba_options = ba_options;` (copy, not default-construct).

**Without it**: `custom_ba_options` was being default-constructed with `backend=CERES`, so `mapper.IterativeGlobalRefinement` silently used Ceres for the inner refinement rounds even when the user asked for BAE. The reported "BAE time" was BAE-outer + Ceres-inner.

## What to do if you think a load-bearing fix is wrong

1. Read the linked `info.md` section. The fix has documented evidence.
2. Run the benchmark **with** the fix and **without** it. Measure the regression.
3. If you have a measured regression that suggests the fix should be removed, propose it via [skills/propose-experiment.md](propose-experiment.md). Don't silently delete.
