# Skill — Use Ceres as the canonical reference

`ceres.md` is the verified, line-cited description of how Ceres BA works in this codebase. Reach for it before guessing about Ceres behavior. The Ceres source itself is in `ceres-solver-ceres-solver-8a5edab282632443.txt`.

## When to consult

- "Would Ceres do X?" — read `ceres.md §13` first.
- "How does Ceres pick the linear solver?" — `ceres.md §13.3 / §13.8` + `bundle_adjustment_ceres.cc:194-204`.
- "What's Ceres's LM damping look like?" — `ceres.md §13.2` + `levenberg_marquardt_strategy.cc`.
- "How does Schur reduction actually work?" — `ceres.md §13.4-13.5` + `schur_eliminator_impl.h`.
- "Which preconditioner pairs with which solver?" — `ceres.md §13.6` + `iterative_schur_complement_solver.cc`.

## Quick facts that surprised this project once

These are documented in `ceres.md` because the assistant previously got them wrong from memory.

1. **`global_mapper.h:38-51` hardcodes the Ceres BA settings** for the global mapper, with `auto_select_solver_type = false` and `linear_solver_type = SPARSE_SCHUR` + `preconditioner_type = CLUSTER_TRIDIAGONAL`. The image-size auto-select logic in `bundle_adjustment_ceres.cc:194-204` is **dead** on this code path. (`ceres.md §13.8`, corrected entry.)

2. **`jacobi_scaling` is computed once at iter 0** and reused across all outer iterations: `s_j = 1 / (1 + ‖J_{:,j}‖₂)`. Not per-iter. (`ceres.md §13.1`.)

3. **Ceres's LM damping is Marquardt-scaled**: `D = sqrt(clamp(diag(J'J), 1e-6, 1e32) / radius)`. Augments as `J'J + D'D`, additive (not multiplicative on diag). (`ceres.md §13.2`.)

4. **Nielsen cubic rule on accept**:
   ```
   radius = radius / max(1/3, 1 - (2*quality - 1)^3)
   decrease_factor = 2
   ```
   On reject:
   ```
   radius = radius / decrease_factor
   decrease_factor *= 2
   ```
   (`ceres.md §13.2`, `levenberg_marquardt_strategy.cc:StepAccepted/StepRejected`.)

5. **Ceres's PCG only uses q_tolerance, not r_tolerance**:
   ```cpp
   // levenberg_marquardt_strategy.cc
   solve_options.r_tolerance = -1.0;
   ```
   Truncated Newton. (`ceres.md §13.5`.)

## The empirical caveat (don't skip this)

`ceres.md §13.10` documents the κ(S) ≤ κ(H) inequality is **provable** but **not measured** on our datasets. The diag(JᵀJ) lower bound on κ shows roughly the same value across all 8 datasets, with **ignatius (passes) at the high end and mihama (fails) at the low end**. The textbook narrative "BA fails because H is ill-conditioned, Schur fixes it" is theoretically correct but does not predict our failure pattern.

This is a *real caveat*, not a hedge. Don't recommend Schur as "the fix" without first measuring κ on our datasets.

## The side-by-side table (`ceres.md §13.9`)

| Aspect | Ceres @ ≤1000 imgs | Ceres @ >1000 imgs (in theory) | BAE (current) |
|---|---|---|---|
| Approach | Direct Schur + sparse Cholesky | Iterative Schur (PCG on S) | PCG on full H |
| Diag scaling | `1/(1+‖J_:j‖)` once | same | recomputed every iter, clamped |
| LM damping | `D = sqrt(diag(J'J)/radius)` | same | `(1+λ)·diag(J'J)` |
| Trust-region update | Nielsen cubic | same | pypose `TrustRegion(up=2, down=0.5)` |
| Linear solve | Cholesky on S (exact) | PCG on S with SCHUR_JACOBI | PCG on H |
| Truncated Newton | q_tolerance from eta=1e-1 | same | `r_tolerance=1e-5` |

Note: per #1 above, *Ceres in our actual benchmark uses SPARSE_SCHUR on every dataset including mihama*, because `global_mapper.h:45` disables auto-select.

## Bad

```
"Ceres uses ITERATIVE_SCHUR on mihama because num_images > 1000."
```

(Wrong. The auto-select is dead on global_mapper. Ceres uses SPARSE_SCHUR.)

## Good

```
"Per ceres.md §13.8, global_mapper.h:45 disables auto-select. Ceres uses
SPARSE_SCHUR + CLUSTER_TRIDIAGONAL on all dataset sizes in our benchmark."
```
