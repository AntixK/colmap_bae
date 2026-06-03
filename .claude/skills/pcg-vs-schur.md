# Skill — PCG vs Schur, in one page

Pulled from `pcg.md` and `ceres.md §13.4-13.5`. Use when discussing the linear solver of any BA-style problem.

## The system

Inside one LM step:

```
(JᵀJ + λD) δ = -Jᵀr
```

Block-partition with cameras first, points second:

```
H = | U   W |     δ = | δ_c |     g = | g_c |
    | Wᵀ  V |         | δ_p |         | g_p |
```

- `U`: cameras × cameras — small (~few thousand DoFs)
- `V`: points × points — block-diagonal with one 3×3 block per point — large total (~1M DoFs) but trivially invertible
- `W`: cameras × points coupling

## PCG-on-full-system (what BAE does today)

Iterative solve on `H δ = g`. Per iter: one sparse matvec `H · v`. Convergence rate bound:

```
‖δ_k − δ*‖_H  ≤  2 · ((√κ(H) − 1)/(√κ(H) + 1))^k · ‖δ_0 − δ*‖_H
```

Per-component error bound on ill-conditioned `H`:

```
‖δ_pcg − δ_true‖  ≤  κ(H) · ‖H·δ_pcg − g‖ / ‖H‖
```

PCG can hit `tol=1e-5` on residual norm and still have per-component step
error orders of magnitude larger than `tol` in directions corresponding to
small singular values of `H`. This is the §3.29 finding.

## Schur reduction (what Ceres does)

Algebraic transformation, *not* a solver. From the block system:

```
δ_p = V⁻¹ (g_p − Wᵀ δ_c)        ← back-substitution
(U − W V⁻¹ Wᵀ) δ_c = g_c − W V⁻¹ g_p   ← reduced camera system
```

The reduced LHS matrix `S = U − W V⁻¹ Wᵀ` is **the Schur complement**.

After forming `S`, you still need to solve `S δ_c = ...`. Options:

- **Direct**: sparse Cholesky on `S` (Ceres's `SPARSE_SCHUR`, used for ≤1000 imgs in their auto-select, hardcoded on global_mapper). Exact step in `δ_c`.
- **Iterative**: PCG on `S` (Ceres's `ITERATIVE_SCHUR`). Better-conditioned than PCG on `H` *in theory*.

`V` is block-diagonal with 3×3 blocks, so `V⁻¹` is computed by inverting
each block independently — cheap. The implicit-matvec trick (`ceres.md §13.5`)
evaluates `S · x` via four sparse matvecs without ever forming `S`.

## What's provable, what's not

**Provable**: `κ₂(S) ≤ κ₂(H)` for any SPD `H`. The bound is tight when `V`
dominates `H`'s conditioning, loose otherwise. See `pcg.md` for the proof
sketch.

**Empirical, well-documented**: in BAL-scale problems with high-variance point
visibility, `V` typically dominates and the gap is several orders of magnitude.
Cited e.g. in Agarwal et al. 2010 (BAL) and Triggs et al. 2000.

**Empirical, on our datasets**: `ceres.md §13.10` — the diag-range lower bound
on κ(H) is roughly similar across all 8 datasets, with ignatius (passes) at
the high end and mihama (fails) at the low end. The conditioning narrative
**does not predict** our failure pattern. Real κ estimation (Lanczos or
Hager's 1-norm) is **not yet done** on our matrices.

## The honest summary

- "PCG converges to tolerance" rules out one failure mode (info.md §3.18).
- "Per-component error on ill-conditioned H" remains a candidate (info.md §3.29).
- "Schur eliminates the ill-conditioning" is provable as inequality, unmeasured on us.
- The diag-range proxy refutes the simple κ-tells-everything narrative for our datasets.

**Don't recommend Schur as "the fix" without measuring κ first.** See `rules/08-no-overpromising.md`.
