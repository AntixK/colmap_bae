# Skill — Working with the three source dumps

Three flat-file dumps of dependencies live at the project root. Use them when answering "what does library X do?" — these are the authoritative source, not memory.

## Files

| Path | Contents | Format |
|---|---|---|
| `sair-lab-bae-8a5edab282632443 (1).txt` | The `bae` library source | Concatenated repo, each file preceded by `FILE: path/to/file.py` |
| `ceres-solver-ceres-solver-8a5edab282632443.txt` | The Ceres Solver source | Same format, ~153k lines |
| `cre185-instantsfm-8a5edab282632443.txt` | InstantSfM source | Same format |

## Locating a file inside a dump

```bash
grep -n "^FILE: " <dump.txt> | grep -E "<pattern>"
```

Example: find Ceres's `schur_eliminator_impl.h`:

```bash
grep -n "^FILE: " ceres-solver-ceres-solver-8a5edab282632443.txt | grep "schur_eliminator"
```

Returns the line number where the file begins. Pass that to `Read` with an `offset`.

## High-value index of each dump

### bae library

- `bae/optim/optimizer.py` — `LM` class. The override of `step()` that drops the kernel corrector (Quirk 1 in `bae-library-quirks.md`).
- `bae/autograd/graph.py` — `jacobian` builder for sparse BSR.
- `bae/autograd/function.py` — `TrackingTensor`, `map_transform` decorator.
- `bae/sparse/py_ops.py` — `diagonal_op_` (with a known `UnboundLocalError` quirk, see issue.md).
- `bae/utils/pysolvers.py` — `PCG`.
- `bae/utils/ba.py` — `rotate_quat`.
- `bae/sparse/spgemm.py` — `CuSparse` (sparse-sparse matmul used as `self.mm`).

### Ceres

- `include/ceres/solver.h` — `Solver::Options` defaults.
- `include/ceres/types.h` — `LinearSolverType`, `PreconditionerType` enums.
- `internal/ceres/trust_region_minimizer.cc` — outer loop, `jacobi_scaling`.
- `internal/ceres/levenberg_marquardt_strategy.cc` — LM damping, Nielsen rule.
- `internal/ceres/linear_solver.{cc,h}` — dispatch on `LinearSolverType`.
- `internal/ceres/schur_eliminator.{h,impl.h}` — Schur reduction.
- `internal/ceres/schur_complement_solver.{cc,h}` — DENSE_SCHUR / SPARSE_SCHUR.
- `internal/ceres/iterative_schur_complement_solver.{cc,h}` — ITERATIVE_SCHUR.
- `internal/ceres/implicit_schur_complement.{cc,h}` — matvec without forming S.
- `internal/ceres/schur_jacobi_preconditioner.{cc,h}` — SCHUR_JACOBI.
- `internal/ceres/conjugate_gradients_solver.h` — templated PCG.
- `internal/ceres/visibility_based_preconditioner.{cc,h}` — CLUSTER_JACOBI / CLUSTER_TRIDIAGONAL.
- `internal/ceres/power_series_expansion_preconditioner.{cc,h}` — SCHUR_POWER_SERIES_EXPANSION.

### InstantSfM

- `instantsfm/processors/bundle_adjustment.py` — `TorchBA.SolveSingle/Multi`. The reference BA configuration.
- `instantsfm/utils/optimization_models.py` — `ReprojectionModel` etc.
- `instantsfm/utils/cost_function.py` — `reproject_simple_radial_no_depth`, all camera-model reprojection variants.
- `instantsfm/controllers/global_mapper.py` — outer pipeline (BA → filter → BA loop).
- `instantsfm/config/colmap.py` — `BUNDLE_ADJUSTER_OPTIONS`, `INLIER_THRESHOLD_OPTIONS` defaults.

## Bad — claim from memory

> "InstantSfM uses `Cauchy(1.0)` loss."

(Memory may be wrong. Even if you "remember" the value, verify.)

## Good — grep first

```bash
grep -n "Huber\|Cauchy\|thres_loss_function" cre185-instantsfm-8a5edab282632443.txt | head
```

then quote the line.

## When the dump doesn't have what you need

If a library version differs from the dump's snapshot, `find / -path "*<file>"` may find the actually-installed version. Example:

```bash
find / -path '*/bae/optim/optimizer.py' 2>/dev/null
```

(That's how we caught the kernel-corrector-drop in the installed version.)

## Read this before claiming a quote

`rules/02-no-fabrication.md`. The dumps exist precisely so you don't have
to claim from memory.
