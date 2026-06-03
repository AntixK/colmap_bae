# Rule 01 — Editable files

You may edit **only these four files**:

- [python/pycolmap/bae_solver.py](../../python/pycolmap/bae_solver.py)
- [src/colmap/estimators/bundle_adjustment_bae.cc](../../src/colmap/estimators/bundle_adjustment_bae.cc)
- [src/colmap/estimators/bundle_adjustment_bae.h](../../src/colmap/estimators/bundle_adjustment_bae.h)
- [src/colmap/sfm/global_mapper.cc](../../src/colmap/sfm/global_mapper.cc)

Plus narrative files (`info.md`, `ceres.md`, `problem.md`, etc.) and the benchmark driver (`run_benchmark.py`) when they are explicitly part of the task.

## Files that are NOT editable

- `src/colmap/sfm/global_mapper.h`
- `src/colmap/estimators/bundle_adjustment_ceres.{cc,h}`
- `src/colmap/controllers/option_manager.cc`
- The `bae` library, `pypose`, any third-party source
- Any file not in the list above

## When the fix appears to need a non-editable file

Stop. Restate the problem. Usually the change can be expressed inside the editable set (for example, as a replacement step inside `_debug_step` in `bae_solver.py`).

If it truly cannot, **ask the user before touching anything**. Do not proactively expand the scope.

## Bad

```
> User: "Can you make Ceres use ITERATIVE_SCHUR?"
> Assistant: *edits bundle_adjustment_ceres.cc without asking*
```

## Good

```
> User: "Can you make Ceres use ITERATIVE_SCHUR?"
> Assistant: "That requires editing bundle_adjustment_ceres.cc, which is not in
> the editable set. The minimum change would be a 5-line override in
> CreateSolverOptions. Want me to expand the scope just for this experiment,
> or find another path?"
```

Violating this rule has happened in this project before. Each violation cost user trust. Always confirm before exceeding the scope.
