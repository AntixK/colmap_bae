# Rule 06 — BAE runs on GPU only

BAE must run on GPU. No CPU fallback. This is a project-level constraint from
[issue.md](../../issue.md).

## Concrete consequences

- `bae_solver.py` may freely use CUDA-only ops (Triton via `bae.sparse.warp_wrappers`, batched `torch.linalg` on `cuda:0`, sparse-CSR ops).
- `BaeBundleAdjuster::Solve` in `bundle_adjustment_bae.cc` requires `Py_IsInitialized()` and `torch.cuda.is_available()`. The check at line 78–88 enforces this; do not weaken it.
- `BaeBundleAdjustmentOptions::use_gpu = true` is the only supported mode. A `WARN` is logged at `bundle_adjustment_bae.cc:215-217` if a caller tries to set `use_gpu=false` — the code forces it back to `true`.

## What this rules out

- Don't add a `if not cuda.is_available()` CPU fallback path in `bae_solver.py`.
- Don't propose mixing CPU and GPU solves for "fast small problems."
- Don't propose a hybrid where BAE uses CPU torch for the LM step.

## Performance corollary

BAE's win is GPU-bulk-throughput. The expected speed envelope is:

- **Large datasets (> ~500 images)**: BAE should be substantially faster than Ceres.
- **Small datasets (< ~150 images)**: Ceres-on-CPU may be faster per call because BAE has fixed per-call GPU overhead (Python interpreter, pybind11, sparse-tensor setup). This is acceptable; the constraint is correctness across all sizes, and speed on the *large* end.

If a BAE optimization makes large datasets slower without a measured quality gain, revert it. The whole point of GPU bulk processing is to win on scale.
