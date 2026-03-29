4. No Iterative BA + Track Filtering (Critical)
InstantSFM runs 3 rounds of BA with progressive filtering:


for iter in range(3):
    ba_engine.Solve(...)
    UndistortImages(cameras, images)
    FilterTracksByReprojectionNormalized(..., max_error * max(1, 3 - iter))
    # iter 0: threshold = 3x → removes worst outliers
    # iter 1: threshold = 2x → tightens
    # iter 2: threshold = 1x → final pass
Our code: single BA call with all observations. No filtering between rounds. This means:

The first BA iteration fights 900K observations including outliers
No pruning ever happens — memory never decreases
InstantSFM's problem size shrinks with each round
This is probably the single biggest reason InstantSFM scales while we OOM. It's not about sparse Jacobians or Schur complements — it's about not throwing the whole problem at one BA call.

5. PCG Solver Tolerance
InstantSFM:


sparse_solver = PCG(tol=1e-5)  # no maxiter — uses default
Our code:


solver = PCG(tol=1e-4, maxiter=250)
maxiter=250 forces PCG to run up to 250 inner iterations per LM step. For a large system, this allocates memory for 250 intermediate vectors. InstantSFM doesn't set maxiter, relying on the default (likely much lower).

6. test_bae.py vs InstantSFM Pipeline
The test calls colmap global_mapper --GlobalMapper.ba_backend BAE, which invokes COLMAP's global mapper. COLMAP's global mapper does its own BA calls, but:

COLMAP's global_mapper (GLOMAP) was designed for Ceres, not BAE
The iterative BA+filtering pattern is inside GLOMAP's C++ code, not controlled by our Python solver
Our BAE solver gets called with the full observation set that GLOMAP gives it
In contrast, InstantSFM controls the entire pipeline in Python — it calls BA, filters, calls BA again. This is a pipeline-level difference, not a solver-level one.