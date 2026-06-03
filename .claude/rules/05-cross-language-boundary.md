# Rule 05 — Verify cross-language data flow

Any data crossing C++ ↔ Python via `pybind11` (or any other binding layer) is a potential silent-corruption boundary. Log it. Trust nothing.

## The §1.1 stride bug — why this rule exists

For months the assistant chased convergence pathologies in BAE. The bug was
a `pybind11` `py::array_t<int>(ShapeContainer{N}, ptr)` constructor producing
`strides=(0,)` (broadcast scalar) for 1-D arrays. Every `image_indices[i]`,
`camera_indices[i]`, `point_indices[i]` Python-side saw was `0`. The
optimizer was solving a rank-1 problem with all observations pointing at
image 0 / point 0.

Every measurement before the fix was meaningless:
- "median residual 500 px" — artifact of rank-1 problem
- "damping caps at 1e6" — artifact
- "kernel scale should be 2·median" — artifact
- "PCG converges to wrong direction" — artifact

The fix is one line. Detecting it required logging `shape`, `dtype`, `strides`,
`unique`, `min`, `max`, `first 10 elements` at the C++↔Python boundary.

## The mandate

Probes are already wired in this codebase. See:

- `bundle_adjustment_bae.cc:LogProbeResidualsFromArrays` ("pre_python_arrays")
- `bundle_adjustment_bae.cc:LogProbeResidualsFromFlatStateUsingCamera`
  ("pre_python_arrays_colmap_camera")
- `bundle_adjustment_bae.cc:LogProbeResidualsFromReconstruction`
  ("post_points_writeback" / "post_extrinsics_writeback" / etc.)
- `bae_solver.py:_log_probe_errors` ("probe pre_opt" / "probe post_opt")

Before suspecting an algorithm, **always confirm these probes agree** at
each boundary:

```
pre_python_arrays  ==  pre_python_arrays_colmap_camera  ==  pre_python_reconstruction
post_python_arrays  ==  post_python_arrays_colmap_camera  ==  post_points_writeback
```

If they don't, the bug is in data transfer (projection convention, writeback,
masks, stride, dtype, layout), not in the optimizer. Fix that first.

## The §1.5 / §3.9 — fake `k2` under `SIMPLE_RADIAL`

Another instance: BAE's intrinsics tensor was 3-wide `(f, k1, k2)` but
COLMAP's `SIMPLE_RADIAL` model has only `(f, k1)`. The optimizer freely moved
`k2` during BA; writeback discarded it. Pre-BA `post_python_arrays` looked
fine because `k2` was zero. After BA `post_python_arrays` and
`post_python_arrays_colmap_camera` disagreed — same state, different
projection models. The fix was clamping `k2` to 0 in the Python projection.

The probe instrumentation caught this. Without it, the symptom looked like
"BAE doesn't converge on bridge" and would have spawned more tuning experiments.

## Bad

```
> User: "iter1.full p50 is still 1.7"
> Assistant: "Let me try a smaller trust-region radius..."
```

(Without first checking probe equality. Could be a writeback bug, not an LM issue.)

## Good

```
> User: "iter1.full p50 is still 1.7"
> Assistant: "Checking probe agreement first.
>   `post_python_arrays p50 = X`
>   `post_python_arrays_colmap_camera p50 = Y`
>   `post_points_writeback p50 = Z`
> If X == Y == Z, the LM is at fault. If X < Y, projection mismatch.
> If Y < Z, writeback bug."
```
