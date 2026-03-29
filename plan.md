# Plan: Add BAE Bundle Adjustment Backend to COLMAP

## TL;DR
Add BAE (Bundle Adjustment in Eager mode) as an alternative BA solver backend in COLMAP's C++ code. BAE's core (LM optimizer, Jacobian computation) is pure Python/PyTorch, so BaeBundleAdjuster::Solve() embeds a Python interpreter call via pybind11. The C++ side handles data extraction from Reconstruction and writeback; the Python side runs BAE's optimizer. Gated behind `BAE_ENABLED` CMake option.

## Decisions
- Camera models: SimpleRadial and Radial only (maps to BAE's f/k1/k2). NOT SimplePinhole (no distortion params — see Constraint C2).
- No rig support initially (single-sensor trivial-frame only).
- Standard BA only (no PosePriorBundleAdjuster).
- Gauge fixing: exclude constant params from optimization tensors.
- Shared intrinsics: duplicate per-image (user's choice).
- Quaternion reorder: COLMAP [qx,qy,qz,qw,tx,ty,tz] ↔ BAE [tx,ty,tz,qx,qy,qz,qw].
- Loss function: use PyPose's TrustRegion strategy (user's choice).
- Jacobians: let BAE compute them (user's choice).
- Build: optional dependency, BAE_ENABLED CMake flag.
- Writeback: direct copy from tensors back to Reconstruction.

## Constraints & Known Risks

**C1 — Python interpreter dual-mode.** BaeBundleAdjuster::Solve() can be called from:
  (a) colmap CLI binary — no Python interpreter running → must use `py::scoped_interpreter`.
  (b) pycolmap Python module — interpreter already running → `py::scoped_interpreter` will crash.
  **Mitigation:** Check `Py_IsInitialized()` at Solve() entry. If false, create scoped_interpreter. If true, just acquire GIL with `py::gil_scoped_acquire`.

**C2 — BAE has NO principal point (cx,cy).** BAE's project() does: `u = f * r * (-x/z)`. COLMAP's models include cx,cy. BAE's SimpleRadial equivalent is `f, k1, k2` but COLMAP's SimpleRadial is `[f, cx, cy, k]` and Radial is `[f, cx, cy, k1, k2]`.
  **Mitigation:** BAE uses centered pixel coordinates. On the C++ side, transform observed 2D points before passing to BAE: `points_2d[i] = [cx - xy[0], cy - xy[1]]`. This centers observations around the principal point in BAE's expected sign convention. cx,cy remain untouched in Camera.params (not optimized by BAE).

**C3 — BAE's sign convention.** BAE's project() negates the projection: `-points_proj[..., :2] / z`. This follows BAL dataset convention. COLMAP uses standard pinhole: `x/z, y/z` (no negation). COLMAP's ReprojErrorCostFunctor applies the camera model directly (CamFromImg/ImgFromCam).
  **Mitigation:** Must write a COLMAP-compatible Reproj function in bae_solver.py that matches COLMAP's projection, NOT reuse BAE's default project(). The Reproj forward() must reproduce COLMAP's camera model exactly.

**C4 — AddPointToProblem residuals.** Ceres adds extra residuals for points in config.VariablePoints()/ConstantPoints() whose tracks extend to images NOT in config.Images(). These use constant poses from those external images. In global BA, VariablePoints/ConstantPoints are typically empty (all images in config). In local BA, they may be non-empty (new short-track points).
  **Mitigation:** For the initial implementation, handle only the common case: all observations come from images in config.Images(). If VariablePoints/ConstantPoints is non-empty, add those extra observations with frozen poses (treat external-image poses as constants in the camera_params tensor).

**C5 — Intrinsics writeback with shared cameras.** Multiple images may share one Camera. We duplicate intrinsics per-image. After optimization, the per-image copies may diverge.
  **Mitigation:** Average the optimized intrinsics across all images sharing the same camera_id before writing back. Or: use a separate intrinsics tensor indexed by camera_id (cleaner but requires modifying BAE's Reproj to take two index arrays).

**C6 — min_track_length filtering.** Ceres skips observations where `point3D.track.Length() < options.min_track_length`. BAE extraction must apply the same filter.

**C7 — Ignored points.** config.IsIgnoredPoint() must be checked during extraction. Points in the ignored set are skipped entirely.

**C8 — refine_* flags.** BundleAdjustmentOptions has refine_focal_length, refine_principal_point, refine_extra_params, refine_points3D, refine_rig_from_world. BAE must respect these by marking the corresponding slices of camera_params / points_3d as non-optimizable.

**C9 — constant_rig_from_world_rotation.** When true, only translation is refined. BAE SE3 manifold update modifies all 6 DOF. Need to zero out the rotation component of the update or handle differently.

**C10 — Quaternion normalization.** After BAE optimization, quaternions may not be unit-length. Ceres uses a quaternion manifold to enforce this. BAE uses SE3 Lie group updates which preserve unit quaternions by construction (exponential map). Should be fine but verify.

## Steps

### Phase 1: Dev Environment (Docker)
1. **docker/Dockerfile** — In builder stage: add `python3-dev`, `python3-pip`, `pybind11-dev` to apt-get. pip install PyTorch (CUDA 12), PyPose (bae branch), BAE. Add `-DBAE_ENABLED=ON` to cmake. In runtime stage: add `python3` to apt-get, pip install same PyTorch + PyPose + BAE.
2. **Verify**: Docker image builds successfully. COLMAP compiles inside container. `python3 -c "import torch; import pypose; import bae"` succeeds.

### Phase 2: C++ Skeleton (wiring only, no logic)
3. **bundle_adjustment.h** — Add `BAE` to `BundleAdjustmentBackend` enum. Add `std::shared_ptr<BaeBundleAdjustmentOptions> bae` to `BundleAdjustmentBackendOptions`. Forward-declare `BaeBundleAdjustmentOptions`.
4. **bundle_adjustment.cc** — Add `case BundleAdjustmentBackend::BAE:` in `CreateDefaultBundleAdjuster()` calling `CreateDefaultBaeBundleAdjuster()`. Update `BundleAdjustmentBackendOptions` copy/assign to handle `bae`. Update `Check()` to handle BAE backend.
5. **bundle_adjustment_bae.h** — Declare `BaeBundleAdjustmentOptions` struct (max_num_iterations, device, solver_type). Declare `CreateDefaultBaeBundleAdjuster()` factory.
6. **bundle_adjustment_bae.cc** — Stub `BaeBundleAdjuster` class inheriting `BundleAdjuster`. Stub `Solve()` returning FAILURE. Implement factory.
7. **CMakeLists.txt** — Add bundle_adjustment_bae.h/.cc conditionally on BAE_ENABLED. Add pybind11_embed + Python link libs.
8. **Verify**: builds with BAE_ENABLED=OFF (no change), builds with BAE_ENABLED=ON (stub Solve returns FAILURE).

### Phase 3: Data Extraction (Reconstruction → flat arrays)
9. **bundle_adjustment_bae.cc** — In constructor, extract from Reconstruction + BundleAdjustmentConfig:
   - Validate: all cameras use SimpleRadial or Radial model, all images are ref-in-frame (no rigs). Fail early otherwise.
   - Build image_id → contiguous index map (only config.Images())
   - Build point3D_id → contiguous index map. Collect all point3D_ids observed in config images (skipping ignored points and those below min_track_length).
   - Extract camera_params: (N_imgs, 10) = [tx,ty,tz, qx,qy,qz,qw, f, k1, k2]. Reorder from COLMAP's [qx,qy,qz,qw,tx,ty,tz].
     For SimpleRadial (1 distortion): set k2=0. For Radial: use k1,k2.
   - Extract points_3d: (N_pts, 3)
   - Extract points_2d: (N_obs, 2), camera_indices: (N_obs,), point_indices: (N_obs,)
     Subtract (cx,cy) from observed 2D points (see C2): `points_2d[i] = [cx - xy[0], cy - xy[1]]`.
   - Also handle config.VariablePoints()/ConstantPoints() extra residuals (see C4):
     add observations from external images with frozen poses.
   - Build constant_camera_mask, constant_point_mask for gauge fixing and refine_* flags (see C8).
10. **Verify**: log extracted array sizes, compare with config.NumResiduals().

### Phase 4: Python BAE Solver Call
11. **python/pycolmap/bae_solver.py** — Python module:
   - Custom `ColmapReproj(nn.Module)` — NOT reusing BAE's default Reproj. Must match COLMAP's projection:
     apply SE3 rotation, divide by z (NO negation), apply radial distortion, multiply by f.
     cx,cy already subtracted from 2D observations on C++ side.
   - `solve(camera_params, points_3d, points_2d, camera_indices, point_indices, constant_camera_mask, constant_point_mask, options_dict)` 
   - Masks constant params by splitting into optimized/frozen sets or disabling gradients.
   - Builds ColmapReproj, creates LM optimizer with TrustRegion + PCG, runs loop.
   - Returns optimized camera_params, points_3d as numpy arrays + convergence info.
12. **bundle_adjustment_bae.cc** — In `Solve()`:
    - Check `Py_IsInitialized()`: if false, create `py::scoped_interpreter`; if true, use `py::gil_scoped_acquire` (see C1).
    - Import pycolmap.bae_solver, call solve() with numpy arrays.
    - Parse returned convergence info into BundleAdjustmentSummary.
    - On Python exception: catch, log, return FAILURE.
12.b **Ample Logging**: Proper logging for BAR on both C++ and python (using pycolmap's logging)

### Phase 5: Writeback (flat arrays → Reconstruction)
13. **bundle_adjustment_bae.cc** — After Python returns:
    - Write optimized poses back to frame.rig_from_world (reorder [tx,ty,tz,qx,qy,qz,qw] → [qx,qy,qz,qw,tx,ty,tz]). Skip if refine_rig_from_world is false or frame is in ConstantRigFromWorldPoses.
    - Write optimized intrinsics (f, k1, k2) back to Camera.params at correct indices. If multiple images share a camera_id, average the per-image intrinsics before writing (see C5). Skip if camera is in ConstantCamIntrinsics.
    - Add cx,cy back — cx,cy are not optimized, so the original values remain in Camera.params.
    - Write optimized points_3d back to Point3D.xyz. Skip constants.
    - Normalize quaternions after writeback (see C10).

### Phase 6: CLI + pycolmap exposure
14. **option_manager.cc** — Add `BundleAdjustmentBae.*` options (max_iterations, device). (Done)
15. **src/pycolmap/estimators/bundle_adjustment.cc** — Add `.value("BAE", BundleAdjustmentBackend::BAE)` to enum binding. Expose BaeBundleAdjustmentOptions.
16. **cmake/FindDependencies.cmake** — Add BAE_ENABLED option, find pybind11 embed + Python.

### Phase 7: Testing
17. Add bundle_adjustment_bae_test.cc — test with synthetic reconstruction (similar to existing bundle_adjustment_test.cc pattern).
18. Verify end-to-end: `colmap mapper --BundleAdjustment.backend=BAE`

## Relevant Files
- `src/colmap/estimators/bundle_adjustment.h` — add BAE to enum, add bae options to BundleAdjustmentBackendOptions
- `src/colmap/estimators/bundle_adjustment.cc` — add case BAE in factory switch, update copy/Check
- `src/colmap/estimators/bundle_adjustment_bae.h` — new: BaeBundleAdjustmentOptions + factory declaration
- `src/colmap/estimators/bundle_adjustment_bae.cc` — new: BaeBundleAdjuster implementation
- `src/colmap/estimators/CMakeLists.txt` — add new files + conditional deps
- `src/colmap/controllers/option_manager.cc` — add BAE CLI options
- `src/pycolmap/estimators/bundle_adjustment.cc` — expose BAE enum + options to Python
- `python/pycolmap/bae_solver.py` — new: Python BAE optimization entry point
- `cmake/FindDependencies.cmake` — BAE_ENABLED flag, pybind11 embed
- `docker/Dockerfile` — add Python/PyTorch/PyPose/BAE to builder + runtime stages

## Verification
1. Phase 1: Docker image builds, COLMAP compiles, `python3 -c "import torch; import pypose; import bae"` succeeds
2. `cmake -DBAE_ENABLED=OFF` — builds without BAE, no regressions
3. `cmake -DBAE_ENABLED=ON` — builds with BAE, links pybind11 embed + Python
4. Phase 2: BaeBundleAdjuster::Solve() returns FAILURE (stub)
5. Phase 3: extracted array sizes match config.NumResiduals() / 2
6. Phase 3: validation rejects non-SimpleRadial/Radial cameras and rig images
7. Phase 4: BAE optimizer runs, loss decreases over iterations
8. Phase 4: Py_IsInitialized() check works from both colmap CLI and pycolmap
9. Phase 5: round-trip test — extract → 0-iteration solve → writeback → verify no data corruption
10. Phase 5: intrinsics writeback averages correctly for shared cameras
11. Phase 7: full test with synthetic data, compare BA result quality vs Ceres
12. CLI: `colmap bundle_adjuster --BundleAdjustment.backend=BAE --input_path=... --output_path=...`
13. Verify ColmapReproj matches COLMAP's projection model by comparing residuals with Ceres at iteration 0
