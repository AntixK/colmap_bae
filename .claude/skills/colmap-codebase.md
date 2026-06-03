# Skill — Navigating the COLMAP codebase

COLMAP is a Structure-from-Motion (SfM) + Multi-View Stereo (MVS) pipeline, C++17 with optional CUDA. One binary (`colmap`) with many subcommands, plus Python bindings (`pycolmap`). Read this skill when you need to locate code, understand module layering, or build/test changes.

## Directory map

| Path | What lives here |
|------|---|
| `src/colmap/util/` | Threading, logging, caching, PLY I/O, CUDA/OpenGL helpers |
| `src/colmap/math/` | Random, polynomials, graph algorithms (cuts, union-find, MST) |
| `src/colmap/geometry/` | `Rigid3d`, `Sim3d`, essential/homography matrices, triangulation, GPS |
| `src/colmap/sensor/` | Camera distortion models, `Bitmap` (image I/O), `Rig`, sensor specs DB |
| `src/colmap/feature/` | SIFT (CPU/GPU), ALIKED (ONNX), LightGlue, FAISS indexing |
| `src/colmap/optim/` | RANSAC, LO-RANSAC, SPRT, samplers, support measurers |
| `src/colmap/scene/` | `Camera`, `Image`, `Frame`, `Point2D/3D`, `Track`, `Reconstruction`, `Database` (SQLite), `CorrespondenceGraph` |
| `src/colmap/estimators/` | Bundle adjustment (Ceres + BAE), absolute/relative pose, two-view geometry, triangulation, alignment |
| `src/colmap/estimators/solvers/` | Minimal solvers: P3P, 5-pt essential, 7/8-pt fundamental, homography (via PoseLib) |
| `src/colmap/estimators/cost_functions/` | Ceres cost functors: reprojection, Sampson, alignment, pose prior |
| `src/colmap/sfm/` | `IncrementalMapper`, `GlobalMapper`, `IncrementalTriangulator`, `ObservationManager` |
| `src/colmap/mvs/` | PatchMatch stereo (CUDA), depth/normal maps, fusion, meshing |
| `src/colmap/image/` | Image undistortion, warping, line detection |
| `src/colmap/retrieval/` | Vocabulary tree, inverted index, vote-and-verify |
| `src/colmap/controllers/` | `AutomaticReconstruction`, `IncrementalPipeline`, `GlobalPipeline`, `HierarchicalPipeline`, `OptionManager` |
| `src/colmap/exe/` | CLI command implementations (`colmap.cc` dispatcher + per-domain `.cc` files) |
| `src/colmap/ui/` | Qt GUI (rarely needed for BAE work) |
| `src/pycolmap/` | `pybind11` C++ bindings |
| `python/pycolmap/` | Python package (`__init__.py`, `bae_solver.py` lives here) |
| `src/thirdparty/` | Bundled VLFeat, SiftGPU, PoissonRecon, LSD; fetched PoseLib, FAISS, ONNX Runtime |
| `cmake/` | Build helpers and dependency discovery (`FindDependencies.cmake`, etc.) |

## Module dependency layers (bottom → top)

Reading from bottom to top tells you what can include what. Lower layers know nothing about higher layers.

```
util  →  math  →  geometry  →  sensor  →  feature  →  optim  →  scene
                                                              ↓
                                                          estimators
                                                              ↓
                                                            sfm  →  image  →  retrieval
                                                              ↓
                                                       controllers  →  exe / ui
```

BAE-relevant chain: `bundle_adjustment_bae.{cc,h}` lives in `estimators/`, called from `sfm/global_mapper.cc`, dispatched from `controllers/option_manager.cc`. The Python solver `python/pycolmap/bae_solver.py` is invoked via embedded `pybind11` from `bundle_adjustment_bae.cc`.

## Key classes (BAE-adjacent first)

| Class / File | Purpose |
|---|---|
| `BundleAdjuster` (`estimators/bundle_adjustment.h`) | Abstract BA interface. Backends: Ceres (`bundle_adjustment_ceres.{cc,h}`) and BAE (`bundle_adjustment_bae.{cc,h}`). `CreateDefaultBundleAdjuster()` dispatches on `options.backend`. |
| `BundleAdjustmentConfig` (`estimators/bundle_adjustment.h`) | What to optimize vs. hold constant; gauge selection (`FixGauge`); variable / constant point sets. |
| `BundleAdjustmentOptions` (`estimators/bundle_adjustment.h`) | Backend choice, refine flags, min track length, etc. |
| `GlobalMapper` (`sfm/global_mapper.h`) | Global SfM (rotation averaging + global positioning + iter_BA + retri-refinement). `RunBundleAdjustment` is in `sfm/global_mapper.cc`. |
| `IncrementalMapper` (`sfm/incremental_mapper.h`) | Core incremental SfM engine. `IterativeGlobalRefinement` is the BA-filter loop called from `global_mapper.cc:IterativeRetriangulateAndRefine`. |
| `IncrementalTriangulator` (`sfm/incremental_triangulator.h`) | Point creation, track merging/completion. The "retri" step. |
| `ObservationManager` (`sfm/observation_manager.h`) | Per-image visibility, `FilterPoints3DWithLargeReprojectionError`, `FilterPoints3DWithSmallTriangulationAngle`. |
| `Reconstruction` (`scene/reconstruction.h`) | Top-level container: cameras, rigs, images, frames, points, tracks. |
| `Camera` (`scene/camera.h`) | Intrinsics (focal, principal point, distortion). |
| `Rig` (`sensor/rig.h`) | Multi-sensor rig with `sensor_from_rig` transforms. |
| `Image` (`scene/image.h`) | Image entry: name, `Point2D` list, `camera_id`, `frame_id`. |
| `Frame` (`scene/frame.h`) | Posed rig instance: `rig_from_world` + sensor data. |
| `Point3D` (`scene/point3d.h`) | Triangulated point: `xyz`, color, error, `Track`. |
| `Track` (`scene/track.h`) | List of `(image_id, point2D_idx)` observations. |
| `Rigid3d` (`geometry/rigid3.h`) | 6-DoF rigid transform (quaternion + translation). |
| `Sim3d` (`geometry/sim3.h`) | 7-DoF similarity transform. |
| `Database` / `DatabaseCache` (`scene/database.h` / `database_cache.h`) | Feature/match SQLite DB + in-memory cache + `CorrespondenceGraph`. |
| `OptionManager` (`controllers/option_manager.h`) | Centralized CLI option parsing. Source for CLI flag discovery. |
| Camera models (`sensor/models.h`) | `SimplePinhole`, `Radial`, `OpenCV`, `Fisheye`, `SIMPLE_RADIAL` (the one our benchmark uses), etc. |

CLI entry: `src/colmap/exe/colmap.cc` (subcommand dispatcher).

## Build

```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
ninja
```

C++ changes in `bundle_adjustment_bae.{cc,h}` or `global_mapper.cc` require a rebuild. Python changes to `bae_solver.py` are loaded at runtime via `COLMAP_BAE_SOLVER_PATH` (info.md §1.3) — **no rebuild needed**.

### Common variants

```bash
# No GUI, no CUDA (minimal)
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DGUI_ENABLED=OFF -DCUDA_ENABLED=OFF

# With C++ tests enabled
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DTESTS_ENABLED=ON
```

### Build pycolmap

```bash
# C++ install first
mkdir build && cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=../install
ninja install

# Then pycolmap (from repo root)
colmap_DIR=./install ./python/incremental_build.sh   # fast
colmap_DIR=./install ./python/build.sh               # clean (slower)
```

If a `.python-version` file is present, use pyenv/uv for Python commands, `pip install`, and pycolmap builds.

## Test

C++ tests from `build/`:

```bash
ctest --output-on-failure                # All C++ tests
ctest -R "util/cache_test"               # Specific test
ctest -E "(feature/sift_test)"           # Exclude GPU tests
```

Python tests from repo root:

```bash
pytest                                   # All Python tests (config in pyproject.toml)
```

- Test files are `*_test.cc` next to the code, registered via `COLMAP_ADD_TEST()` macro
- Framework: GTest/GMock with custom main (`util/gtest_main.cc`)
- Test utilities: `util/testing.h`, `util/eigen_matchers.h`, `geometry/rigid3_matchers.h`, `geometry/sim3_matchers.h`
- CTest names: `module/test_name` (e.g., `estimators/alignment_test`)

**Note for BAE work**: the BAE-specific quality validation happens via `run_benchmark.py`, not ctest. The ctest suite covers correctness of individual estimators, not end-to-end reconstruction quality.

## Code style

| Element | Convention |
|---|---|
| Classes | `PascalCase` |
| Methods / free functions | `PascalCase` (e.g. `FindNextImages()`) |
| Member variables | `snake_case_` (trailing underscore) |
| Local variables | `snake_case` |
| Constants / enums | `kPascalCase` or `UPPER_SNAKE_CASE` |
| Files | `snake_case.h` / `snake_case.cc` / `snake_case_test.cc` |
| Transforms | `target_from_source` (e.g. `cam_from_world`) |
| Coordinates | `x_in_y` (e.g. `point3D_in_world`) |

### Special identifier types (`util/types.h`)

`camera_t`, `image_t`, `image_pair_t`, `frame_t`, `rig_t`, `point2D_t`, `point3D_t`, `sensor_t`, `data_t`, `pose_prior_t`. Use these instead of bare `int` / `size_t` for the relevant identifiers.

### Formatters

```bash
scripts/format/c++.sh       # clang-format
scripts/format/python.sh    # ruff
```

Run these before committing C++ or Python changes.

## External dependencies

### Core (always required)

| Library | Role |
|---|---|
| Eigen3 | Linear algebra |
| Ceres Solver | Nonlinear optimization (the alternative BA backend) |
| Boost | Graph algorithms, CLI options |
| glog | Structured logging |
| SQLite3 | Feature/match database |
| OpenImageIO | Image I/O |
| CHOLMOD | Sparse Cholesky (Ceres dependency) |
| Metis | Graph partitioning |
| PoseLib | Minimal pose solvers |
| FAISS | Fast ANN for descriptor matching |

### Optional

| Library | Role | CMake gate |
|---|---|---|
| CUDA | GPU PatchMatch, SiftGPU, Ceres GPU BA, BAE (required for us) | `CUDA_ENABLED` |
| ONNX Runtime | ALIKED, LightGlue neural features | `ONNX_ENABLED` |
| Qt5/6 | GUI | `GUI_ENABLED` |
| OpenGL/GLEW | 3D visualization, SiftGPU | `OPENGL_ENABLED` |
| CGAL | Delaunay meshing | `CGAL_ENABLED` |

### Bundled (`src/thirdparty/`)

VLFeat (CPU SIFT), SiftGPU (GPU SIFT), PoissonRecon (surface reconstruction), LSD (line detection).

## Useful infra files

| Path | Use |
|---|---|
| `CMakeLists.txt` (root) | Top-level build config |
| `cmake/CMakeHelper.cmake` | `COLMAP_ADD_LIBRARY` / `_EXECUTABLE` / `_TEST` macros |
| `cmake/FindDependencies.cmake` | All dependency discovery |
| `vcpkg.json` | vcpkg manifest (Windows/macOS deps) |
| `pyproject.toml` | Python build config (scikit-build-core, cibuildwheel) |
| `.clang-format` | C++ formatting style |
| `ruff.toml` | Python linting/formatting config |
| `.github/workflows/` | CI: Ubuntu, macOS, Windows, Docker, pycolmap |
| `docker/` | Dockerfile + build/run scripts |
| `benchmark/` | Reconstruction + runtime benchmarks (separate from `run_benchmark.py`) |
