"""End-to-end test: verify BAE bundle adjustment matches Ceres results.

Run inside the Docker container:
    docker/launch.sh
    python3 /working/test_bae.py

Steps:
    1. Extract SIFT features via colmap CLI.
    2. Sequential matching via colmap CLI.
    3. View graph calibration.
    4. Reconstruction via colmap global_mapper (Ceres BA, the default).
    5. Run colmap bundle_adjuster with Ceres on the model  → baseline.
    6. Run colmap bundle_adjuster with BAE (CPU) on the same model → test.
    7. Compare both outputs via colmap model_analyzer.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0


def check(condition, msg):
    global _pass, _fail
    if condition:
        print(f"  [PASS] {msg}")
        _pass += 1
    else:
        print(f"  [FAIL] {msg}")
        _fail += 1


def run(cmd):
    """Run a CLI command with real-time output."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, capture_output=False, text=True)


def run_capture(cmd):
    """Run a CLI command and capture output."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    print(output)
    return result, output


def parse_mean_reproj_error(output):
    """Extract mean reprojection error from model_analyzer output."""
    m = re.search(r"Mean reprojection error:\s+([\d.]+)px", output)
    return float(m.group(1)) if m else None


def parse_num_points(output):
    """Extract number of 3D points from model_analyzer output."""
    m = re.search(r"Points:\s+(\d+)", output)
    return int(m.group(1)) if m else None


def parse_num_observations(output):
    """Extract number of observations from model_analyzer output."""
    m = re.search(r"Observations:\s+(\d+)", output)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    image_dir = Path("data/Ignatius/images")
    if not image_dir.exists():
        print(f"FATAL: Image directory not found: {image_dir}")
        print("  Mount the repo root as /working when launching Docker.")
        sys.exit(1)

    work_dir = Path(tempfile.mkdtemp(prefix="bae_test_"))
    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    ceres_output = work_dir / "ceres_output"
    bae_output = work_dir / "bae_output"

    sparse_dir.mkdir()
    ceres_output.mkdir()
    bae_output.mkdir()

    print(f"Working directory: {work_dir}")
    print(f"Image directory:   {image_dir}\n")

    # ------------------------------------------------------------------
    # Step 1: Feature extraction
    # ------------------------------------------------------------------
    print("== Step 1: Feature extraction ==")
    run([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
    ])
    check(database_path.exists(), "Database created")

    # ------------------------------------------------------------------
    # Step 2: Feature matching
    # ------------------------------------------------------------------
    print("\n== Step 2: Feature matching ==")
    run([
        "colmap", "sequential_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0",
        "--SequentialMatching.overlap", "10",
    ])
    check(True, "Sequential matching succeeded")

    # ------------------------------------------------------------------
    # Step 3: View graph calibration
    # ------------------------------------------------------------------
    print("\n== Step 3: View graph calibration ==")
    run([
        "colmap", "view_graph_calibrator",
        "--database_path", str(database_path),
    ])
    check(True, "View graph calibration succeeded")

    # ------------------------------------------------------------------
    # Step 4: Global reconstruction (Ceres default)
    # ------------------------------------------------------------------
    print("\n== Step 4: Global mapping (GLOMAP, Ceres BA) ==")
    run([
        "colmap", "global_mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ])

    model_dirs = sorted(sparse_dir.glob("*/cameras.bin"))
    check(len(model_dirs) > 0, "At least one reconstruction produced")
    if not model_dirs:
        print("FATAL: cannot continue without a reconstruction")
        shutil.rmtree(work_dir)
        sys.exit(1)

    model_dir = model_dirs[0].parent
    print(f"  Using model: {model_dir}")

    # Analyze input model.
    print("\n-- Input model stats --")
    _, input_stats = run_capture([
        "colmap", "model_analyzer", "--path", str(model_dir),
    ])

    # ------------------------------------------------------------------
    # Step 5: Ceres bundle adjustment (baseline)
    # ------------------------------------------------------------------
    print("\n== Step 5: Ceres bundle adjustment (baseline) ==")
    ceres_result, ceres_log = run_capture([
        "colmap", "bundle_adjuster",
        "--input_path", str(model_dir),
        "--output_path", str(ceres_output),
    ])
    check(ceres_result.returncode == 0, "Ceres BA exited successfully")

    # ------------------------------------------------------------------
    # Step 6: BAE bundle adjustment (CPU, same input)
    # ------------------------------------------------------------------
    print("\n== Step 6: BAE bundle adjustment (CPU) ==")
    bae_result, bae_log = run_capture([
        "colmap", "bundle_adjuster",
        "--BundleAdjustment.backend", "BAE",
        "--BundleAdjustmentBae.use_gpu", "0",
        "--input_path", str(model_dir),
        "--output_path", str(bae_output),
    ])
    check(bae_result.returncode == 0, "BAE BA exited successfully")
    check("BAE extraction:" in bae_log, "BAE extraction ran")
    check("BAE Python error" not in bae_log, "No Python errors during BAE")
    check("BAE bundle adjustment report" in bae_log, "BAE solver completed")

    # ------------------------------------------------------------------
    # Step 7: Compare outputs
    # ------------------------------------------------------------------
    print("\n== Step 7: Compare Ceres vs BAE results ==")

    for backend, outdir in [("Ceres", ceres_output), ("BAE", bae_output)]:
        for fname in ("cameras.bin", "images.bin", "points3D.bin"):
            fpath = outdir / fname
            check(
                fpath.exists() and fpath.stat().st_size > 0,
                f"{backend}: {fname} exists and non-empty",
            )

    print("\n-- Ceres output stats --")
    _, ceres_stats = run_capture([
        "colmap", "model_analyzer", "--path", str(ceres_output),
    ])
    print("\n-- BAE output stats --")
    _, bae_stats = run_capture([
        "colmap", "model_analyzer", "--path", str(bae_output),
    ])

    ceres_err = parse_mean_reproj_error(ceres_stats)
    bae_err = parse_mean_reproj_error(bae_stats)
    ceres_pts = parse_num_points(ceres_stats)
    bae_pts = parse_num_points(bae_stats)

    print(f"\n  Ceres: reproj_err={ceres_err}px, points={ceres_pts}")
    print(f"  BAE:   reproj_err={bae_err}px, points={bae_pts}")

    if ceres_err is not None and bae_err is not None:
        ratio = bae_err / ceres_err if ceres_err > 0 else float("inf")
        print(f"  BAE/Ceres reproj error ratio: {ratio:.3f}")
        check(ratio < 2.0, f"BAE reproj error within 2x of Ceres ({ratio:.3f})")
        check(bae_err < 2.0, f"BAE mean reproj error < 2.0px ({bae_err:.4f}px)")
    else:
        check(False, "Could not parse reprojection errors from model_analyzer")

    if bae_pts is not None and ceres_pts is not None:
        check(bae_pts == ceres_pts,
              f"Same number of points (Ceres={ceres_pts}, BAE={bae_pts})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n== Summary: {_pass} passed, {_fail} failed ==")
    shutil.rmtree(work_dir)

    if _fail > 0:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
