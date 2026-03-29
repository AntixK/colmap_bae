"""End-to-end test for COLMAP BAE bundle adjustment.

Run inside the Docker container:
    docker/launch.sh
    python3 /working/test_bae.py

Steps:
    1. Extract SIFT features via colmap CLI.
    2. Sequential matching via colmap CLI.
    3. View graph calibration (refine focal lengths, filter bad pairs).
    4. Reconstruction via colmap global_mapper CLI (GLOMAP + BAE BA).
    5. Standalone BAE bundle adjustment via colmap CLI.
    6. Validate that the BAE solver pipeline ran correctly.
"""

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


def run(cmd, timeout=600, cwd=None):
    """Run a CLI command with real-time output."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True, capture_output=False, text=True, cwd=cwd, timeout=timeout)  # noqa
    return result


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
    bae_output_dir = work_dir / "bae_output"

    sparse_dir.mkdir()
    bae_output_dir.mkdir()

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
    # Step 2: Exhaustive matching
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
    # Step 4: Global reconstruction (GLOMAP)
    # ------------------------------------------------------------------
    print("\n== Step 4: Global mapping (GLOMAP + BAE) ==")
    run([
        "colmap", "global_mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        "--GlobalMapper.ba_backend", "BAE",
    ])

    # Find the reconstruction directory (usually sparse/0/).
    model_dirs = sorted(sparse_dir.glob("*/cameras.bin"))
    check(len(model_dirs) > 0, "At least one reconstruction produced")
    if not model_dirs:
        print("FATAL: cannot continue without a reconstruction")
        shutil.rmtree(work_dir)
        sys.exit(1)

    model_dir = model_dirs[0].parent
    print(f"  Using model: {model_dir}")

    # ------------------------------------------------------------------
    # Step 5: BAE bundle adjustment
    # ------------------------------------------------------------------
    print("\n== Step 5: BAE bundle adjustment ==")
    # Capture output for BAE-specific validation (bundle_adjuster returns 0
    # even when BAE fails internally, so we must inspect the logs).
    bae_result = subprocess.run(
        [
            "colmap", "bundle_adjuster",
            "--BundleAdjustment.backend", "BAE",
            "--input_path", str(model_dir),
            "--output_path", str(bae_output_dir),
        ],
        capture_output=True, text=True, timeout=600,
    )
    bae_output = (bae_result.stdout or "") + (bae_result.stderr or "")
    print(bae_output)

    # ------------------------------------------------------------------
    # Step 6: Validation
    # ------------------------------------------------------------------
    print("\n== Step 6: Validation ==")

    check(bae_result.returncode == 0, f"Exit code is 0 (got {bae_result.returncode})")
    check("BAE extraction:" in bae_output, "BAE extraction ran")
    check("BAE Python error" not in bae_output, "No Python errors during BAE solve")
    check(
        "BAE bundle adjustment report" in bae_output,
        "BAE solver completed and printed report",
    )

    # Output reconstruction written (BAE results written back to model).
    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        fpath = bae_output_dir / fname
        check(
            fpath.exists() and fpath.stat().st_size > 0,
            f"Output {fname} exists and non-empty",
        )

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
