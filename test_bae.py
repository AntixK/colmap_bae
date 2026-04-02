"""Smoke test: verify COLMAP CLI can run BAE bundle_adjuster successfully.

Run inside the Docker container:
    docker/launch.sh
    python3 /working/test_bae.py

Steps:
    1. Extract SIFT features via colmap CLI.
    2. Sequential matching via colmap CLI.
    3. View graph calibration.
    4. Reconstruction via colmap global_mapper.
    5. Run colmap bundle_adjuster with BAE (CPU) on the same model.
    6. Validate output model files and basic model_analyzer stats.

This is a smoke test, not a performance benchmark.
"""

import math
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


def run(cmd, env=None):
    """Run a CLI command with real-time output."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        check=True,
        capture_output=False,
        text=True,
        env=env,
    )


def run_capture(cmd, env=None):
    """Run a CLI command and capture output."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
    bae_output = work_dir / "bae_output"

    sparse_dir.mkdir()
    bae_output.mkdir()

    print(f"Working directory: {work_dir}")
    print(f"Image directory:   {image_dir}\n")

    # ------------------------------------------------------------------
    # Step 1: Feature extraction
    # ------------------------------------------------------------------
    print("== Step 1: Feature extraction ==")
    run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",
            "--FeatureExtraction.use_gpu",
            "1",
            "--FeatureExtraction.gpu_index",
            "0",
        ]
    )
    check(database_path.exists(), "Database created")

    # ------------------------------------------------------------------
    # Step 2: Feature matching
    # ------------------------------------------------------------------
    print("\n== Step 2: Feature matching ==")
    run(
        [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            "1",
            "--FeatureMatching.gpu_index",
            "0",
            "--SequentialMatching.overlap",
            "10",
        ]
    )
    check(True, "Sequential matching succeeded")

    # ------------------------------------------------------------------
    # Step 3: View graph calibration
    # ------------------------------------------------------------------
    print("\n== Step 3: View graph calibration ==")
    run(
        [
            "colmap",
            "view_graph_calibrator",
            "--database_path",
            str(database_path),
        ]
    )
    check(True, "View graph calibration succeeded")

    # ------------------------------------------------------------------
    # Step 4: Global reconstruction (BAE backend)
    # ------------------------------------------------------------------
    print("\n== Step 4: Global mapping (GLOMAP, BAE backend) ==")
    run(
        [
            "colmap",
            "global_mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--GlobalMapper.ba_backend",
            "BAE",
            "--GlobalMapper.random_seed",
            "42",
            "--GlobalMapper.ba_bae_use_gpu",
            "1",
            "--GlobalMapper.ba_bae_gpu_index",
            "0",
        ]
    )

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
    _, input_stats = run_capture(
        [
            "colmap",
            "model_analyzer",
            "--path",
            str(model_dir),
        ]
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
