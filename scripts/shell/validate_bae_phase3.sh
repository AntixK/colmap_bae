#!/usr/bin/env bash
set -euo pipefail

# Validate Phase 3 (Task 9) BAE extraction implementation.
#
# Usage:
#   scripts/shell/validate_bae_phase3.sh
#   scripts/shell/validate_bae_phase3.sh --model_path /path/to/model --colmap_bin colmap
#
# Notes:
# - Static source checks always run.
# - Runtime check is optional and runs only when --model_path is provided.
# - Runtime check tolerates solver failure (current Solve() is still a stub),
#   but requires extraction log output.

SOURCE_FILE="src/colmap/estimators/bundle_adjustment_bae.cc"
COLMAP_BIN="colmap"
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --colmap_bin)
      COLMAP_BIN="$2"
      shift 2
      ;;
    --source_file)
      SOURCE_FILE="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# This validation is intended for the Docker dev/runtime environment.
if [[ ! -f /.dockerenv ]]; then
  echo "This script must be run inside the Docker container." >&2
  echo "Hint: start the container with docker/launch.sh and run it there." >&2
  exit 2
fi

pass_count=0
fail_count=0

pass() {
  echo "[PASS] $1"
  pass_count=$((pass_count + 1))
}

fail() {
  echo "[FAIL] $1" >&2
  fail_count=$((fail_count + 1))
}

require_pattern() {
  local pattern="$1"
  local description="$2"
  if command -v rg >/dev/null 2>&1; then
    if rg -n --no-heading -F "$pattern" "$SOURCE_FILE" >/dev/null 2>&1; then
      pass "$description"
    else
      fail "$description"
    fi
  else
    if grep -n -F "$pattern" "$SOURCE_FILE" >/dev/null 2>&1; then
      pass "$description"
    else
      fail "$description"
    fi
  fi
}

has_in_file() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F "$pattern" "$file" >/dev/null 2>&1
  else
    grep -F "$pattern" "$file" >/dev/null 2>&1
  fi
}

last_match_line() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg "$pattern" "$file" | tail -n1 || true
  else
    grep "$pattern" "$file" | tail -n1 || true
  fi
}

echo "[INFO] Tooling: using $(command -v rg >/dev/null 2>&1 && echo rg || echo grep) for text search"

echo "== Static checks: $SOURCE_FILE =="

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "Source file not found: $SOURCE_FILE" >&2
  exit 1
fi

# Phase 3 Task 9 checks
require_pattern "camera.model_id == CameraModelId::kSimpleRadial ||" \
  "Validates supported camera models for config images"
require_pattern "THROW_CHECK(image.IsRefInFrame())" \
  "Validates no-rig constraint for config images"
require_pattern "image_id_to_idx_[image_id] = num_images_++" \
  "Builds contiguous image index map"
require_pattern "point3D_id_to_idx_[point2D.point3D_id] = num_points_++" \
  "Collects point indices from config images"
require_pattern "config_.VariablePoints()" \
  "Handles VariablePoints"
require_pattern "config_.ConstantPoints()" \
  "Handles ConstantPoints"
require_pattern "points_2d_.push_back(cx - point2D.xy.x())" \
  "Applies principal-point centering/sign convention for x"
require_pattern "points_2d_.push_back(cy - point2D.xy.y())" \
  "Applies principal-point centering/sign convention for y"
require_pattern "constant_camera_mask_[idx]" \
  "Builds camera const mask"
require_pattern "constant_point_mask_[idx]" \
  "Builds point const mask"
require_pattern "THROW_CHECK(ext_image.IsRefInFrame())" \
  "Validates no-rig constraint for external images"
require_pattern "BAE extraction:" \
  "Logs extraction summary"

if [[ -n "$MODEL_PATH" ]]; then
  echo "== Runtime check: model=$MODEL_PATH =="

  if ! command -v "$COLMAP_BIN" >/dev/null 2>&1; then
    fail "COLMAP binary '$COLMAP_BIN' not found"
  elif [[ ! -d "$MODEL_PATH" ]]; then
    fail "Model path does not exist: $MODEL_PATH"
  else
    tmp_out_dir="$(mktemp -d)"
    tmp_log="$(mktemp)"
    cleanup() {
      rm -rf "$tmp_out_dir" "$tmp_log"
    }
    trap cleanup EXIT

    set +e
    "$COLMAP_BIN" bundle_adjuster \
      --BundleAdjustment.backend=BAE \
      --input_path "$MODEL_PATH" \
      --output_path "$tmp_out_dir" >"$tmp_log" 2>&1
    exit_code=$?
    set -e

    if has_in_file "BAE extraction:" "$tmp_log"; then
      pass "Runtime produced BAE extraction summary"
    else
      fail "Runtime did not produce BAE extraction summary"
    fi

    # Parse extraction sizes
    extracted_line="$(last_match_line "BAE extraction:" "$tmp_log")"
    if [[ -n "$extracted_line" ]]; then
      read -r ext_images ext_points ext_obs < <(
        sed -n 's/.*BAE extraction: \([0-9]\+\) images, \([0-9]\+\) points, \([0-9]\+\) observations.*/\1 \2 \3/p' <<<"$extracted_line"
      )

      if [[ -n "${ext_images:-}" && -n "${ext_points:-}" && -n "${ext_obs:-}" ]]; then
        if (( ext_images > 0 && ext_points >= 0 && ext_obs >= 0 )); then
          pass "Extracted sizes are parseable and non-negative"
        else
          fail "Extracted sizes are invalid"
        fi
      else
        fail "Could not parse extraction sizes"
      fi
    fi

    # Optional parity check for text models (global BA default assumptions).
    if [[ -f "$MODEL_PATH/points3D.txt" ]]; then
      expected_obs=$(awk '
        BEGIN {obs=0}
        /^#/ {next}
        NF >= 8 {
          # Fields: id x y z r g b err, then pairs (image_id, point2D_idx)
          obs += int((NF - 8) / 2)
        }
        END {print obs}
      ' "$MODEL_PATH/points3D.txt")

      if [[ -n "${ext_obs:-}" ]]; then
        if [[ "$expected_obs" == "$ext_obs" ]]; then
          pass "Observation count matches points3D.txt track count"
        else
          fail "Observation mismatch: extracted=$ext_obs expected=$expected_obs"
        fi
      fi
    else
      echo "[INFO] points3D.txt not found; skipped observation parity check"
    fi

    # Current BAE solver may still return failure; do not hard fail on exit code.
    echo "[INFO] bundle_adjuster exit code: $exit_code"
  fi
else
  echo "[INFO] Runtime check skipped (pass --model_path to enable)."
fi

echo "== Summary =="
echo "Pass: $pass_count"
echo "Fail: $fail_count"

if (( fail_count > 0 )); then
  exit 1
fi

exit 0
