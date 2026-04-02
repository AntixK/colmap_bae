#!/usr/bin/env python3
"""Benchmark: COLMAP global_mapper with BAE backend vs Ceres backend.

Runs both pipelines on the same dataset (Ignatius), then compares:
  - Reprojection error (from model_analyzer)
  - Point cloud divergence (Umeyama-aligned RMSE on matched 3D points)
  - Timing: total pipeline wall-clock time
  - Peak CUDA memory (for BAE runs)

Results are exported to JSON.

Usage (inside Docker):
    cp /working/python/pycolmap/bae_solver.py \
       /colmap/python/pycolmap/bae_solver.py
    python3 /working/benchmark_bae.py [--image_dir data/Ignatius/images]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Binary model readers (adapted from COLMAP / InstantSfM read_write_model.py)
# ---------------------------------------------------------------------------

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODEL_IDS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
}


def _read_next(fid, n, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(n))


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        n = _read_next(f, 8, "Q")[0]
        for _ in range(n):
            props = _read_next(f, 24, "iiQQ")
            cid, mid = props[0], props[1]
            np_ = CAMERA_MODEL_IDS[mid].num_params
            params = np.array(_read_next(f, 8 * np_, "d" * np_))
            cameras[cid] = Camera(
                id=cid, model=CAMERA_MODEL_IDS[mid].model_name,
                width=props[2], height=props[3], params=params)
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        n = _read_next(f, 8, "Q")[0]
        for _ in range(n):
            props = _read_next(f, 64, "idddddddi")
            iid = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]
            name = b""
            c = _read_next(f, 1, "c")[0]
            while c != b"\x00":
                name += c
                c = _read_next(f, 1, "c")[0]
            np2d = _read_next(f, 8, "Q")[0]
            raw = _read_next(f, 24 * np2d, "ddq" * np2d) if np2d else ()
            xys = (np.column_stack([raw[0::3], raw[1::3]])
                   if np2d else np.zeros((0, 2)))
            p3d_ids = (np.array(list(raw[2::3]), dtype=np.int64)
                       if np2d else np.array([], dtype=np.int64))
            images[iid] = Image(
                id=iid, qvec=qvec, tvec=tvec, camera_id=cam_id,
                name=name.decode(), xys=xys, point3D_ids=p3d_ids)
    return images


def read_points3D_binary(path):
    pts = {}
    with open(path, "rb") as f:
        n = _read_next(f, 8, "Q")[0]
        for _ in range(n):
            props = _read_next(f, 43, "QdddBBBd")
            pid = props[0]
            xyz = np.array(props[1:4])
            err = float(props[7])
            tl = _read_next(f, 8, "Q")[0]
            tr = _read_next(f, 8 * tl, "ii" * tl) if tl else ()
            pts[pid] = Point3D(
                id=pid, xyz=xyz,
                rgb=np.array(props[4:7], dtype=np.uint8),
                error=err,
                image_ids=(np.array(tr[0::2], dtype=np.int32)
                           if tl else np.array([], dtype=np.int32)),
                point2D_idxs=(np.array(tr[1::2], dtype=np.int32)
                              if tl else np.array([], dtype=np.int32)))
    return pts


def qvec2rotmat(q):
    return np.array([
        [1-2*q[2]**2-2*q[3]**2, 2*q[1]*q[2]-2*q[0]*q[3],
         2*q[3]*q[1]+2*q[0]*q[2]],
        [2*q[1]*q[2]+2*q[0]*q[3], 1-2*q[1]**2-2*q[3]**2,
         2*q[2]*q[3]-2*q[0]*q[1]],
        [2*q[3]*q[1]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1],
         1-2*q[1]**2-2*q[2]**2],
    ])


# ---------------------------------------------------------------------------
# Reprojection error verification
#
# COLMAP's model_analyzer reports the *mean* reprojection error across all
# observations.  For each observation (u_obs, v_obs) linked to a 3D point P
# viewed from image I with camera C, the reprojection error is:
#
#   e = ||proj(C, T_I, P) - (u_obs, v_obs)||_2
#
# where proj applies the camera model (here SIMPLE_RADIAL).  The reported
# value is  mean(e)  over all observations.  See
# src/colmap/scene/reconstruction.cc  ComputeMeanReprojectionError().
# ---------------------------------------------------------------------------

def compute_reprojection_errors(model_dir: str) -> dict:
    """Independently verify reprojection error from binary model files.

    Returns dict with mean, median, p90, p99 reprojection errors in pixels.
    """
    cameras = read_cameras_binary(os.path.join(model_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(model_dir, "images.bin"))
    pts3d = read_points3D_binary(os.path.join(model_dir, "points3D.bin"))

    if not pts3d:
        return {"mean": None, "median": None, "num_observations": 0}

    errors = []
    for img in images.values():
        cam = cameras[img.camera_id]
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        # Camera intrinsics for SIMPLE_RADIAL: [f, cx, cy, k]
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        k1 = cam.params[3] if len(cam.params) > 3 else 0.0
        k2 = cam.params[4] if len(cam.params) > 4 else 0.0

        for i, pid in enumerate(img.point3D_ids):
            if pid < 0:  # unlinked observation
                continue
            if pid not in pts3d:
                continue
            P = pts3d[pid].xyz
            # World -> camera
            pc = R @ P + t
            if pc[2] <= 0:
                continue
            # Perspective divide
            u = pc[0] / pc[2]
            v = pc[1] / pc[2]
            # Radial distortion
            r2 = u * u + v * v
            dist = 1.0 + k1 * r2 + k2 * r2 * r2
            px = f * dist * u + cx
            py = f * dist * v + cy
            # Observation
            obs = img.xys[i]
            err = np.sqrt((px - obs[0])**2 + (py - obs[1])**2)
            errors.append(err)

    errors = np.array(errors)
    if len(errors) == 0:
        return {"mean": None, "median": None, "num_observations": 0}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p99": float(np.percentile(errors, 99)),
        "max": float(np.max(errors)),
        "num_observations": len(errors),
    }


# ---------------------------------------------------------------------------
# Umeyama alignment (similarity transform: scale + rotation + translation)
# ---------------------------------------------------------------------------

def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """Compute Sim3 (s, R, t) aligning src to dst.

    Returns (scale, R, t) such that dst ~ s * R @ src + t.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = np.sum(src_c ** 2) / n
    cov = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd, capture=False):
    """Run a shell command, print it, return (returncode, output)."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out
    r = subprocess.run(cmd)
    return r.returncode, ""


def parse_model_stats(output: str) -> dict:
    """Extract stats from ``colmap model_analyzer`` output."""
    def _f(pat):
        m = re.search(pat, output)
        return float(m.group(1)) if m else None
    def _i(pat):
        m = re.search(pat, output)
        return int(m.group(1)) if m else None
    return {
        "num_registered_images": _i(r"Registered images:\s+(\d+)"),
        "num_points": _i(r"Points:\s+(\d+)"),
        "num_observations": _i(r"Observations:\s+(\d+)"),
        "mean_track_length": _f(r"Mean track length:\s+([\d.]+)"),
        "mean_observations_per_image": _f(
            r"Mean observations per image:\s+([\d.]+)"),
        "mean_reprojection_error_px": _f(
            r"Mean reprojection error:\s+([\d.]+)px"),
    }


def analyze_model(model_dir: str) -> dict:
    rc, out = run_cmd(
        ["colmap", "model_analyzer", "--path", model_dir], capture=True)
    return parse_model_stats(out) if rc == 0 else {}


import threading
import GPUtil


class GpuMemoryMonitor:
    """Polls GPU memory via GPUtil in a background thread to capture peak."""

    def __init__(self, gpu_id: int = 0, interval: float = 0.25):
        self.gpu_id = gpu_id
        self.interval = interval
        self.peak_mb: float = 0.0
        self.baseline_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> float | None:
        gpus = GPUtil.getGPUs()
        if gpus and self.gpu_id < len(gpus):
            return gpus[self.gpu_id].memoryUsed
        return None

    def _poll(self):
        while not self._stop.is_set():
            mb = self._sample()
            if mb is not None and mb > self.peak_mb:
                self.peak_mb = mb
            self._stop.wait(self.interval)

    def start(self):
        self.baseline_mb = self._sample() or 0.0
        self.peak_mb = self.baseline_mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return {
            "baseline_mb": round(self.baseline_mb, 1),
            "peak_mb": round(self.peak_mb, 1),
            "delta_mb": round(self.peak_mb - self.baseline_mb, 1),
        }


# ---------------------------------------------------------------------------
# Pipeline: shared front-end
# ---------------------------------------------------------------------------

def run_frontend(image_dir: Path, work_dir: Path) -> Path:
    """Feature extraction + matching + view-graph calibration."""
    db = work_dir / "database.db"
    rc, _ = run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
    ])
    assert rc == 0, "feature_extractor failed"

    rc, _ = run_cmd([
        "colmap", "sequential_matcher",
        "--database_path", str(db),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0",
        "--SequentialMatching.overlap", "10",
    ])
    assert rc == 0, "sequential_matcher failed"

    rc, _ = run_cmd([
        "colmap", "view_graph_calibrator",
        "--database_path", str(db),
    ])
    assert rc == 0, "view_graph_calibrator failed"
    return db


# ---------------------------------------------------------------------------
# Pipeline: global_mapper
# ---------------------------------------------------------------------------

def run_global_mapper(
    db: Path, image_dir: Path, output_dir: Path, backend: str,
) -> tuple[float, str | None]:
    """Run global_mapper. Returns (wall_seconds, model_dir | None)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap", "global_mapper",
        "--database_path", str(db),
        "--image_path", str(image_dir),
        "--output_path", str(output_dir),
        "--GlobalMapper.ba_backend", backend,
        "--GlobalMapper.random_seed", "42",
        # Ensure GPU is used for global positioning (Ceres).
        "--GlobalMapper.gp_use_gpu", "1",
        "--GlobalMapper.gp_gpu_index", "0",
    ]
    if backend == "CERES":
        cmd += [
            "--GlobalMapper.ba_ceres_use_gpu", "1",
            "--GlobalMapper.ba_ceres_gpu_index", "0",
        ]
    elif backend == "BAE":
        cmd += [
            "--GlobalMapper.ba_bae_use_gpu", "1",
            "--GlobalMapper.ba_bae_gpu_index", "0",
        ]

    t0 = time.perf_counter()
    rc, output = run_cmd(cmd, capture=True)
    elapsed = time.perf_counter() - t0

    if rc != 0:
        print(f"  WARNING: global_mapper ({backend}) exited with {rc}")
        print(output[-2000:] if len(output) > 2000 else output)
        return elapsed, None, {}

    models = sorted(output_dir.glob("*/cameras.bin"))
    if not models:
        print(f"  WARNING: no model produced by {backend}")
        return elapsed, None, {}

    # Extract sub-timings from COLMAP log output.
    timings = {}
    for label, pattern in [
        ("rotation_averaging_s",
         r"Rotation averaging done in ([\d.]+) seconds"),
        ("track_establishment_s",
         r"Track establishment done in ([\d.]+) seconds"),
        ("global_positioning_s",
         r"Global positioning done in ([\d.]+) seconds"),
        ("iterative_ba_s",
         r"Iterative bundle adjustment done in ([\d.]+) seconds"),
        ("retriangulation_s",
         r"Iterative retriangulation and refinement done in ([\d.]+) seconds"),
        ("reconstruction_s",
         r"Reconstruction done in ([\d.]+) seconds"),
    ]:
        m = re.search(pattern, output)
        if m:
            timings[label] = float(m.group(1))

    return elapsed, str(models[0].parent), timings


# ---------------------------------------------------------------------------
# Point cloud comparison
# ---------------------------------------------------------------------------

def compare_point_clouds(ref_dir: str, test_dir: str,
                         pose_alignment: tuple | None = None) -> dict:
    """Align test point cloud to ref, then compare via nearest-neighbor.

    Uses the camera-pose Umeyama alignment (if provided) to transform the test
    points into the ref coordinate frame, then finds NN correspondences.

    Args:
        ref_dir: path to Ceres model directory
        test_dir: path to BAE model directory
        pose_alignment: (scale, R, t) from camera pose alignment, or None to
            compute alignment from point clouds directly via random subsample.
    """
    from scipy.spatial import cKDTree

    ref_pts = read_points3D_binary(os.path.join(ref_dir, "points3D.bin"))
    test_pts = read_points3D_binary(os.path.join(test_dir, "points3D.bin"))

    ref_xyz = np.array([p.xyz for p in ref_pts.values()])
    test_xyz = np.array([p.xyz for p in test_pts.values()])

    result = {
        "num_ref_points": len(ref_xyz),
        "num_test_points": len(test_xyz),
    }

    if len(ref_xyz) < 10 or len(test_xyz) < 10:
        result["note"] = "too few points"
        return result

    # --- Alignment ---
    if pose_alignment is not None:
        s, R, t = pose_alignment
    else:
        # Fallback: subsample and use Umeyama on NN pairs (iterative)
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(test_xyz), min(5000, len(test_xyz)),
                             replace=False)
        sub_test = test_xyz[sub_idx]
        tree_ref = cKDTree(ref_xyz)
        _, nn_idx = tree_ref.query(sub_test)
        s, R, t = umeyama_alignment(sub_test, ref_xyz[nn_idx])

    aligned_test = s * (R @ test_xyz.T).T + t
    result["alignment_scale"] = float(s)

    # --- NN matching: test -> ref ---
    tree_ref = cKDTree(ref_xyz)
    dists_fwd, _ = tree_ref.query(aligned_test)

    # --- NN matching: ref -> test (symmetric) ---
    tree_test = cKDTree(aligned_test)
    dists_bwd, _ = tree_test.query(ref_xyz)

    # --- Mutual NN: keep only pairs where both agree ---
    _, nn_fwd_idx = tree_ref.query(aligned_test)
    _, nn_bwd_idx = tree_test.query(ref_xyz)
    mutual_mask = np.zeros(len(aligned_test), dtype=bool)
    for i, j in enumerate(nn_fwd_idx):
        if nn_bwd_idx[j] == i:
            mutual_mask[i] = True
    mutual_dists = dists_fwd[mutual_mask]

    result.update(
        # Forward: for each BAE point, distance to closest Ceres point
        fwd_mean=float(np.mean(dists_fwd)),
        fwd_median=float(np.median(dists_fwd)),
        fwd_p90=float(np.percentile(dists_fwd, 90)),
        fwd_p99=float(np.percentile(dists_fwd, 99)),
        # Backward: for each Ceres point, distance to closest BAE point
        bwd_mean=float(np.mean(dists_bwd)),
        bwd_median=float(np.median(dists_bwd)),
        bwd_p90=float(np.percentile(dists_bwd, 90)),
        bwd_p99=float(np.percentile(dists_bwd, 99)),
        # Mutual NN (most reliable correspondences)
        mutual_nn_count=int(mutual_mask.sum()),
        mutual_mean=float(np.mean(mutual_dists)) if len(mutual_dists) else None,
        mutual_median=float(np.median(mutual_dists)) if len(mutual_dists) else None,
        mutual_p90=float(np.percentile(mutual_dists, 90)) if len(mutual_dists) else None,
        mutual_p99=float(np.percentile(mutual_dists, 99)) if len(mutual_dists) else None,
    )
    return result


# ---------------------------------------------------------------------------
# Camera pose comparison
# ---------------------------------------------------------------------------

def compare_camera_poses(ref_dir: str, test_dir: str) -> dict:
    """Compare camera poses between two reconstructions.

    Aligns test poses to ref via Umeyama on camera centers,
    then reports rotation and translation errors per image.
    """
    ref_images = read_images_binary(os.path.join(ref_dir, "images.bin"))
    test_images = read_images_binary(os.path.join(test_dir, "images.bin"))

    # Match by image name
    ref_by_name = {img.name: img for img in ref_images.values()}
    test_by_name = {img.name: img for img in test_images.values()}
    common_names = sorted(set(ref_by_name) & set(test_by_name))

    result = {
        "num_ref_images": len(ref_images),
        "num_test_images": len(test_images),
        "num_common_images": len(common_names),
    }

    if len(common_names) < 3:
        result["note"] = "too few common images"
        return result

    # Extract camera centers: C = -R^T @ t
    ref_centers = []
    test_centers = []
    ref_rotations = []
    test_rotations = []
    for name in common_names:
        ri = ref_by_name[name]
        ti = test_by_name[name]
        R_ref = qvec2rotmat(ri.qvec)
        R_test = qvec2rotmat(ti.qvec)
        ref_centers.append(-R_ref.T @ ri.tvec)
        test_centers.append(-R_test.T @ ti.tvec)
        ref_rotations.append(R_ref)
        test_rotations.append(R_test)

    ref_centers = np.array(ref_centers)
    test_centers = np.array(test_centers)

    # Align test camera centers to ref via Umeyama
    s, R_align, t_align = umeyama_alignment(test_centers, ref_centers)
    aligned_centers = s * (R_align @ test_centers.T).T + t_align

    # Translation errors (after alignment)
    trans_errors = np.linalg.norm(aligned_centers - ref_centers, axis=1)

    # Rotation errors: angle between R_ref and R_align @ R_test
    rot_errors_deg = []
    for R_ref, R_test in zip(ref_rotations, test_rotations):
        # After alignment, the test rotation becomes R_align @ R_test
        R_diff = R_ref @ (R_align @ R_test).T
        # Angle from rotation matrix: cos(theta) = (trace(R) - 1) / 2
        cos_angle = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
        rot_errors_deg.append(np.degrees(np.arccos(cos_angle)))

    rot_errors_deg = np.array(rot_errors_deg)

    result.update(
        alignment_scale=float(s),
        translation_error_mean=float(np.mean(trans_errors)),
        translation_error_median=float(np.median(trans_errors)),
        translation_error_p90=float(np.percentile(trans_errors, 90)),
        translation_error_max=float(np.max(trans_errors)),
        rotation_error_mean_deg=float(np.mean(rot_errors_deg)),
        rotation_error_median_deg=float(np.median(rot_errors_deg)),
        rotation_error_p90_deg=float(np.percentile(rot_errors_deg, 90)),
        rotation_error_max_deg=float(np.max(rot_errors_deg)),
        # Store alignment transform for reuse (not serialized to JSON)
        _R=R_align,
        _t=t_align,
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BAE vs Ceres BA in COLMAP global_mapper")
    parser.add_argument("--image_dir", default="data/Ignatius/images")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--keep_workdir", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"FATAL: image directory not found: {image_dir}")
        sys.exit(1)

    work_dir = Path(tempfile.mkdtemp(prefix="bae_bench_"))
    print(f"Working directory: {work_dir}")
    print(f"Image directory:   {image_dir}\n")

    results: dict = {
        "image_dir": str(image_dir),
        "work_dir": str(work_dir),
        "backends": {},
    }

    # ---- Shared front-end ----
    print("=" * 60)
    print("Phase 1: Feature extraction + matching (shared)")
    print("=" * 60)
    t0 = time.perf_counter()
    db = run_frontend(image_dir, work_dir)
    frontend_time = time.perf_counter() - t0
    results["frontend_time_s"] = round(frontend_time, 3)
    print(f"\nFront-end completed in {frontend_time:.1f}s\n")

    # ---- Run each backend ----
    for backend in ["CERES", "BAE"]:
        print("=" * 60)
        print(f"Phase 2: global_mapper  [{backend}]")
        print("=" * 60)

        # Copy DB so both runs start from identical state.
        bdir = work_dir / backend.lower()
        bdir.mkdir()
        shutil.copy2(db, bdir / "database.db")
        sparse_dir = bdir / "sparse"

        gpu_mon = GpuMemoryMonitor(gpu_id=0, interval=0.25)
        gpu_mon.start()
        pipeline_time, model_dir, timings = run_global_mapper(
            bdir / "database.db", image_dir, sparse_dir, backend)
        gpu_stats = gpu_mon.stop()

        entry: dict = {
            "pipeline_time_s": round(pipeline_time, 3),
            "sub_timings": timings,
            "gpu_peak_mb": gpu_stats["peak_mb"],
            "gpu_baseline_mb": gpu_stats["baseline_mb"],
            "gpu_delta_mb": gpu_stats["delta_mb"],
        }

        if model_dir:
            entry["model_dir"] = model_dir

            # COLMAP's reported stats
            stats = analyze_model(model_dir)
            entry["colmap_stats"] = stats

            # Independent reprojection error verification
            verified = compute_reprojection_errors(model_dir)
            entry["verified_reproj"] = verified

            print(f"\n  {backend} COLMAP stats:")
            for k, v in stats.items():
                print(f"    {k}: {v}")
            print(f"  {backend} verified reproj (mean): "
                  f"{verified.get('mean', '?')}")
        else:
            entry["model_dir"] = None
            entry["colmap_stats"] = None
            entry["verified_reproj"] = None

        results["backends"][backend] = entry
        print(f"\n  {backend} total: {pipeline_time:.1f}s\n")

    ceres_dir = (results["backends"].get("CERES") or {}).get("model_dir")
    bae_dir = (results["backends"].get("BAE") or {}).get("model_dir")

    # ---- Camera pose comparison (run first to get alignment) ----
    print("=" * 60)
    print("Phase 3: Camera pose comparison")
    print("=" * 60)

    pose_alignment = None
    if ceres_dir and bae_dir:
        pose_comp = compare_camera_poses(ceres_dir, bae_dir)
        results["camera_pose_comparison"] = pose_comp
        print(f"\n  Reference (Ceres): {pose_comp['num_ref_images']} images")
        print(f"  Test (BAE):        {pose_comp['num_test_images']} images")
        print(f"  Common images:     {pose_comp['num_common_images']}")
        if pose_comp.get("rotation_error_mean_deg") is not None:
            pose_alignment = (
                pose_comp["alignment_scale"],
                pose_comp["_R"],
                pose_comp["_t"],
            )
            print(f"  Alignment scale:   {pose_comp['alignment_scale']:.6f}")
            print(f"  Rotation error (deg):")
            print(f"    mean:   {pose_comp['rotation_error_mean_deg']:.4f}")
            print(f"    median: {pose_comp['rotation_error_median_deg']:.4f}")
            print(f"    p90:    {pose_comp['rotation_error_p90_deg']:.4f}")
            print(f"    max:    {pose_comp['rotation_error_max_deg']:.4f}")
            print(f"  Translation error (after alignment):")
            print(f"    mean:   {pose_comp['translation_error_mean']:.4f}")
            print(f"    median: {pose_comp['translation_error_median']:.4f}")
            print(f"    p90:    {pose_comp['translation_error_p90']:.4f}")
            print(f"    max:    {pose_comp['translation_error_max']:.4f}")
    else:
        results["camera_pose_comparison"] = {
            "note": "one or both backends failed"}
        print("\n  Skipped: one or both backends failed.")

    # ---- Point cloud comparison (NN-based) ----
    print("\n" + "=" * 60)
    print("Phase 4: Point cloud comparison (NN-based, pose-aligned)")
    print("=" * 60)

    if ceres_dir and bae_dir:
        comparison = compare_point_clouds(ceres_dir, bae_dir,
                                          pose_alignment=pose_alignment)
        results["point_cloud_comparison"] = comparison
        print(f"\n  Reference (Ceres): {comparison['num_ref_points']} pts")
        print(f"  Test (BAE):        {comparison['num_test_points']} pts")
        print(f"  Alignment scale:   {comparison.get('alignment_scale', 'N/A')}")
        if comparison.get("mutual_mean") is not None:
            print(f"  Mutual NN pairs:   {comparison['mutual_nn_count']}")
            print(f"  Mutual NN distance:")
            print(f"    mean:   {comparison['mutual_mean']:.6f}")
            print(f"    median: {comparison['mutual_median']:.6f}")
            print(f"    p90:    {comparison['mutual_p90']:.6f}")
            print(f"    p99:    {comparison['mutual_p99']:.6f}")
            print(f"  Forward (BAE->Ceres) distance:")
            print(f"    mean:   {comparison['fwd_mean']:.6f}")
            print(f"    median: {comparison['fwd_median']:.6f}")
            print(f"    p90:    {comparison['fwd_p90']:.6f}")
            print(f"  Backward (Ceres->BAE) distance:")
            print(f"    mean:   {comparison['bwd_mean']:.6f}")
            print(f"    median: {comparison['bwd_median']:.6f}")
            print(f"    p90:    {comparison['bwd_p90']:.6f}")
        else:
            print(f"  NOTE: {comparison.get('note', 'comparison failed')}")
    else:
        results["point_cloud_comparison"] = {
            "note": "one or both backends failed to produce a model"}
        print("\n  Skipped: one or both backends failed.")

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    fmt = "  {:<25s} {:>15s} {:>15s}"
    print(fmt.format("", "CERES", "BAE"))
    print(fmt.format("-" * 25, "-" * 15, "-" * 15))

    def _get(backend, *keys):
        v = results["backends"].get(backend, {})
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k)
            else:
                return "N/A"
        return v if v is not None else "N/A"

    rows = [
        ("Pipeline time (s)",
         _get("CERES", "pipeline_time_s"), _get("BAE", "pipeline_time_s")),
        ("Points",
         _get("CERES", "colmap_stats", "num_points"),
         _get("BAE", "colmap_stats", "num_points")),
        ("Observations",
         _get("CERES", "colmap_stats", "num_observations"),
         _get("BAE", "colmap_stats", "num_observations")),
        ("Reproj error (COLMAP)",
         _get("CERES", "colmap_stats", "mean_reprojection_error_px"),
         _get("BAE", "colmap_stats", "mean_reprojection_error_px")),
        ("Reproj error (verified)",
         _get("CERES", "verified_reproj", "mean"),
         _get("BAE", "verified_reproj", "mean")),
        ("GPU peak (MB)",
         _get("CERES", "gpu_peak_mb"),
         _get("BAE", "gpu_peak_mb")),
        ("GPU delta (MB)",
         _get("CERES", "gpu_delta_mb"),
         _get("BAE", "gpu_delta_mb")),
    ]
    for label, c, b in rows:
        cs = f"{c}" if not isinstance(c, float) else f"{c:.4f}"
        bs = f"{b}" if not isinstance(b, float) else f"{b:.4f}"
        print(fmt.format(label, cs, bs))

    comp = results.get("point_cloud_comparison", {})
    if comp.get("mutual_mean") is not None:
        print(f"\n  Point cloud divergence (BAE vs Ceres, NN-based):")
        print(f"    Mutual NN pairs:    {comp['mutual_nn_count']}")
        print(f"    Mutual NN median:   {comp['mutual_median']:.6f}")
        print(f"    Mutual NN p90:      {comp['mutual_p90']:.6f}")
        print(f"    Fwd (BAE->Ceres) median: {comp['fwd_median']:.6f}")
        print(f"    Bwd (Ceres->BAE) median: {comp['bwd_median']:.6f}")

    pose = results.get("camera_pose_comparison", {})
    if pose.get("rotation_error_mean_deg") is not None:
        print(f"\n  Camera pose divergence (BAE vs Ceres):")
        print(f"    Common images:          {pose['num_common_images']}")
        print(f"    Rotation err mean (deg): {pose['rotation_error_mean_deg']:.4f}")
        print(f"    Rotation err p90 (deg):  {pose['rotation_error_p90_deg']:.4f}")
        print(f"    Translation err mean:    {pose['translation_error_mean']:.4f}")
        print(f"    Translation err p90:     {pose['translation_error_p90']:.4f}")

    # ---- Export ----
    # Remove non-serializable internal fields before JSON export
    if "_R" in results.get("camera_pose_comparison", {}):
        del results["camera_pose_comparison"]["_R"]
        del results["camera_pose_comparison"]["_t"]

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    if not args.keep_workdir:
        shutil.rmtree(work_dir)
        print(f"Cleaned up {work_dir}")
    else:
        print(f"Working directory kept at {work_dir}")


if __name__ == "__main__":
    main()
