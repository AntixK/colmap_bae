"""End-to-end benchmark: BAE vs Ceres bundle adjustment backends.

Runs the full COLMAP pipeline twice (once per backend) on every dataset
discovered under data/.  A dataset is any subdirectory data/<name>/ that
contains an images/ folder.

Per-stage wall time, peak CPU RSS, peak GPU memory (via GPUtil), and
sub-stage timings (parsed from global_mapper output) are captured.
After both backends finish, the two reconstructions are compared via
RANSAC similarity alignment on shared cameras plus nearest-neighbour
matching on the 3D points (Ceres treated as reference).

Output layout:
    bench/<dataset>/<dataset>_ceres/    # full Ceres reconstruction + run.log
    bench/<dataset>/<dataset>_bae/      # full BAE reconstruction + run.log
    bench/<dataset>/benchmark_results.json
    bench/summary.json                  # concise aggregate across datasets

Run inside the Docker container:
    docker/launch.sh
    python3 /working/run_benchmark.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("WARNING: GPUtil not installed; GPU memory will not be tracked.",
          flush=True)

try:
    import pycolmap
    HAS_PYCOLMAP = True
except Exception as _e:  # pycolmap can fail to import inside this env
    HAS_PYCOLMAP = False
    print(f"WARNING: pycolmap unavailable ({_e!r}); reconstruction comparison "
          "will be skipped.", flush=True)

from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data")
BENCH_ROOT = Path("bench")
SUMMARY_FILE = BENCH_ROOT / "summary.json"

GPU_INDEX_BAE = "0"          # BAE solver GPU (its own setting)
GPU_INDEX_CERES = "0"        # Ceres BA GPU (apples-to-apples vs BAE)
SHARED_GPU_INDICES = "0,1,2,3"  # for feature_extractor + matcher
GPU_POLL_INTERVAL_S = 0.5

# Treat Ceres as the reference; BAE is the candidate being measured.
REFERENCE_BACKEND = "ceres"
CANDIDATE_BACKEND = "bae"

BACKENDS = ["ceres", "bae"]


# ---------------------------------------------------------------------------
# Subprocess runner: wall time, peak CPU RSS, peak GPU memory, log capture.
# ---------------------------------------------------------------------------

def _spawn_gpu_monitor(stop_event, peak_box):
    """Background thread polling GPUtil for peak total VRAM across all GPUs."""

    def loop():
        while not stop_event.is_set():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    total = sum(float(g.memoryUsed) for g in gpus)
                    peak_box[0] = max(peak_box[0], total)
            except Exception:
                pass
            stop_event.wait(GPU_POLL_INTERVAL_S)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _spawn_rss_monitor(pid, stop_event, peak_box):
    """Background thread polling /proc/<pid>/status for peak VmRSS (MB).

    Picks up the parent process plus all its threads (they share the
    address space).  Forked children are NOT included — colmap is
    primarily threaded so this is close enough for benchmark purposes.
    """
    proc_status = Path(f"/proc/{pid}/status")
    rss_re = re.compile(r"^VmRSS:\s+(\d+)\s+kB", re.MULTILINE)

    def loop():
        while not stop_event.is_set():
            try:
                if proc_status.exists():
                    text = proc_status.read_text()
                    m = rss_re.search(text)
                    if m:
                        rss_mb = int(m.group(1)) / 1024.0
                        peak_box[0] = max(peak_box[0], rss_mb)
            except Exception:
                pass
            stop_event.wait(GPU_POLL_INTERVAL_S)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _save_json_atomic(path, data):
    """Write JSON atomically (write to .tmp, fsync, rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"WARNING: failed to load {path}: {e!r}", flush=True)
        return None


def run_stage(stage_name, cmd, log_fh, env=None, fail_on_error=True):
    """Run a CLI command, tee output to terminal + log_fh, capture metrics.

    Polls /proc/<pid>/status for peak VmRSS (CPU memory) and GPUtil for
    peak GPU memory across all GPUs.  Returns a dict with wall_s,
    peak_cpu_mb, peak_gpu_mb, exit_code, stdout.

    Raises RuntimeError on non-zero exit when fail_on_error=True (default),
    matching the `check=True` semantics in run_bae.py.  This makes the
    benchmark fail fast so a resume can re-attempt the failed stage.
    """
    print(f"\n  $ {' '.join(cmd)}", flush=True)
    log_fh.write(f"\n$ {' '.join(cmd)}\n")
    log_fh.flush()

    gpu_peak = [0.0]
    rss_peak = [0.0]
    stop_event = threading.Event()
    gpu_monitor_thread = None
    rss_monitor_thread = None
    if HAS_GPUTIL:
        gpu_monitor_thread = _spawn_gpu_monitor(stop_event, gpu_peak)

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    # Spawn RSS poller now that we have a pid.
    rss_monitor_thread = _spawn_rss_monitor(proc.pid, stop_event, rss_peak)

    output_chunks = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_fh.write(line)
        output_chunks.append(line)
    proc.wait()
    log_fh.flush()

    stop_event.set()
    if gpu_monitor_thread is not None:
        gpu_monitor_thread.join(timeout=2.0)
    if rss_monitor_thread is not None:
        rss_monitor_thread.join(timeout=2.0)

    wall_s = time.perf_counter() - t0
    output = "".join(output_chunks)

    peak_cpu_mb = rss_peak[0] if rss_peak[0] > 0 else None

    print(
        f"  [{stage_name}] wall={wall_s:.1f}s  "
        f"cpu_peak={peak_cpu_mb:.0f}MB  "
        f"gpu_peak={gpu_peak[0]:.0f}MB  "
        f"exit={proc.returncode}",
        flush=True,
    )
    if fail_on_error and proc.returncode != 0:
        raise RuntimeError(
            f"Stage '{stage_name}' failed with exit code "
            f"{proc.returncode}.  See log for details.")
    return {
        "wall_s": wall_s,
        "peak_cpu_mb": peak_cpu_mb,
        "peak_gpu_mb": gpu_peak[0] if HAS_GPUTIL else None,
        "exit_code": proc.returncode,
        "stdout": output,
    }


# ---------------------------------------------------------------------------
# Log parsers
# ---------------------------------------------------------------------------

def parse_global_mapper_substages(output):
    """Extract per-substage seconds from global_mapper stdout."""
    patterns = {
        "rotation_averaging_s":
            r"Rotation averaging done in ([\d.]+) seconds",
        "track_establishment_s":
            r"Track establishment done in ([\d.]+) seconds",
        "global_positioning_s":
            r"Global positioning done in ([\d.]+) seconds",
        "iterative_ba_s":
            r"Iterative bundle adjustment done in ([\d.]+) seconds",
        "retri_refinement_s":
            r"Iterative retriangulation and refinement done in "
            r"([\d.]+) seconds",
        "reconstruction_total_s":
            r"Reconstruction done in ([\d.]+) seconds",
    }
    return {
        k: float(m.group(1)) if (m := re.search(p, output)) else None
        for k, p in patterns.items()
    }


def parse_bae_stats(output):
    """Aggregate per-call BAE stats from `[BAE]` log lines.

    Returns mean/median across all BAE solve invocations during the run.
    """
    pre_filter_ratios = []
    post_p50_norms = []
    iters_per_call = []
    initial_costs = []
    final_costs = []
    kernel_deltas = []

    for m in re.finditer(
            r"\[BAE\] pre-filter: kept (\d+) / (\d+)", output):
        kept, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            pre_filter_ratios.append(kept / total)

    for m in re.finditer(
            r"\[BAE\] post err \[norm\] p50=([\d.eE+-]+)", output):
        post_p50_norms.append(float(m.group(1)))

    for m in re.finditer(
            r"\[BAE\] finished: (\d+) iters, cost ([\d.eE+-]+) -> "
            r"([\d.eE+-]+)", output):
        iters_per_call.append(int(m.group(1)))
        initial_costs.append(float(m.group(2)))
        final_costs.append(float(m.group(3)))

    for m in re.finditer(
            r"\[BAE\] kernel: Huber\(delta=([\d.]+) px\)", output):
        kernel_deltas.append(float(m.group(1)))

    if not iters_per_call:
        return None

    def _stats(vals):
        if not vals:
            return None
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "n_calls": len(iters_per_call),
        "iters_per_call": _stats(iters_per_call),
        "pre_filter_keep_ratio": _stats(pre_filter_ratios),
        "post_p50_norm": _stats(post_p50_norms),
        "kernel_delta_px": _stats(kernel_deltas),
        "initial_cost": _stats(initial_costs),
        "final_cost": _stats(final_costs),
    }


def parse_model_analyzer(output):
    def _pick(pat, cast):
        m = re.search(pat, output)
        return cast(m.group(1)) if m else None

    return {
        "images": _pick(r"Images: (\d+)", int),
        "registered_images": _pick(r"Registered images: (\d+)", int),
        "points": _pick(r"Points: (\d+)", int),
        "observations": _pick(r"Observations: (\d+)", int),
        "mean_track_length": _pick(r"Mean track length: ([\d.]+)", float),
        "mean_obs_per_image": _pick(
            r"Mean observations per image: ([\d.]+)", float),
        "mean_reproj_px": _pick(
            r"Mean reprojection error: ([\d.]+)px", float),
    }


# ---------------------------------------------------------------------------
# Reconstruction comparison (Kabsch + NN on point clouds)
# ---------------------------------------------------------------------------

def _load_reconstruction(model_dir):
    """Return (cam_centers, cam_rotations, points3D) keyed by image name.

    cam_centers[name] : (3,) world-frame camera position
    cam_rotations[name] : (3,3) world->cam rotation (R from cam_from_world)
    points3D : (N, 3) array of 3D point positions
    """
    if not HAS_PYCOLMAP:
        raise RuntimeError("pycolmap not available")
    rec = pycolmap.Reconstruction(str(model_dir))
    cam_centers = {}
    cam_rotations = {}
    for image in rec.images.values():
        if not image.has_pose:
            continue
        # In pycolmap, cam_from_world is a METHOD (not a property) and
        # projection_center() returns the world-frame camera center
        # directly (= -R^T t).  Verified against the reference impl.
        cfw = image.cam_from_world()
        Rwc = np.asarray(cfw.rotation.matrix(), dtype=np.float64)  # (3,3)
        center = np.asarray(image.projection_center(), dtype=np.float64)
        cam_centers[image.name] = center
        cam_rotations[image.name] = Rwc
    pts = np.array(
        [np.asarray(p.xyz) for p in rec.points3D.values()],
        dtype=np.float64,
    ) if len(rec.points3D) > 0 else np.zeros((0, 3))
    return cam_centers, cam_rotations, pts


def _kabsch_similarity(P, Q):
    """Best similarity transform mapping P -> Q (Umeyama 1991).

    Q ≈ scale * (P @ R.T) + t.  Returns (scale, R, t).  Sensitive to
    outliers — use _ransac_similarity for noisy data.
    """
    assert P.shape == Q.shape
    n = P.shape[0]
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    P0 = P - pc
    Q0 = Q - qc
    H = P0.T @ Q0 / n  # cross-cov
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    var_p = (P0 ** 2).sum() / n
    scale = float(np.trace(D @ np.diag(S)) / var_p) if var_p > 0 else 1.0
    t = qc - scale * R @ pc
    return scale, R, t


def _ransac_similarity(P, Q, n_iters=1000, inlier_rel_threshold=0.05,
                       rng_seed=42):
    """Robust similarity transform P -> Q via RANSAC + Umeyama refit.

    Sample 3 correspondences, fit Umeyama, count inliers under a residual
    threshold proportional to a robust scene radius.  Repeat n_iters times,
    keep best-inlier-count model, refit Umeyama on all inliers.

    Returns dict with scale, R, t, n_inliers, scene_radius, threshold.
    Falls back to plain Umeyama on the full set if RANSAC fails.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(P)
    # Robust scene radius from Q (the reference): median distance from
    # the geometric median (componentwise).  Used to scale the inlier
    # threshold so it's invariant to absolute scene size.
    qc_robust = np.median(Q, axis=0)
    scene_radius = float(np.median(np.linalg.norm(Q - qc_robust, axis=1)))
    threshold = inlier_rel_threshold * max(scene_radius, 1e-9)

    if n < 3:
        scale, R, t = _kabsch_similarity(P, Q)
        return {"scale": scale, "R": R, "t": t,
                "inliers": np.ones(n, dtype=bool),
                "n_inliers": n, "scene_radius": scene_radius,
                "threshold": threshold, "ransac_success": False}

    best_n = 0
    best_inliers = None
    for _ in range(n_iters):
        idx = rng.choice(n, size=3, replace=False)
        scale, R, t = _kabsch_similarity(P[idx], Q[idx])
        if not np.isfinite(scale) or scale <= 0:
            continue
        # Sanity reject crazy-scale models (e.g. 3 nearly collinear pts).
        if scale > 1e6 or scale < 1e-6:
            continue
        P_pred = scale * (P @ R.T) + t
        resid = np.linalg.norm(P_pred - Q, axis=1)
        n_in = int((resid < threshold).sum())
        if n_in > best_n:
            best_n = n_in
            best_inliers = resid < threshold

    if best_inliers is None or best_n < 3:
        # RANSAC could not find a model; fall back to all-points Umeyama.
        scale, R, t = _kabsch_similarity(P, Q)
        return {"scale": scale, "R": R, "t": t,
                "inliers": np.ones(n, dtype=bool),
                "n_inliers": n, "scene_radius": scene_radius,
                "threshold": threshold, "ransac_success": False}

    # Refit Umeyama on all inliers for a precise model.
    scale, R, t = _kabsch_similarity(P[best_inliers], Q[best_inliers])
    return {"scale": scale, "R": R, "t": t,
            "inliers": best_inliers,
            "n_inliers": best_n, "scene_radius": scene_radius,
            "threshold": threshold, "ransac_success": True}


def _camera_distribution_diag(P, label):
    """Print and return percentiles of camera-distance-from-median.

    Used to detect outlier cameras that would corrupt plain Umeyama:
    a heavy tail (p99 >> p50) means a few cameras are at extreme
    positions and the alignment must be done robustly.
    """
    pc = np.median(P, axis=0)
    dists = np.linalg.norm(P - pc, axis=1)
    pcts = np.percentile(dists, [10, 50, 90, 99])
    info = {
        "n": int(len(P)),
        "p10": float(pcts[0]), "p50": float(pcts[1]),
        "p90": float(pcts[2]), "p99": float(pcts[3]),
        "max": float(dists.max()),
        "p99_over_p50": (float(pcts[3] / pcts[1])
                         if pcts[1] > 0 else float("inf")),
    }
    print(
        f"  [{label}] cam dist from median: "
        f"p10={info['p10']:.3f} p50={info['p50']:.3f} "
        f"p90={info['p90']:.3f} p99={info['p99']:.3f} "
        f"max={info['max']:.3f}  (p99/p50={info['p99_over_p50']:.1f})",
        flush=True,
    )
    return info


def _percentiles(arr, name=""):
    if len(arr) == 0:
        return None
    pcts = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(pcts[0]), "p25": float(pcts[1]),
        "p50": float(pcts[2]), "p75": float(pcts[3]),
        "p90": float(pcts[4]), "p95": float(pcts[5]),
        "p99": float(pcts[6]),
        "max": float(np.max(arr)),
    }


def _rotation_angle_deg(R1, R2):
    """Angle between two rotation matrices, in degrees, vectorised."""
    cos_t = (np.einsum('ijk,ijk->i', R1, R2) - 1.0) / 2.0
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def compare_reconstructions(bae_dir, ceres_dir):
    """Compute Kabsch alignment (cameras) + NN distances (points).

    Treats Ceres as ground truth: returns BAE deviation from Ceres after
    aligning BAE → Ceres world frame via shared cameras.
    """
    print("\n== Loading reconstructions for comparison ==", flush=True)
    bae_centers, bae_rots, bae_pts = _load_reconstruction(bae_dir)
    ceres_centers, ceres_rots, ceres_pts = _load_reconstruction(ceres_dir)
    print(f"  BAE   : {len(bae_centers)} images, {len(bae_pts)} points",
          flush=True)
    print(f"  CERES : {len(ceres_centers)} images, {len(ceres_pts)} points",
          flush=True)

    shared_names = sorted(set(bae_centers) & set(ceres_centers))
    print(f"  Shared images: {len(shared_names)}", flush=True)
    if len(shared_names) < 3:
        return {"error": "insufficient shared images for alignment"}

    P = np.stack([bae_centers[n] for n in shared_names])     # BAE
    Q = np.stack([ceres_centers[n] for n in shared_names])   # Ceres (ref)

    # Diagnose camera distributions before alignment.  A heavy tail
    # (p99/p50 >> 10) means the previous run was almost certainly broken
    # by an outlier camera.
    bae_cam_dist = _camera_distribution_diag(P, "BAE")
    ceres_cam_dist = _camera_distribution_diag(Q, "CERES")

    # Robust similarity alignment: RANSAC + Umeyama refit on inliers.
    align = _ransac_similarity(P, Q, n_iters=1000,
                               inlier_rel_threshold=0.05)
    scale, R, t = align["scale"], align["R"], align["t"]
    inliers = align["inliers"]
    print(
        f"  RANSAC: scale={scale:.6f}  ||t||={np.linalg.norm(t):.4f}  "
        f"inliers={align['n_inliers']}/{len(P)}  "
        f"scene_radius={align['scene_radius']:.3f}  "
        f"threshold={align['threshold']:.3f}  "
        f"success={align['ransac_success']}",
        flush=True,
    )

    # Camera translation residuals.  Report inlier-only AND full-set
    # percentiles so the reader can see how concentrated the bulk is
    # vs how bad the outliers are.
    P_aligned = scale * (P @ R.T) + t
    cam_trans_resid_all = np.linalg.norm(P_aligned - Q, axis=1)
    cam_trans_resid_in = cam_trans_resid_all[inliers]

    # Camera rotation residuals.  In the Ceres frame, the BAE pose's
    # world->cam rotation is R_bae @ R.T (rotation of world axes only;
    # scale doesn't matter for rotation).
    R_bae = np.stack([bae_rots[n] for n in shared_names])     # (N,3,3)
    R_ce  = np.stack([ceres_rots[n] for n in shared_names])
    R_bae_aligned = np.einsum('nij,jk->nik', R_bae, R.T)
    cam_rot_deg_all = _rotation_angle_deg(R_bae_aligned, R_ce)
    cam_rot_deg_in = cam_rot_deg_all[inliers]

    # Point cloud: apply the same transform, NN match.
    point_resid = None
    if len(bae_pts) > 0 and len(ceres_pts) > 0:
        bae_pts_aligned = scale * (bae_pts @ R.T) + t
        tree = cKDTree(ceres_pts)
        dists, _ = tree.query(bae_pts_aligned, k=1)
        point_resid = dists

    # Match-rate thresholds (relative to scene radius — invariant to
    # absolute SfM scale).
    if point_resid is not None and len(point_resid) > 0:
        scene_radius = align["scene_radius"]
        rel = point_resid / max(scene_radius, 1e-9)
        match_rate_buckets = {
            "lt_0.001_scene": float(np.mean(rel < 1e-3)),
            "lt_0.005_scene": float(np.mean(rel < 5e-3)),
            "lt_0.01_scene":  float(np.mean(rel < 1e-2)),
            "lt_0.05_scene":  float(np.mean(rel < 5e-2)),
        }
    else:
        match_rate_buckets = None

    # Identify outlier camera names so the user can inspect them in
    # the binary models if needed.
    outlier_idx = np.where(~inliers)[0]
    outlier_names = [shared_names[i] for i in outlier_idx[:50]]  # cap

    return {
        "shared_images": len(shared_names),
        "ransac_align": {
            "scale": float(scale),
            "translation": t.tolist(),
            "translation_norm": float(np.linalg.norm(t)),
            "rotation_matrix": R.tolist(),
            "n_inliers": int(align["n_inliers"]),
            "n_total": int(len(P)),
            "inlier_ratio": float(align["n_inliers"] / max(len(P), 1)),
            "scene_radius": float(align["scene_radius"]),
            "inlier_threshold": float(align["threshold"]),
            "ransac_success": bool(align["ransac_success"]),
        },
        "camera_distribution": {
            "bae": bae_cam_dist,
            "ceres": ceres_cam_dist,
        },
        "camera_translation_residual_inliers": _percentiles(
            cam_trans_resid_in),
        "camera_translation_residual_all": _percentiles(
            cam_trans_resid_all),
        "camera_rotation_deg_inliers": _percentiles(cam_rot_deg_in),
        "camera_rotation_deg_all": _percentiles(cam_rot_deg_all),
        "outlier_camera_names_first50": outlier_names,
        "point_nn_distance": (
            _percentiles(point_resid)
            if point_resid is not None else None),
        "point_match_rate": match_rate_buckets,
        "n_bae_points": int(len(bae_pts)),
        "n_ceres_points": int(len(ceres_pts)),
    }


# ---------------------------------------------------------------------------
# Pipeline driver (per backend)
# ---------------------------------------------------------------------------

def run_pipeline(backend, work_dir, image_dir, on_progress=None):
    """Run the full COLMAP pipeline for one backend, with checkpoint resume.

    Each stage's metrics are persisted to work_dir/stages.json after it
    completes.  On re-invocation, any stage already present in stages.json
    is skipped and its prior metrics are reused.  This means a crashed run
    can be resumed by simply re-executing the script — completed stages
    will not re-run.

    Granularity: per-stage.  Within a stage, restarting re-runs the full
    subprocess.  Stale outputs from a partial previous run (e.g., a
    half-written sparse/ from a crashed global_mapper) are wiped before
    re-attempting that stage so the rerun is clean.

    on_progress: optional callback invoked as on_progress(checkpoint_dict)
    after every stage completes; used by the top-level driver to write
    the master results.json so partial progress survives a kill -9.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "run.log"
    ckpt_path = work_dir / "stages.json"

    # Force COLMAP to load host-mounted bae_solver.py (only matters for BAE).
    host_solver = (Path(__file__).resolve().parent
                   / "python" / "pycolmap" / "bae_solver.py")
    env = os.environ.copy()
    if host_solver.exists():
        env["COLMAP_BAE_SOLVER_PATH"] = str(host_solver)

    bae_trust_region_env = {}
    if backend == "bae":
        for key in (
            "COLMAP_BAE_FIXED_ROT_RADIUS",
            "COLMAP_BAE_FIXED_ROT_MAX_RADIUS",
            "COLMAP_BAE_FIXED_ROT_TR_UP",
            "COLMAP_BAE_FIXED_ROT_TR_DOWN",
            "COLMAP_BAE_FULL_BA_RADIUS",
            "COLMAP_BAE_FULL_BA_MIN_RADIUS",
            "COLMAP_BAE_FULL_BA_MAX_RADIUS",
            "COLMAP_BAE_FULL_BA_TR_UP",
            "COLMAP_BAE_FULL_BA_TR_DOWN",
            "COLMAP_BAE_FULL_BA_OVERRIDE_FIRST_ONLY",
            "COLMAP_BAE_LM_REJECT_CAP",
        ):
            if key in env:
                bae_trust_region_env[key] = env[key]

    print(f"\n{'='*72}\n== Backend: {backend.upper()}  (work_dir={work_dir}) "
          f"==\n{'='*72}", flush=True)

    # Resume state.
    ckpt = _load_json(ckpt_path) or {}
    stages = ckpt.get("stages", {})
    model_dir_str = ckpt.get("model_dir")
    model_stats = ckpt.get("model_stats")

    if stages:
        print(f"  resuming from checkpoint: {sorted(stages.keys())}",
              flush=True)

    def _checkpoint(extras=None):
        snapshot = {
            "backend": backend,
            "work_dir": str(work_dir),
            "log_file": str(log_path),
            "stages": stages,
            "model_dir": model_dir_str,
            "model_stats": model_stats,
        }
        if bae_trust_region_env:
            snapshot["bae_trust_region_env"] = bae_trust_region_env
        if extras:
            snapshot.update(extras)
        _save_json_atomic(ckpt_path, snapshot)
        if on_progress is not None:
            on_progress(snapshot)

    # Append to run.log on resume; write fresh on first run.
    log_mode = "a" if stages else "w"
    with log_path.open(log_mode, buffering=1) as log_fh:
        log_fh.write(
            f"\n# benchmark backend={backend} "
            f"@ {datetime.now().isoformat(timespec='seconds')}"
            f" {'(resume)' if stages else '(fresh)'}\n")
        if bae_trust_region_env:
            log_fh.write(
                "# bae_trust_region_env "
                + json.dumps(bae_trust_region_env, sort_keys=True)
                + "\n"
            )

        def _maybe_run(stage_key, banner, cmd, post=None):
            """Run stage if not already done; checkpoint after success."""
            if stage_key in stages:
                print(f"  [{stage_key}] SKIP (resumed from checkpoint)",
                      flush=True)
                return stages[stage_key]
            print(banner, flush=True)
            result = run_stage(stage_key, cmd, log_fh, env=env)
            if post is not None:
                post(result)
            # Drop bulky stdout from the on-disk checkpoint; full output
            # is already in run.log.
            cached = {k: v for k, v in result.items() if k != "stdout"}
            stages[stage_key] = cached
            _checkpoint()
            return result

        # 1. Feature extraction
        _maybe_run(
            "feature_extraction",
            f"\n-- [{backend}] Step 1: Feature extraction --",
            [
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(image_dir),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "SIMPLE_RADIAL",
                "--FeatureExtraction.use_gpu", "1",
                "--FeatureExtraction.gpu_index", SHARED_GPU_INDICES,
            ],
        )

        # 2. Sequential matching
        _maybe_run(
            "matching",
            f"\n-- [{backend}] Step 2: Sequential matching --",
            [
                "colmap", "sequential_matcher",
                "--database_path", str(database_path),
                "--FeatureMatching.use_gpu", "1",
                "--FeatureMatching.gpu_index", SHARED_GPU_INDICES,
                "--SequentialMatching.overlap", "10",
            ],
        )

        # 3. View graph calibration
        _maybe_run(
            "view_graph_calibration",
            f"\n-- [{backend}] Step 3: View graph calibration --",
            [
                "colmap", "view_graph_calibrator",
                "--database_path", str(database_path),
            ],
        )

        # 4. Global mapping.  If global_mapper isn't checkpointed but a
        # stale sparse/ exists from a crashed previous run, wipe it so
        # the rerun starts clean.
        if "global_mapper" not in stages:
            stale = list(sparse_dir.glob("*/cameras.bin"))
            if stale:
                print(f"  wiping stale {sparse_dir} before global_mapper",
                      flush=True)
                shutil.rmtree(sparse_dir)
                sparse_dir.mkdir(parents=True, exist_ok=True)
        gm_cmd = [
            "colmap", "global_mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
            "--GlobalMapper.ba_backend", backend.upper(),
            "--GlobalMapper.random_seed", "42",
        ]
        if backend == "bae":
            gm_cmd += [
                "--GlobalMapper.ba_bae_use_gpu", "1",
                "--GlobalMapper.ba_bae_gpu_index", GPU_INDEX_BAE,
            ]
        elif backend == "ceres":
            # Enable Ceres CUDA linear solvers (CUDA / CUDA_SPARSE) so the
            # comparison is apples-to-apples GPU.  Ceres falls back to CPU
            # for sub-problems below min_num_images_gpu_solver (default 50).
            gm_cmd += [
                "--GlobalMapper.ba_ceres_use_gpu", "1",
                "--GlobalMapper.ba_ceres_gpu_index", GPU_INDEX_CERES,
            ]

        def _post_global_mapper(result):
            gm_out = result["stdout"]
            result["substages"] = parse_global_mapper_substages(gm_out)
            if backend == "bae":
                result["bae_stats"] = parse_bae_stats(gm_out)

        _maybe_run(
            "global_mapper",
            f"\n-- [{backend}] Step 4: Global mapping ({backend}) --",
            gm_cmd,
            post=_post_global_mapper,
        )

        # 5. Pick produced model and analyze it.
        if model_dir_str is None:
            model_dirs = sorted(sparse_dir.glob("*/cameras.bin"))
            if not model_dirs:
                raise RuntimeError(
                    f"No reconstruction produced for {backend} at "
                    f"{sparse_dir}; cannot proceed.")
            model_dir_str = str(model_dirs[0].parent)
            _checkpoint()
        model_dir = Path(model_dir_str)
        print(f"  Using model: {model_dir}", flush=True)

        ma_result = _maybe_run(
            "model_analyzer",
            f"\n-- [{backend}] Step 5: model_analyzer --",
            ["colmap", "model_analyzer", "--path", str(model_dir)],
        )
        if model_stats is None:
            # Fresh run: ma_result has stdout.  Resume: stdout was
            # stripped before persisting, so re-run analyzer cheaply
            # (it's a few seconds) just to recover the stats.
            ma_out = ma_result.get("stdout")
            if not ma_out:
                print("  re-running model_analyzer to recover stats",
                      flush=True)
                fresh = run_stage(
                    "model_analyzer_recover",
                    ["colmap", "model_analyzer",
                     "--path", str(model_dir)],
                    log_fh, env=env)
                ma_out = fresh["stdout"]
            model_stats = parse_model_analyzer(ma_out)
            print(f"  model: images={model_stats.get('images')}  "
                  f"points={model_stats.get('points')}  "
                  f"obs={model_stats.get('observations')}  "
                  f"mean_reproj={model_stats.get('mean_reproj_px')}",
                  flush=True)
            _checkpoint()

        # 6. Export sparse point cloud as PLY (alongside the .bin model).
        ply_path = model_dir / "points3D.ply"
        _maybe_run(
            "model_to_ply",
            f"\n-- [{backend}] Step 6: PLY export ({ply_path}) --",
            [
                "colmap", "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(ply_path),
                "--output_type", "PLY",
            ],
        )

    total_pipeline_s = sum(
        v["wall_s"] for v in stages.values() if v.get("wall_s") is not None)
    final = {
        "backend": backend,
        "work_dir": str(work_dir),
        "log_file": str(log_path),
        "model_dir": model_dir_str,
        "stages": stages,
        "model_stats": model_stats,
        "total_pipeline_s": total_pipeline_s,
    }
    _save_json_atomic(ckpt_path, final)
    return final


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

# Bumped any time the comparison logic changes — old cached results auto-
# invalidate.  v1: plain Kabsch.  v2: RANSAC similarity.
COMPARISON_VERSION = 2


def run_for_dataset(dataset_name):
    """Run the full benchmark for a single dataset.

    Layout:
        bench/<dataset>/<dataset>_ceres/   work_dir for ceres backend
        bench/<dataset>/<dataset>_bae/     work_dir for bae backend
        bench/<dataset>/benchmark_results.json

    Returns the final results dict (for inclusion in the top-level summary).
    """
    image_dir = DATA_ROOT / dataset_name / "images"
    if not image_dir.is_dir():
        print(f"FATAL: image directory not found: {image_dir}", flush=True)
        return {"dataset": dataset_name, "error": "no images dir"}

    bench_dir = BENCH_ROOT / dataset_name
    bench_dir.mkdir(parents=True, exist_ok=True)
    results_file = bench_dir / "benchmark_results.json"

    print(f"\n{'#'*72}\n# DATASET: {dataset_name}\n# images: {image_dir}\n"
          f"# bench:  {bench_dir}\n{'#'*72}", flush=True)

    # Resume from existing per-dataset results if present; otherwise fresh.
    results = _load_json(results_file) or {}
    results.setdefault("dataset", dataset_name)
    results.setdefault("image_dir", str(image_dir))
    results.setdefault("started_at",
                       datetime.now().isoformat(timespec="seconds"))
    results.setdefault("n_runs_per_backend", 1)
    results["reference_backend"] = REFERENCE_BACKEND
    results["candidate_backend"] = CANDIDATE_BACKEND
    results.setdefault("backends", {})
    results["last_seen_at"] = datetime.now().isoformat(timespec="seconds")
    _save_json_atomic(results_file, results)

    def _persist_progress(backend, snapshot):
        """on_progress callback from run_pipeline: update master json."""
        results["backends"][backend] = snapshot
        results["last_seen_at"] = datetime.now().isoformat(timespec="seconds")
        _save_json_atomic(results_file, results)

    for backend in BACKENDS:
        work_dir = bench_dir / f"{dataset_name}_{backend}"
        callback = lambda snap, _be=backend: _persist_progress(_be, snap)
        results["backends"][backend] = run_pipeline(
            backend, work_dir, image_dir, on_progress=callback)
        _save_json_atomic(results_file, results)

    # ------------------------------------------------------------------
    # Reconstruction comparison.  Skip only if (a) result is non-error,
    # (b) backend fingerprints match what was in effect last time, AND
    # (c) the comparison logic version matches.
    # ------------------------------------------------------------------
    def _fingerprint(backend_result):
        ms = (backend_result or {}).get("model_stats") or {}
        return [ms.get("points"), ms.get("registered_images"),
                ms.get("observations"), ms.get("mean_reproj_px")]

    current_fingerprint = {
        "bae": _fingerprint(results["backends"].get("bae")),
        "ceres": _fingerprint(results["backends"].get("ceres")),
    }
    have_comparison = (
        isinstance(results.get("comparison"), dict)
        and "error" not in results["comparison"]
        and "shared_images" in results["comparison"]
        and results.get("comparison_fingerprint") == current_fingerprint
        and results.get("comparison_version") == COMPARISON_VERSION)
    if have_comparison:
        print("\n== Comparison already in results.json; skipping ==",
              flush=True)
    else:
        bae_dir = results["backends"]["bae"].get("model_dir")
        ce_dir = results["backends"]["ceres"].get("model_dir")
        if bae_dir and ce_dir and HAS_PYCOLMAP:
            try:
                results["comparison"] = compare_reconstructions(
                    Path(bae_dir), Path(ce_dir))
                results["comparison_fingerprint"] = current_fingerprint
                results["comparison_version"] = COMPARISON_VERSION
            except Exception as e:
                print(f"  comparison failed: {e!r}", flush=True)
                results["comparison"] = {"error": repr(e)}
        else:
            results["comparison"] = {
                "error": "missing model_dir or pycolmap unavailable"}
        _save_json_atomic(results_file, results)

    # Quality metric: point retention (BAE points / Ceres points).
    # First-class metric — promoted to top of results.json so it's visible
    # without diving into nested "comparison" fields.  This is the most
    # discriminating BAE-vs-Ceres quality signal across the benchmark
    # suite (4-dataset run showed 1.00 on ignatius, 0.95 mihama, 0.46
    # bridge, 0.47 soil) — much more sensitive than mean_reproj.
    bae_pts = ((results["backends"].get("bae") or {})
               .get("model_stats") or {}).get("points")
    ce_pts = ((results["backends"].get("ceres") or {})
              .get("model_stats") or {}).get("points")
    bae_obs = ((results["backends"].get("bae") or {})
               .get("model_stats") or {}).get("observations")
    ce_obs = ((results["backends"].get("ceres") or {})
              .get("model_stats") or {}).get("observations")
    quality = {
        "bae_points": bae_pts,
        "ceres_points": ce_pts,
        "bae_observations": bae_obs,
        "ceres_observations": ce_obs,
        "point_retention_ratio": (
            float(bae_pts) / float(ce_pts) if bae_pts and ce_pts else None),
        "observation_retention_ratio": (
            float(bae_obs) / float(ce_obs) if bae_obs and ce_obs else None),
    }
    results["quality"] = quality

    results["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _save_json_atomic(results_file, results)
    print(f"\n== [{dataset_name}] Results written to {results_file} ==",
          flush=True)

    # Per-dataset console summary.
    print(f"\n== [{dataset_name}] SUMMARY ==", flush=True)
    for backend in BACKENDS:
        r = results["backends"][backend]
        gm = r["stages"]["global_mapper"]
        print(
            f"  {backend.upper():5s}  "
            f"total={r['total_pipeline_s']:7.1f}s  "
            f"global_mapper={gm['wall_s']:7.1f}s  "
            f"cpu_peak={gm['peak_cpu_mb']:.0f}MB  "
            f"gpu_peak={gm.get('peak_gpu_mb', 0) or 0:.0f}MB  "
            f"points={r['model_stats'].get('points')}  "
            f"mean_reproj={r['model_stats'].get('mean_reproj_px')}",
            flush=True)

    # Quality summary line — point/obs retention is the headline metric.
    pr = quality.get("point_retention_ratio")
    or_ = quality.get("observation_retention_ratio")
    if pr is not None and or_ is not None:
        print(
            f"  RETENTION  point_retention={pr*100:.1f}% "
            f"({bae_pts}/{ce_pts})  "
            f"obs_retention={or_*100:.1f}% ({bae_obs}/{ce_obs})",
            flush=True,
        )

    cmp = results.get("comparison", {})
    if "error" not in cmp:
        ra = cmp.get("ransac_align", {}) or {}
        ct = cmp.get("camera_translation_residual_inliers") or {}
        cr = cmp.get("camera_rotation_deg_inliers") or {}
        pp = cmp.get("point_nn_distance") or {}
        print(
            f"  cmp shared_imgs={cmp.get('shared_images')}  "
            f"inliers={ra.get('n_inliers')}/{ra.get('n_total')}  "
            f"scale={ra.get('scale'):.4f}  "
            f"scene_radius={ra.get('scene_radius'):.3f}",
            flush=True)
        if ct:
            print(
                f"  cmp [inliers] cam_trans p50={ct.get('p50'):.4f} "
                f"p90={ct.get('p90'):.4f}  "
                f"cam_rot p50={cr.get('p50'):.4f}deg "
                f"p90={cr.get('p90'):.4f}deg",
                flush=True)
        if pp:
            print(
                f"  cmp pt_nn p50={pp.get('p50'):.4f} "
                f"p90={pp.get('p90'):.4f} p99={pp.get('p99'):.4f}",
                flush=True)
    return results


# ---------------------------------------------------------------------------
# Top-level: discover datasets, run all, write aggregate summary.
# ---------------------------------------------------------------------------

def discover_datasets():
    """Return dataset names: any subdir of data/ that has an images/ child."""
    if not DATA_ROOT.is_dir():
        return []
    datasets = sorted(
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and (d / "images").is_dir()
    )
    selected = os.environ.get("COLMAP_BENCH_DATASETS", "").strip()
    if not selected:
        return datasets
    requested = [ds.strip() for ds in selected.split(",") if ds.strip()]
    requested_set = set(requested)
    filtered = [ds for ds in datasets if ds in requested_set]
    missing = [ds for ds in requested if ds not in datasets]
    print(
        f"\nDataset filter COLMAP_BENCH_DATASETS={requested} -> {filtered}",
        flush=True,
    )
    if missing:
        print(f"WARNING: requested datasets not found: {missing}", flush=True)
    return filtered


def _summarize_dataset(results):
    """Concise per-dataset summary for the top-level aggregate.

    Strips bulky fields (full ransac rotation matrix, percentile dicts)
    and keeps only the headline numbers.
    """
    if not isinstance(results, dict) or "error" in results:
        return {"dataset": (results or {}).get("dataset"),
                "error": (results or {}).get("error", "unknown")}

    out = {
        "dataset": results.get("dataset"),
        "image_dir": results.get("image_dir"),
        "finished_at": results.get("finished_at"),
    }
    for be in BACKENDS:
        be_data = results.get("backends", {}).get(be) or {}
        gm = (be_data.get("stages") or {}).get("global_mapper") or {}
        ms = be_data.get("model_stats") or {}
        out[be] = {
            "total_s": be_data.get("total_pipeline_s"),
            "global_mapper_s": gm.get("wall_s"),
            "substages": gm.get("substages"),
            "peak_cpu_mb": gm.get("peak_cpu_mb"),
            "peak_gpu_mb": gm.get("peak_gpu_mb"),
            "points": ms.get("points"),
            "registered_images": ms.get("registered_images"),
            "observations": ms.get("observations"),
            "mean_reproj_px": ms.get("mean_reproj_px"),
        }
    cmp = results.get("comparison") or {}
    if "error" not in cmp:
        ra = cmp.get("ransac_align") or {}
        ct = cmp.get("camera_translation_residual_inliers") or {}
        cr = cmp.get("camera_rotation_deg_inliers") or {}
        pn = cmp.get("point_nn_distance") or {}
        out["comparison"] = {
            "shared_images": cmp.get("shared_images"),
            "inlier_ratio": ra.get("inlier_ratio"),
            "scale": ra.get("scale"),
            "scene_radius": ra.get("scene_radius"),
            "cam_trans_p50_inliers": ct.get("p50"),
            "cam_trans_p90_inliers": ct.get("p90"),
            "cam_rot_deg_p50_inliers": cr.get("p50"),
            "cam_rot_deg_p90_inliers": cr.get("p90"),
            "point_nn_p50": pn.get("p50"),
            "point_nn_p90": pn.get("p90"),
            "point_match_rate": cmp.get("point_match_rate"),
        }
    else:
        out["comparison"] = {"error": cmp.get("error")}

    # Headline derived metrics.
    ce_total = (out.get("ceres") or {}).get("total_s")
    bae_total = (out.get("bae") or {}).get("total_s")
    if ce_total and bae_total:
        out["bae_speedup_total"] = float(ce_total) / float(bae_total)
    ce_pts = (out.get("ceres") or {}).get("points")
    bae_pts = (out.get("bae") or {}).get("points")
    if ce_pts and bae_pts:
        out["bae_point_retention_vs_ceres"] = float(bae_pts) / float(ce_pts)
    return out


def main():
    """Multi-dataset driver.

    Discovers every data/<name>/images/ directory and runs the benchmark
    for each.  Each dataset's full results live in
    bench/<name>/benchmark_results.json; a concise cross-dataset summary
    lives in bench/summary.json.
    """
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets()
    if not datasets:
        print(f"FATAL: no datasets found under {DATA_ROOT} (need "
              f"{DATA_ROOT}/<name>/images/)")
        sys.exit(1)
    print(f"\nDiscovered {len(datasets)} datasets: {datasets}\n", flush=True)

    # Resume-friendly aggregate summary: load existing, merge new results.
    summary = _load_json(SUMMARY_FILE) or {}
    summary.setdefault("started_at",
                       datetime.now().isoformat(timespec="seconds"))
    summary["last_seen_at"] = datetime.now().isoformat(timespec="seconds")
    summary.setdefault("datasets", {})
    _save_json_atomic(SUMMARY_FILE, summary)

    failures = []
    for ds in datasets:
        try:
            results = run_for_dataset(ds)
            summary["datasets"][ds] = _summarize_dataset(results)
        except Exception as e:
            print(f"\n!!! DATASET {ds} FAILED: {e!r}", flush=True)
            summary["datasets"][ds] = {"dataset": ds, "error": repr(e)}
            failures.append(ds)
        # Persist aggregate after every dataset (so partial progress
        # survives a crash on the next dataset).
        summary["last_seen_at"] = datetime.now().isoformat(timespec="seconds")
        _save_json_atomic(SUMMARY_FILE, summary)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _save_json_atomic(SUMMARY_FILE, summary)
    print(f"\n{'='*72}\n== AGGREGATE SUMMARY  →  {SUMMARY_FILE}\n{'='*72}",
          flush=True)

    # Cross-dataset console table.
    header = (f"  {'dataset':<12}{'speedup':>10}{'bae_pts':>12}"
              f"{'ceres_pts':>12}{'pt_keep':>10}{'inlier%':>10}"
              f"{'bae_total':>12}{'ce_total':>12}")
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for ds, s in summary["datasets"].items():
        if "error" in s:
            print(f"  {ds:<12}  ERROR: {s['error']}", flush=True)
            continue
        bae_pts = (s.get("bae") or {}).get("points") or 0
        ce_pts = (s.get("ceres") or {}).get("points") or 0
        bae_total = (s.get("bae") or {}).get("total_s") or 0
        ce_total = (s.get("ceres") or {}).get("total_s") or 0
        speedup = s.get("bae_speedup_total") or 0
        pt_keep = s.get("bae_point_retention_vs_ceres") or 0
        inlier = ((s.get("comparison") or {}).get("inlier_ratio") or 0) * 100
        print(
            f"  {ds:<12}{speedup:>9.2f}x{bae_pts:>12d}{ce_pts:>12d}"
            f"{pt_keep*100:>9.1f}%{inlier:>9.1f}%"
            f"{bae_total:>11.1f}s{ce_total:>11.1f}s",
            flush=True,
        )

    if failures:
        print(f"\nFAILED datasets: {failures}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
