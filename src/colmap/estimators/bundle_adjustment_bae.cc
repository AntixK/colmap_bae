#include "colmap/estimators/bundle_adjustment_bae.h"

#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// BaeBundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

bool BaeBundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GT(max_num_iterations, 0);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// BaeBundleAdjustmentSummary
////////////////////////////////////////////////////////////////////////////////

std::string BaeBundleAdjustmentSummary::BriefReport() const {
  return "BAE bundle adjustment report: termination=" +
         std::string(
             BundleAdjustmentTerminationTypeToString(termination_type)) +
         ", num_iterations=" + std::to_string(num_iterations) +
         ", initial_cost=" + std::to_string(initial_cost) +
         ", final_cost=" + std::to_string(final_cost);
}

////////////////////////////////////////////////////////////////////////////////
// BaeBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

class BaeBundleAdjuster : public BundleAdjuster {
 public:
  BaeBundleAdjuster(const BundleAdjustmentOptions& options,
                    const BundleAdjustmentConfig& config,
                    Reconstruction& reconstruction)
      : BundleAdjuster(options, config), reconstruction_(reconstruction) {
    SetupProblem();
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    auto summary = std::make_shared<BaeBundleAdjustmentSummary>();
    summary->termination_type = BundleAdjustmentTerminationType::FAILURE;
    summary->num_residuals = static_cast<int>(num_observations_ * 2);

    // Nothing to optimize.
    if (num_observations_ == 0 || num_images_ == 0) {
      summary->termination_type =
          BundleAdjustmentTerminationType::NO_CONVERGENCE;
      return summary;
    }

    // C1: Ensure Python interpreter is available.
    // CLI mode: no interpreter running -> initialize once, release GIL.
    // pycolmap mode: interpreter already running -> skip initialization.
    static std::once_flag py_init_flag;
    std::call_once(py_init_flag, []() {
      if (!Py_IsInitialized()) {
        py::initialize_interpreter();
        // Release GIL so we don't hold it between Solve() calls.
        PyEval_SaveThread();
      }
    });

    // Acquire the GIL for the duration of this call.
    py::gil_scoped_acquire gil;

    Timer timer;
    timer.Start();

    try {
      LogProbeResidualsFromArrays("pre_python_arrays");
      LogProbeResidualsFromFlatStateUsingCamera(
          "pre_python_arrays_colmap_camera");
      LogProbeResidualsFromReconstruction("pre_python_reconstruction");

      // Load bae_solver.py directly by file path.  We must NOT import the
      // pycolmap package (its __init__.py loads _core.so which calls
      // InitGoogleLogging, fatal when glog is already initialized by CLI).
      //
      // Search order:
      //  0. Environment override COLMAP_BAE_SOLVER_PATH (exact file path).
      //  1. Compile-time source tree path (development builds).
      //  2. Locate the installed pycolmap package directory on sys.path
      //     using find_spec (does NOT execute __init__.py).
      std::string solver_path;
      if (const char* env_solver_path =
              std::getenv("COLMAP_BAE_SOLVER_PATH");
          env_solver_path != nullptr && std::strlen(env_solver_path) > 0) {
        solver_path = env_solver_path;
      }
#ifdef BAE_SOLVER_MODULE_DIR
      if (solver_path.empty()) {
        solver_path =
            std::string(BAE_SOLVER_MODULE_DIR) + "/pycolmap/bae_solver.py";
      }
#endif
      if (solver_path.empty() || !std::filesystem::exists(solver_path)) {
        // find_spec locates the package without importing it.
        py::module_ importlib_util =
            py::module_::import("importlib.util");
        py::object pkg_spec =
            importlib_util.attr("find_spec")("pycolmap");
        THROW_CHECK(!pkg_spec.is_none())
            << "Cannot find pycolmap package on sys.path";
        std::string pkg_dir =
            pkg_spec.attr("submodule_search_locations")
                .cast<py::list>()[0]
                .cast<std::string>();
        solver_path = pkg_dir + "/bae_solver.py";
      }
      THROW_CHECK(std::filesystem::exists(solver_path))
          << "Cannot find BAE solver module at " << solver_path;
      LOG(INFO) << "BAE solver module: " << solver_path;

      py::module_ importlib_util =
          py::module_::import("importlib.util");
      auto file_spec = importlib_util.attr("spec_from_file_location")(
          "bae_solver", solver_path);
      py::module_ bae_solver =
          importlib_util.attr("module_from_spec")(file_spec);
      file_spec.attr("loader").attr("exec_module")(bae_solver);

      // Wrap C++ vectors as numpy arrays (zero-copy views into member data).
      const auto si = [](size_t n) { return static_cast<py::ssize_t>(n); };

      py::array_t<double> extr_arr({si(num_images_), si(3), si(4)},
                                   extrinsics_.data());
      py::array_t<double> intr_arr({si(num_cameras_), si(3)},
                                   intrinsics_.data());
      py::array_t<double> pts3d_arr({si(num_points_), si(3)},
                                    points_3d_.data());
      py::array_t<double> pts2d_arr({si(num_observations_), si(2)},
                                    points_2d_.data());
      // 1D arrays: pybind11's `array_t(ShapeContainer{N}, ptr)` constructor
      // mis-computes strides for 1D shapes (produces stride=0 → every read
      // returns the first element).  Pass explicit strides via buffer_info
      // to force the correct sizeof(T) stride.  Confirmed by the symptom
      // `unique=1 min=0 max=0` on the Python side despite the C++ vectors
      // being populated correctly.
      auto make_1d = [&](auto* ptr, size_t n) {
          using T = std::remove_pointer_t<decltype(ptr)>;
          return py::array_t<T>(py::buffer_info(
              ptr,
              sizeof(T),
              py::format_descriptor<T>::format(),
              1,
              {si(n)},
              {si(sizeof(T))}));
      };
      auto img_idx_arr = make_1d(image_indices_.data(), num_observations_);
      auto cam_idx_arr =
          make_1d(camera_obs_indices_.data(), num_observations_);
      auto pt_idx_arr = make_1d(point_indices_.data(), num_observations_);
      auto const_pose_arr =
          make_1d(constant_pose_mask_.data(), num_images_);
      auto const_pt_arr =
          make_1d(constant_point_mask_.data(), num_points_);
      std::vector<int> probe_img_indices;
      std::vector<int> probe_cam_indices;
      std::vector<int> probe_pt_indices;
      std::vector<double> probe_points_2d;
      py::list probe_labels;
      probe_img_indices.reserve(probes_.size());
      probe_cam_indices.reserve(probes_.size());
      probe_pt_indices.reserve(probes_.size());
      probe_points_2d.reserve(probes_.size() * 2);
      for (const ProbeObservation& probe : probes_) {
        probe_img_indices.push_back(static_cast<int>(probe.img_idx));
        probe_cam_indices.push_back(static_cast<int>(probe.cam_idx));
        probe_pt_indices.push_back(static_cast<int>(probe.pt_idx));
        probe_points_2d.push_back(probe.obs_x_centered);
        probe_points_2d.push_back(probe.obs_y_centered);
        std::ostringstream label;
        label << "image_id=" << probe.image_id << " point2D_idx="
              << probe.point2D_idx << " point3D_id=" << probe.point3D_id;
        probe_labels.append(py::str(label.str()));
      }
      auto probe_img_arr = make_1d(probe_img_indices.data(), probes_.size());
      auto probe_cam_arr = make_1d(probe_cam_indices.data(), probes_.size());
      auto probe_pt_arr = make_1d(probe_pt_indices.data(), probes_.size());
      py::array_t<double> probe_pts2d_arr({si(probes_.size()), si(2)},
                                          probe_points_2d.data());

      // Build options dict from BaeBundleAdjustmentOptions + refine_* flags.
      py::dict options_dict;
      if (options_.bae) {
        options_dict["max_num_iterations"] =
            options_.bae->max_num_iterations;
        if (!options_.bae->use_gpu) {
          LOG(WARNING) << "BAE backend is CUDA-only. Forcing "
                          "BundleAdjustmentBae.use_gpu=true.";
        }
        options_dict["use_gpu"] = true;
        options_dict["gpu_index"] = options_.bae->gpu_index;
      }
      options_dict["refine_focal_length"] = options_.refine_focal_length;
      options_dict["refine_extra_params"] = options_.refine_extra_params;
      options_dict["constant_rig_from_world_rotation"] =
          options_.constant_rig_from_world_rotation;
      options_dict["probe_image_indices"] = probe_img_arr;
      options_dict["probe_camera_indices"] = probe_cam_arr;
      options_dict["probe_point_indices"] = probe_pt_arr;
      options_dict["probe_points_2d"] = probe_pts2d_arr;
      options_dict["probe_labels"] = probe_labels;
      options_dict["gauge_mode"] =
          BundleAdjustmentGaugeToString(gauge_selection_.mode);
      options_dict["gauge_already_fixed"] = gauge_selection_.already_fixed;
      options_dict["gauge_anchor_image_idx"] =
          gauge_selection_.anchor_image_idx;
      options_dict["gauge_second_image_idx"] =
          gauge_selection_.second_image_idx;
      options_dict["gauge_anchor_image_id"] =
          static_cast<long long>(gauge_selection_.anchor_image_id);
      options_dict["gauge_second_image_id"] =
          static_cast<long long>(gauge_selection_.second_image_id);
      options_dict["gauge_anchor_frame_id"] =
          static_cast<long long>(gauge_selection_.anchor_frame_id);
      options_dict["gauge_second_frame_id"] =
          static_cast<long long>(gauge_selection_.second_frame_id);
      options_dict["gauge_second_translation_dim"] =
          gauge_selection_.second_translation_dim;
      options_dict["gauge_baseline_norm"] =
          gauge_selection_.baseline_norm;
      options_dict["gauge_baseline_locked_component"] =
          gauge_selection_.baseline_locked_component;
      if (!gauge_selection_.point_indices.empty()) {
        auto gauge_pt_arr = make_1d(gauge_selection_.point_indices.data(),
                                    gauge_selection_.point_indices.size());
        options_dict["gauge_point_indices"] = gauge_pt_arr;
      } else {
        options_dict["gauge_point_indices"] = py::array_t<int>();
      }

      // Call the Python BAE solver.
      py::dict result = bae_solver.attr("solve")(extr_arr,
                                                 intr_arr,
                                                 pts3d_arr,
                                                 pts2d_arr,
                                                 img_idx_arr,
                                                 cam_idx_arr,
                                                 pt_idx_arr,
                                                 const_pose_arr,
                                                 const_pt_arr,
                                                 options_dict);

      // Parse convergence info into summary.
      summary->num_iterations = result["num_iterations"].cast<int>();
      summary->initial_cost = result["initial_cost"].cast<double>();
      summary->final_cost = result["final_cost"].cast<double>();
      const bool converged = result["converged"].cast<bool>();
      summary->termination_type =
          converged ? BundleAdjustmentTerminationType::CONVERGENCE
                    : BundleAdjustmentTerminationType::NO_CONVERGENCE;

      // Copy optimized parameters back into member arrays.
      auto opt_extr = result["extrinsics"].cast<py::array_t<
          double, py::array::c_style | py::array::forcecast>>();
      auto opt_intr = result["intrinsics"].cast<py::array_t<
          double, py::array::c_style | py::array::forcecast>>();
      auto opt_pts = result["points_3d"].cast<py::array_t<
          double, py::array::c_style | py::array::forcecast>>();
      py::array_t<double, py::array::c_style | py::array::forcecast>
          opt_extr_se3;
      if (!options_.constant_rig_from_world_rotation) {
        THROW_CHECK(result.contains("extrinsics_se3_data"))
            << "BAE full BA did not return extrinsics_se3_data";
        opt_extr_se3 = result["extrinsics_se3_data"].cast<py::array_t<
            double, py::array::c_style | py::array::forcecast>>();
        THROW_CHECK_EQ(opt_extr_se3.ndim(), 2);
        THROW_CHECK_EQ(opt_extr_se3.shape(0), static_cast<py::ssize_t>(num_images_));
        THROW_CHECK_EQ(opt_extr_se3.shape(1), 7);
      }
      std::memcpy(extrinsics_.data(),
                  opt_extr.data(),
                  extrinsics_.size() * sizeof(double));
      std::memcpy(intrinsics_.data(),
                  opt_intr.data(),
                  intrinsics_.size() * sizeof(double));
      std::memcpy(points_3d_.data(),
                  opt_pts.data(),
                  points_3d_.size() * sizeof(double));
      LogProbeResidualsFromArrays("post_python_arrays");
      LogProbeResidualsFromFlatStateUsingCamera(
          "post_python_arrays_colmap_camera");

      // Write optimized extrinsics back to Reconstruction.
      std::unordered_map<frame_t, Rigid3d> frame_updates;
      frame_updates.reserve(image_id_to_idx_.size());
      for (const auto& [image_id, idx] : image_id_to_idx_) {
        if (constant_pose_mask_[idx]) continue;
        if (!reconstruction_.ExistsImage(image_id)) continue;
        auto& image = reconstruction_.Image(image_id);
        auto& frame = reconstruction_.Frame(image.FrameId());
        THROW_CHECK_LT(idx, image_frame_ids_.size());
        THROW_CHECK_EQ(image_frame_ids_[idx], image.FrameId());

        Rigid3d updated_pose = frame.RigFromWorld();
        if (options_.constant_rig_from_world_rotation) {
          const Eigen::Vector3d translation(extrinsics_[idx * 12 + 3],
                                            extrinsics_[idx * 12 + 7],
                                            extrinsics_[idx * 12 + 11]);
          updated_pose.translation() = translation;
        } else {
          const double* se3 = opt_extr_se3.data(idx, 0);
          updated_pose = Rigid3dFromPyPoseSe3Data(se3);
          const Eigen::Matrix3x4d matrix_from_se3 = updated_pose.ToMatrix();
          const Eigen::Matrix3x4d matrix_from_python =
              MatrixFromFlatExtrinsic(&extrinsics_[idx * 12]);
          const double matrix_diff =
              (matrix_from_se3 - matrix_from_python).norm();
          THROW_CHECK_LT(matrix_diff, 1e-6)
              << "PyPose SE3.data writeback mismatch for image " << image_id
              << " frame " << image.FrameId()
              << " matrix_diff=" << matrix_diff;
        }

        const auto [it, inserted] =
            frame_updates.emplace(image.FrameId(), updated_pose);
        if (!inserted) {
          const double pose_diff =
              (it->second.ToMatrix() - updated_pose.ToMatrix()).norm();
          THROW_CHECK_LT(pose_diff, 1e-6)
              << "Conflicting optimized poses for frame " << image.FrameId()
              << " pose_diff=" << pose_diff;
        }
      }
      for (const auto& [frame_id, pose] : frame_updates) {
        reconstruction_.Frame(frame_id).SetRigFromWorld(pose);
      }
      LogProbeResidualsFromReconstruction("post_extrinsics_writeback");
      LogPoseWritebackDiffs("post_extrinsics_writeback");

      // Write optimized intrinsics back to Reconstruction.
      if (options_.refine_focal_length || options_.refine_extra_params) {
        for (const auto& [cam_id, cidx] : camera_id_to_idx_) {
          if (!reconstruction_.ExistsCamera(cam_id)) continue;
          const double* ip = &intrinsics_[cidx * 3];
          auto& camera = reconstruction_.Camera(cam_id);
          if (options_.refine_focal_length) {
            camera.params[0] = ip[0];  // f
          }
          if (options_.refine_extra_params) {
            if (camera.model_id == CameraModelId::kSimpleRadial) {
              camera.params[3] = ip[1];  // k1
            } else {  // Radial
              camera.params[3] = ip[1];  // k1
              camera.params[4] = ip[2];  // k2
            }
          }
        }
      }
      LogProbeResidualsFromReconstruction("post_intrinsics_writeback");
      LogIntrinsicWritebackDiffs("post_intrinsics_writeback");

      // Write optimized 3D points back to Reconstruction.
      for (const auto& [point3D_id, idx] : point3D_id_to_idx_) {
        if (constant_point_mask_[idx]) continue;
        if (!reconstruction_.ExistsPoint3D(point3D_id)) continue;
        auto& point3D = reconstruction_.Point3D(point3D_id);
        point3D.xyz.x() = points_3d_[idx * 3 + 0];
        point3D.xyz.y() = points_3d_[idx * 3 + 1];
        point3D.xyz.z() = points_3d_[idx * 3 + 2];
      }
      LogProbeResidualsFromReconstruction("post_points_writeback");
      LogPointWritebackDiffs("post_points_writeback");
    } catch (py::error_already_set& e) {
      LOG(ERROR) << "BAE Python error: " << e.what();
      return summary;
    } catch (const std::exception& e) {
      LOG(ERROR) << "BAE solver error: " << e.what();
      return summary;
    }

    timer.Pause();

    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintBaeSolverSummary(*summary, timer.ElapsedSeconds());
    }

    return summary;
 }

 private:
  struct ProbeObservation {
    image_t image_id = kInvalidImageId;
    point2D_t point2D_idx = kInvalidPoint2DIdx;
    point3D_t point3D_id = kInvalidPoint3DId;
    size_t img_idx = 0;
    size_t cam_idx = 0;
    size_t pt_idx = 0;
    double obs_x_centered = 0.0;
    double obs_y_centered = 0.0;
  };

  struct GaugeSelection {
    BundleAdjustmentGauge mode = BundleAdjustmentGauge::UNSPECIFIED;
    bool already_fixed = false;
    int anchor_image_idx = -1;
    int second_image_idx = -1;
    int second_translation_dim = -1;
    image_t anchor_image_id = kInvalidImageId;
    image_t second_image_id = kInvalidImageId;
    frame_t anchor_frame_id = kInvalidFrameId;
    frame_t second_frame_id = kInvalidFrameId;
    double baseline_norm = 0.0;
    double baseline_locked_component = 0.0;
    std::vector<int> point_indices;
  };

  void SetupProblem();
  void SelectGaugeConstraints();
  static Eigen::Matrix3x4d MatrixFromFlatExtrinsic(const double* p);
  static Rigid3d Rigid3dFromPyPoseSe3Data(const double* p);
  void LogProbeResidualsFromArrays(const std::string& tag) const;
  void LogProbeResidualsFromFlatStateUsingCamera(
      const std::string& tag) const;
  void LogProbeResidualsFromReconstruction(const std::string& tag) const;
  bool ComputeProbeProjectionFromArrays(const ProbeObservation& probe,
                                        Eigen::Vector2d* proj_centered,
                                        double* depth) const;
  bool ComputeProbeProjectionFromFlatStateUsingCamera(
      const ProbeObservation& probe,
      Eigen::Vector2d* proj,
      double* depth) const;
  void LogPoseWritebackDiffs(const std::string& tag) const;
  void LogIntrinsicWritebackDiffs(const std::string& tag) const;
  void LogPointWritebackDiffs(const std::string& tag) const;
  static void LogProbeSummary(const std::string& tag,
                              const std::vector<ProbeObservation>& probes,
                              const std::vector<double>& errors,
                              const std::vector<double>& depths);

  static void PrintBaeSolverSummary(
      const BaeBundleAdjustmentSummary& summary, double elapsed_seconds) {
    std::ostringstream log;
    log << "BAE bundle adjustment report\n";
    log << std::right << std::setw(16) << "Residuals : " << std::left
        << summary.num_residuals << '\n';
    log << std::right << std::setw(16) << "Iterations : " << std::left
        << summary.num_iterations << '\n';
    log << std::right << std::setw(16) << "Time : " << std::left
        << elapsed_seconds << " [s]\n";
    log << std::right << std::setw(16) << "Initial cost : " << std::right
        << std::setprecision(6)
        << std::sqrt(summary.initial_cost /
                     std::max(summary.num_residuals, 1))
        << " [px]\n";
    log << std::right << std::setw(16) << "Final cost : " << std::right
        << std::setprecision(6)
        << std::sqrt(summary.final_cost /
                     std::max(summary.num_residuals, 1))
        << " [px]\n";
    log << std::right << std::setw(16) << "Termination : " << std::right
        << BundleAdjustmentTerminationTypeToString(summary.termination_type)
        << "\n\n";
    LOG(INFO) << log.str();
  }

  Reconstruction& reconstruction_;

  // Extracted flat arrays for Python BAE solver.
  size_t num_images_ = 0;
  size_t num_cameras_ = 0;
  size_t num_points_ = 0;
  size_t num_observations_ = 0;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  std::unordered_map<camera_t, size_t> camera_id_to_idx_;
  std::unordered_map<point3D_t, size_t> point3D_id_to_idx_;

  // (num_images_ * 12): row-major 3x4 [R | t] world-to-camera matrix.
  std::vector<double> extrinsics_;
  // (num_cameras_ * 3): [f, k1, k2] per unique camera.
  std::vector<double> intrinsics_;
  // (num_points_ * 3): [x, y, z] per point.
  std::vector<double> points_3d_;
  // (num_observations_ * 2): centered 2D observations.
  std::vector<double> points_2d_;
  // (num_observations_): image index per observation (for extrinsics).
  std::vector<int> image_indices_;
  // (num_observations_): camera index per observation (for intrinsics).
  std::vector<int> camera_obs_indices_;
  // (num_observations_): point index per observation.
  std::vector<int> point_indices_;
  // (num_images_): 1 if camera pose is constant.
  std::vector<uint8_t> constant_pose_mask_;
  // (num_points_): 1 if 3D point is constant.
  std::vector<uint8_t> constant_point_mask_;
  // (num_images_): camera index for each image (for intrinsics lookup).
  std::vector<int> image_camera_idx_;
  // (num_images_): camera_id for each image (for intrinsics writeback).
  std::vector<camera_t> image_camera_ids_;
  // (num_images_): image_id for each image pose row.
  std::vector<image_t> image_image_ids_;
  // (num_images_): frame_id for each image pose row.
  std::vector<frame_t> image_frame_ids_;
  // (num_cameras_ * 2): [cx, cy] captured during extraction.
  std::vector<double> principal_points_;
  // Deterministic observation probes traced across Python/C++ boundaries.
  std::vector<ProbeObservation> probes_;
  // Stationary gauge constraints selected to match COLMAP's Ceres policy.
  GaugeSelection gauge_selection_;
};

void BaeBundleAdjuster::LogProbeSummary(
    const std::string& tag,
    const std::vector<ProbeObservation>& probes,
    const std::vector<double>& errors,
    const std::vector<double>& depths) {
  if (probes.empty()) {
    LOG(INFO) << "[BAE probe " << tag << "] no probes configured";
    return;
  }

  if (errors.empty()) {
    LOG(INFO) << "[BAE probe " << tag << "] no valid probe residuals";
    return;
  }

  std::vector<double> sorted_errors = errors;
  std::sort(sorted_errors.begin(), sorted_errors.end());
  const auto pct = [&](double q) {
    const size_t idx = static_cast<size_t>(
        q * static_cast<double>(sorted_errors.size() - 1));
    return sorted_errors[idx];
  };
  LOG(INFO) << "[BAE probe " << tag << "] n=" << sorted_errors.size()
            << " p50=" << pct(0.50) << " p90=" << pct(0.90)
            << " max=" << sorted_errors.back();

  for (size_t i = 0; i < errors.size(); ++i) {
    std::ostringstream line;
    line << "[BAE probe " << tag << "] #" << i
         << " image_id=" << probes[i].image_id
         << " point2D_idx=" << probes[i].point2D_idx
         << " point3D_id=" << probes[i].point3D_id
         << " err_px=" << errors[i];
    if (std::isfinite(depths[i])) {
      line << " depth=" << depths[i];
    } else {
      line << " depth=invalid";
    }
    LOG(INFO) << line.str();
  }
}

Eigen::Matrix3x4d BaeBundleAdjuster::MatrixFromFlatExtrinsic(const double* p) {
  Eigen::Matrix3x4d matrix;
  matrix << p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
      p[10], p[11];
  return matrix;
}

Rigid3d BaeBundleAdjuster::Rigid3dFromPyPoseSe3Data(const double* p) {
  const Eigen::Vector3d translation(p[0], p[1], p[2]);
  Eigen::Quaterniond rotation(p[6], p[3], p[4], p[5]);
  rotation.normalize();
  return Rigid3d(rotation, translation);
}

void BaeBundleAdjuster::SelectGaugeConstraints() {
  gauge_selection_ = GaugeSelection{};
  gauge_selection_.mode = config_.FixedGauge();

  if (!options_.refine_rig_from_world) {
    LOG(INFO) << "[BAE gauge] pose refinement disabled; no extra gauge fix";
    return;
  }

  if (gauge_selection_.mode == BundleAdjustmentGauge::UNSPECIFIED) {
    LOG(INFO) << "[BAE gauge] unspecified; no extra gauge fix";
    return;
  }

  auto maybe_add_fixed_point = [](const Eigen::Vector3d& xyz,
                                  Eigen::Matrix3d* fixed_points,
                                  Eigen::Index* num_fixed_points) {
    if (*num_fixed_points >= 3) {
      return false;
    }
    fixed_points->col(*num_fixed_points) = xyz;
    if (fixed_points->colPivHouseholderQr().rank() > *num_fixed_points) {
      ++(*num_fixed_points);
      return true;
    }
    fixed_points->col(*num_fixed_points).setZero();
    return false;
  };

  auto select_three_points = [&]() {
    Eigen::Matrix3d fixed_points = Eigen::Matrix3d::Zero();
    Eigen::Index num_fixed_points = 0;

    for (const auto& [point3D_id, idx] : point3D_id_to_idx_) {
      if (!constant_point_mask_[idx] ||
          !reconstruction_.ExistsPoint3D(point3D_id)) {
        continue;
      }
      const Point3D& point3D = reconstruction_.Point3D(point3D_id);
      if (maybe_add_fixed_point(
              point3D.xyz, &fixed_points, &num_fixed_points) &&
          num_fixed_points >= 3) {
        gauge_selection_.already_fixed = true;
        gauge_selection_.point_indices.clear();
        return;
      }
    }

    gauge_selection_.point_indices.clear();
    for (const auto& [point3D_id, idx] : point3D_id_to_idx_) {
      if (constant_point_mask_[idx] ||
          !reconstruction_.ExistsPoint3D(point3D_id)) {
        continue;
      }
      const Point3D& point3D = reconstruction_.Point3D(point3D_id);
      if (maybe_add_fixed_point(
              point3D.xyz, &fixed_points, &num_fixed_points)) {
        gauge_selection_.point_indices.push_back(static_cast<int>(idx));
        if (num_fixed_points >= 3) {
          return;
        }
      }
    }

    LOG(WARNING) << "[BAE gauge] failed to select three independent points; "
                 << "num_fixed_points=" << num_fixed_points;
    gauge_selection_.point_indices.clear();
  };

  if (gauge_selection_.mode == BundleAdjustmentGauge::THREE_POINTS) {
    select_three_points();
    LOG(INFO) << "[BAE gauge] mode=three_points already_fixed="
              << gauge_selection_.already_fixed
              << " selected_points=" << gauge_selection_.point_indices.size();
    return;
  }

  THROW_CHECK_EQ(gauge_selection_.mode,
                 BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  auto is_parameterized_const_sensor = [&](const Image& image) {
    const sensor_t sensor_id = image.CameraPtr()->SensorId();
    if (image.FramePtr()->RigPtr()->IsRefSensor(sensor_id)) {
      return true;
    }
    if (config_.HasConstantSensorFromRigPose(sensor_id) ||
        !options_.refine_sensor_from_rig) {
      return true;
    }
    return false;
  };

  Image* image1 = nullptr;
  Image* image2 = nullptr;
  int image1_idx = -1;
  int image2_idx = -1;
  int second_dim = -1;
  Eigen::Vector3d selected_baseline = Eigen::Vector3d::Zero();

  // Match Ceres: first search through already fixed images in the BA image set.
  for (const image_t image_id : config_.Images()) {
    auto idx_it = image_id_to_idx_.find(image_id);
    if (idx_it == image_id_to_idx_.end()) {
      continue;
    }
    if (!reconstruction_.ExistsImage(image_id)) {
      continue;
    }
    Image& image = reconstruction_.Image(image_id);
    if (config_.HasConstantRigFromWorldPose(image.FrameId()) &&
        is_parameterized_const_sensor(image)) {
      if (image1 == nullptr) {
        image1 = &image;
        image1_idx = static_cast<int>(idx_it->second);
      } else if (image1->FrameId() != image.FrameId()) {
        gauge_selection_.already_fixed = true;
        gauge_selection_.anchor_image_idx = image1_idx;
        gauge_selection_.anchor_image_id = image1->ImageId();
        gauge_selection_.anchor_frame_id = image1->FrameId();
        gauge_selection_.second_image_idx = static_cast<int>(idx_it->second);
        gauge_selection_.second_image_id = image.ImageId();
        gauge_selection_.second_frame_id = image.FrameId();
        LOG(INFO) << "[BAE gauge] mode=two_cams_from_world already_fixed=true"
                  << " image1_id=" << gauge_selection_.anchor_image_id
                  << " image2_id=" << gauge_selection_.second_image_id
                  << " frame1_id=" << gauge_selection_.anchor_frame_id
                  << " frame2_id=" << gauge_selection_.second_frame_id;
        return;
      }
    }
  }

  // Otherwise, search through variable images in the BA image set.
  //
  // Difference from Ceres: Ceres picks the first (image1, image2) pair with
  // any nonzero baseline and breaks. For sequentially-captured datasets that
  // pair is consecutive frames, with baseline often << 1% of scene radius;
  // observed values include 0.028 (bridge), 0.0044 (barn). A microscopic
  // locked baseline leaves the scale gauge effectively free, so PCG's
  // per-component error along that direction lets sub-graphs drift to
  // different metric scales — visible as intersecting planes in the point
  // cloud and as low cross-backend RANSAC inlier ratios.
  //
  // To produce a strong, scale-pinning constraint we iterate over all
  // (image1, image2) candidate pairs and pick the one with the largest
  // baseline component. The locked dimension is the argmax component of
  // that baseline, matching the Ceres SubsetManifold convention. Cost is
  // O(N_const_sensor_imgs^2) — sub-second even on multi-thousand-image
  // datasets.
  struct Candidate {
    Image* image;
    int idx;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(num_images_);
  for (const image_t image_id : config_.Images()) {
    auto idx_it = image_id_to_idx_.find(image_id);
    if (idx_it == image_id_to_idx_.end()) {
      continue;
    }
    if (!reconstruction_.ExistsImage(image_id)) {
      continue;
    }
    Image& image = reconstruction_.Image(image_id);
    if (!is_parameterized_const_sensor(image)) {
      continue;
    }
    candidates.push_back({&image, static_cast<int>(idx_it->second)});
  }

  double best_max_component = 0.0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    Image* cand1 = candidates[i].image;
    const int cand1_idx = candidates[i].idx;
    for (size_t j = 0; j < candidates.size(); ++j) {
      if (i == j) {
        continue;
      }
      Image* cand2 = candidates[j].image;
      const int cand2_idx = candidates[j].idx;
      if (cand1->FrameId() == cand2->FrameId()) {
        continue;
      }
      if (constant_pose_mask_[cand2_idx]) {
        continue;
      }
      const Eigen::Vector3d baseline =
          (cand1->FramePtr()->RigFromWorld() *
           Inverse(cand2->FramePtr()->RigFromWorld()))
              .translation();
      Eigen::Index max_coeff_idx = 0;
      const double max_component = baseline.cwiseAbs().maxCoeff(&max_coeff_idx);
      if (max_component > best_max_component) {
        best_max_component = max_component;
        image1 = cand1;
        image2 = cand2;
        image1_idx = cand1_idx;
        image2_idx = cand2_idx;
        second_dim = static_cast<int>(max_coeff_idx);
        selected_baseline = baseline;
      }
    }
  }

  if (best_max_component <= 1e-9) {
    // No pair has a usable baseline; let the fallback below switch to
    // THREE_POINTS gauge.
    image1 = nullptr;
    image2 = nullptr;
    image1_idx = -1;
    image2_idx = -1;
    second_dim = -1;
    selected_baseline = Eigen::Vector3d::Zero();
  }

  if (image1 == nullptr || image2 == nullptr || image1_idx < 0 ||
      image2_idx < 0 || second_dim < 0) {
    LOG(WARNING) << "[BAE gauge] failed to select two-camera gauge; "
                    "falling back to three points";
    gauge_selection_.mode = BundleAdjustmentGauge::THREE_POINTS;
    select_three_points();
    LOG(INFO) << "[BAE gauge] mode=three_points already_fixed="
              << gauge_selection_.already_fixed
              << " selected_points=" << gauge_selection_.point_indices.size();
    return;
  }

  gauge_selection_.anchor_image_idx = image1_idx;
  gauge_selection_.second_image_idx = image2_idx;
  gauge_selection_.second_translation_dim = second_dim;
  gauge_selection_.anchor_image_id = image1->ImageId();
  gauge_selection_.second_image_id = image2->ImageId();
  gauge_selection_.anchor_frame_id = image1->FrameId();
  gauge_selection_.second_frame_id = image2->FrameId();
  gauge_selection_.baseline_norm = selected_baseline.norm();
  gauge_selection_.baseline_locked_component =
      std::abs(selected_baseline[second_dim]);
  LOG(INFO) << "[BAE gauge] mode=two_cams_from_world anchor_image_idx="
            << gauge_selection_.anchor_image_idx
            << " second_image_idx=" << gauge_selection_.second_image_idx
            << " image1_id=" << gauge_selection_.anchor_image_id
            << " image2_id=" << gauge_selection_.second_image_id
            << " frame1_id=" << gauge_selection_.anchor_frame_id
            << " frame2_id=" << gauge_selection_.second_frame_id
            << " second_translation_dim="
            << gauge_selection_.second_translation_dim
            << " baseline_norm=" << gauge_selection_.baseline_norm
            << " locked_component_abs="
            << gauge_selection_.baseline_locked_component;
}

bool BaeBundleAdjuster::ComputeProbeProjectionFromArrays(
    const ProbeObservation& probe,
    Eigen::Vector2d* proj_centered,
    double* depth) const {
  THROW_CHECK_LT(probe.img_idx, num_images_);
  THROW_CHECK_LT(probe.cam_idx, num_cameras_);
  THROW_CHECK_LT(probe.pt_idx, num_points_);

  const Eigen::Matrix3x4d extrinsic =
      MatrixFromFlatExtrinsic(&extrinsics_[probe.img_idx * 12]);
  const Eigen::Vector3d xyz(points_3d_[probe.pt_idx * 3 + 0],
                            points_3d_[probe.pt_idx * 3 + 1],
                            points_3d_[probe.pt_idx * 3 + 2]);
  const Eigen::Vector3d point_cam =
      extrinsic.leftCols<3>() * xyz + extrinsic.col(3);
  *depth = point_cam.z();
  if (point_cam.z() <= std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double* ip = &intrinsics_[probe.cam_idx * 3];
  const double focal = ip[0];
  const double k1 = ip[1];
  const double k2 = ip[2];
  const double u = point_cam.x() / point_cam.z();
  const double v = point_cam.y() / point_cam.z();
  const double r2 = u * u + v * v;
  const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
  *proj_centered = focal * radial * Eigen::Vector2d(u, v);
  return true;
}

bool BaeBundleAdjuster::ComputeProbeProjectionFromFlatStateUsingCamera(
    const ProbeObservation& probe,
    Eigen::Vector2d* proj,
    double* depth) const {
  THROW_CHECK_LT(probe.img_idx, num_images_);
  THROW_CHECK_LT(probe.cam_idx, num_cameras_);
  THROW_CHECK_LT(probe.pt_idx, num_points_);

  if (!reconstruction_.ExistsImage(probe.image_id)) {
    return false;
  }
  const Image& image = reconstruction_.Image(probe.image_id);
  if (!reconstruction_.ExistsCamera(image.CameraId())) {
    return false;
  }

  const Eigen::Matrix3x4d extrinsic =
      MatrixFromFlatExtrinsic(&extrinsics_[probe.img_idx * 12]);
  const Eigen::Vector3d xyz(points_3d_[probe.pt_idx * 3 + 0],
                            points_3d_[probe.pt_idx * 3 + 1],
                            points_3d_[probe.pt_idx * 3 + 2]);
  const Eigen::Vector3d point_cam =
      extrinsic.leftCols<3>() * xyz + extrinsic.col(3);
  *depth = point_cam.z();
  if (point_cam.z() <= std::numeric_limits<double>::epsilon()) {
    return false;
  }

  Camera camera = reconstruction_.Camera(image.CameraId());
  const double* ip = &intrinsics_[probe.cam_idx * 3];
  camera.params[0] = ip[0];
  if (camera.model_id == CameraModelId::kSimpleRadial) {
    camera.params[3] = ip[1];
  } else {
    camera.params[3] = ip[1];
    camera.params[4] = ip[2];
  }

  const auto candidate_proj = camera.ImgFromCam(point_cam);
  if (!candidate_proj.has_value()) {
    return false;
  }
  *proj = *candidate_proj;
  return true;
}

void BaeBundleAdjuster::LogProbeResidualsFromArrays(
    const std::string& tag) const {
  std::vector<ProbeObservation> valid_probes;
  std::vector<double> errors;
  std::vector<double> depths;
  valid_probes.reserve(probes_.size());
  errors.reserve(probes_.size());
  depths.reserve(probes_.size());

  for (const ProbeObservation& probe : probes_) {
    Eigen::Vector2d proj_centered;
    double depth = std::numeric_limits<double>::quiet_NaN();
    if (!ComputeProbeProjectionFromArrays(probe, &proj_centered, &depth)) {
      continue;
    }
    const Eigen::Vector2d obs(probe.obs_x_centered, probe.obs_y_centered);
    valid_probes.push_back(probe);
    errors.push_back((proj_centered - obs).norm());
    depths.push_back(depth);
  }

  LogProbeSummary(tag, valid_probes, errors, depths);
}

void BaeBundleAdjuster::LogProbeResidualsFromFlatStateUsingCamera(
    const std::string& tag) const {
  std::vector<ProbeObservation> valid_probes;
  std::vector<double> errors;
  std::vector<double> depths;
  valid_probes.reserve(probes_.size());
  errors.reserve(probes_.size());
  depths.reserve(probes_.size());

  for (const ProbeObservation& probe : probes_) {
    if (!reconstruction_.ExistsImage(probe.image_id)) {
      continue;
    }
    const Image& image = reconstruction_.Image(probe.image_id);
    Eigen::Vector2d proj;
    double depth = std::numeric_limits<double>::quiet_NaN();
    if (!ComputeProbeProjectionFromFlatStateUsingCamera(
            probe, &proj, &depth)) {
      continue;
    }
    const Point2D& point2D = image.Point2D(probe.point2D_idx);
    valid_probes.push_back(probe);
    errors.push_back((proj - point2D.xy).norm());
    depths.push_back(depth);
  }

  LogProbeSummary(tag, valid_probes, errors, depths);
}

void BaeBundleAdjuster::LogProbeResidualsFromReconstruction(
    const std::string& tag) const {
  std::vector<ProbeObservation> valid_probes;
  std::vector<double> errors;
  std::vector<double> depths;
  valid_probes.reserve(probes_.size());
  errors.reserve(probes_.size());
  depths.reserve(probes_.size());

  for (const ProbeObservation& probe : probes_) {
    if (!reconstruction_.ExistsImage(probe.image_id) ||
        !reconstruction_.ExistsPoint3D(probe.point3D_id)) {
      continue;
    }
    const Image& image = reconstruction_.Image(probe.image_id);
    if (!image.HasPose() || !image.HasCameraPtr()) {
      continue;
    }
    const auto proj =
        image.ProjectPoint(reconstruction_.Point3D(probe.point3D_id).xyz);
    if (!proj.has_value()) {
      continue;
    }
    const Point2D& point2D = image.Point2D(probe.point2D_idx);
    const Eigen::Vector3d point_cam =
        image.CamFromWorld() * reconstruction_.Point3D(probe.point3D_id).xyz;
    valid_probes.push_back(probe);
    errors.push_back((*proj - point2D.xy).norm());
    depths.push_back(point_cam.z());
  }

  LogProbeSummary(tag, valid_probes, errors, depths);
}

void BaeBundleAdjuster::LogPoseWritebackDiffs(const std::string& tag) const {
  std::unordered_set<image_t> seen_images;
  for (const ProbeObservation& probe : probes_) {
    if (!seen_images.insert(probe.image_id).second) {
      continue;
    }
    if (!reconstruction_.ExistsImage(probe.image_id)) {
      continue;
    }
    const Image& image = reconstruction_.Image(probe.image_id);
    if (!image.HasPose()) {
      continue;
    }

    const Eigen::Matrix3x4d flat_matrix =
        MatrixFromFlatExtrinsic(&extrinsics_[probe.img_idx * 12]);
    const Eigen::Matrix3x4d reconstruction_matrix =
        image.CamFromWorld().ToMatrix();
    const double matrix_diff = (flat_matrix - reconstruction_matrix).norm();

    std::ostringstream line;
    line << "[BAE diff " << tag << " pose] image_id=" << probe.image_id
         << " frame_id=" << image.FrameId()
         << " matrix_diff=" << matrix_diff
         << " flat_t=[" << flat_matrix(0, 3) << ", " << flat_matrix(1, 3)
         << ", " << flat_matrix(2, 3) << "]"
         << " recon_t=[" << reconstruction_matrix(0, 3) << ", "
         << reconstruction_matrix(1, 3) << ", "
         << reconstruction_matrix(2, 3) << "]";
    LOG(INFO) << line.str();
  }
}

void BaeBundleAdjuster::LogIntrinsicWritebackDiffs(
    const std::string& tag) const {
  std::unordered_set<size_t> seen_cameras;
  for (const ProbeObservation& probe : probes_) {
    if (!seen_cameras.insert(probe.cam_idx).second) {
      continue;
    }
    THROW_CHECK_LT(probe.cam_idx, num_cameras_);
    THROW_CHECK_LT(probe.cam_idx * 2 + 1, principal_points_.size());

    const camera_t camera_id = image_camera_ids_[probe.img_idx];
    if (!reconstruction_.ExistsCamera(camera_id)) {
      continue;
    }
    const Camera& camera = reconstruction_.Camera(camera_id);
    const double* ip = &intrinsics_[probe.cam_idx * 3];
    const double cx_extracted = principal_points_[probe.cam_idx * 2 + 0];
    const double cy_extracted = principal_points_[probe.cam_idx * 2 + 1];

    std::ostringstream line;
    line << "[BAE diff " << tag << " intr] camera_id=" << camera_id
         << " cam_idx=" << probe.cam_idx
         << " f_buf=" << ip[0]
         << " f_cam=" << camera.params[0]
         << " df=" << (camera.params[0] - ip[0])
         << " cx_extracted=" << cx_extracted
         << " cx_cam=" << camera.params[1]
         << " dcx=" << (camera.params[1] - cx_extracted)
         << " cy_extracted=" << cy_extracted
         << " cy_cam=" << camera.params[2]
         << " dcy=" << (camera.params[2] - cy_extracted);
    if (camera.model_id == CameraModelId::kSimpleRadial) {
      line << " k1_buf=" << ip[1]
           << " k1_cam=" << camera.params[3]
           << " dk1=" << (camera.params[3] - ip[1]);
    } else {
      line << " k1_buf=" << ip[1]
           << " k1_cam=" << camera.params[3]
           << " dk1=" << (camera.params[3] - ip[1])
           << " k2_buf=" << ip[2]
           << " k2_cam=" << camera.params[4]
           << " dk2=" << (camera.params[4] - ip[2]);
    }
    LOG(INFO) << line.str();
  }
}

void BaeBundleAdjuster::LogPointWritebackDiffs(const std::string& tag) const {
  std::unordered_set<point3D_t> seen_points;
  for (const ProbeObservation& probe : probes_) {
    if (!seen_points.insert(probe.point3D_id).second) {
      continue;
    }
    if (!reconstruction_.ExistsPoint3D(probe.point3D_id)) {
      continue;
    }
    const Eigen::Vector3d raw_point(points_3d_[probe.pt_idx * 3 + 0],
                                    points_3d_[probe.pt_idx * 3 + 1],
                                    points_3d_[probe.pt_idx * 3 + 2]);
    const Eigen::Vector3d reconstruction_point =
        reconstruction_.Point3D(probe.point3D_id).xyz;
    const Eigen::Vector3d diff = reconstruction_point - raw_point;

    std::ostringstream line;
    line << "[BAE diff " << tag << " point] point3D_id=" << probe.point3D_id
         << " pt_idx=" << probe.pt_idx
         << " diff_norm=" << diff.norm()
         << " raw=[" << raw_point.x() << ", " << raw_point.y() << ", "
         << raw_point.z() << "]"
         << " recon=[" << reconstruction_point.x() << ", "
         << reconstruction_point.y() << ", " << reconstruction_point.z()
         << "]";
    LOG(INFO) << line.str();
  }
}

void BaeBundleAdjuster::SetupProblem() {
  // Validate: all cameras use SIMPLE_RADIAL.
  // Currently BAE does not support multiRig camera setups, and the Python
  // solver only models COLMAP SIMPLE_RADIAL distortion (f, cx, cy, k1).
  for (const image_t image_id : config_.Images()) {
    if (!reconstruction_.ExistsImage(image_id)) continue;
    const auto& image = reconstruction_.Image(image_id);
    const auto& camera = reconstruction_.Camera(image.CameraId());
    THROW_CHECK(camera.model_id == CameraModelId::kSimpleRadial)
        << "BAE currently only supports SimpleRadial camera models";
    THROW_CHECK(image.IsRefInFrame())
        << "BAE does not support multi-sensor rigs";
  }

  // First pass: collect point3D_ids and discover which images have
  // observations.  Only images with at least one valid observation are
  // included in the BA problem — this avoids zero-Jacobian columns that
  // trigger sparse-solver bugs in BAE.
  std::unordered_set<image_t> images_with_obs;
  for (const image_t image_id : config_.Images()) {
    if (!reconstruction_.ExistsImage(image_id)) continue;
    const auto& image = reconstruction_.Image(image_id);
    for (const auto& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) continue;
      if (config_.IsIgnoredPoint(point2D.point3D_id)) continue;
      if (!reconstruction_.ExistsPoint3D(point2D.point3D_id)) continue;
      const auto& point3D = reconstruction_.Point3D(point2D.point3D_id);
      if (options_.min_track_length > 0 &&
          static_cast<int>(point3D.track.Length()) <
              options_.min_track_length) {
        continue;
      }
      if (point3D_id_to_idx_.count(point2D.point3D_id) == 0) {
        point3D_id_to_idx_[point2D.point3D_id] = num_points_++;
      }
      images_with_obs.insert(image_id);
    }
  }

  // Build image_id -> contiguous index map and camera_id -> index map,
  // including only images that have at least one observation.
  for (const image_t image_id : config_.Images()) {
    if (images_with_obs.count(image_id) == 0) continue;
    image_id_to_idx_[image_id] = num_images_++;
    const auto& image = reconstruction_.Image(image_id);
    camera_t cam_id = image.CameraId();
    if (camera_id_to_idx_.count(cam_id) == 0) {
      camera_id_to_idx_[cam_id] = num_cameras_++;
    }
  }

  // Append the extra points from config's VariablePoints and ConstantPoints
  // that are not already included above from the config images.
  auto collect_extra_point = [&](const point3D_t point3D_id) {
    if (config_.IsIgnoredPoint(point3D_id)) return;
    if (!reconstruction_.ExistsPoint3D(point3D_id)) return;
    const auto& point3D = reconstruction_.Point3D(point3D_id);
    if (options_.min_track_length > 0 &&
        static_cast<int>(point3D.track.Length()) <
            options_.min_track_length) {
      return;
    }
    if (point3D_id_to_idx_.count(point3D_id) == 0) {
      point3D_id_to_idx_[point3D_id] = num_points_++;
    }
  };
  for (const auto point3D_id : config_.VariablePoints()) {
    collect_extra_point(point3D_id);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    collect_extra_point(point3D_id);
  }

  // Extract extrinsics (per-image) and intrinsics (per-camera).
  extrinsics_.resize(num_images_ * 12);
  intrinsics_.resize(num_cameras_ * 3);
  constant_pose_mask_.resize(num_images_, 0);
  image_camera_idx_.resize(num_images_);
  image_camera_ids_.resize(num_images_);
  image_image_ids_.resize(num_images_);
  image_frame_ids_.resize(num_images_);
  principal_points_.assign(num_cameras_ * 2, 0.0);

  for (const image_t image_id : config_.Images()) {
    auto it = image_id_to_idx_.find(image_id);
    if (it == image_id_to_idx_.end()) continue;  // No observations for image.
    const size_t idx = it->second;
    const auto& image = reconstruction_.Image(image_id);
    const auto& camera = reconstruction_.Camera(image.CameraId());
    const Rigid3d& rig_from_world =
        reconstruction_.Frame(image.FrameId()).RigFromWorld();

    // Extrinsics: row-major 3x4 world-to-camera matrix [R | t].
    double* p = &extrinsics_[idx * 12];
    const Eigen::Matrix3d rotation =
        rig_from_world.rotation().toRotationMatrix();
    const auto t = rig_from_world.translation();
    p[0] = rotation(0, 0);
    p[1] = rotation(0, 1);
    p[2] = rotation(0, 2);
    p[3] = t.x();
    p[4] = rotation(1, 0);
    p[5] = rotation(1, 1);
    p[6] = rotation(1, 2);
    p[7] = t.y();
    p[8] = rotation(2, 0);
    p[9] = rotation(2, 1);
    p[10] = rotation(2, 2);
    p[11] = t.z();

    // Map image -> camera index.
    const size_t cam_idx = camera_id_to_idx_.at(image.CameraId());
    image_camera_idx_[idx] = static_cast<int>(cam_idx);
    image_camera_ids_[idx] = image.CameraId();
    image_image_ids_[idx] = image_id;
    image_frame_ids_[idx] = image.FrameId();

    // Intrinsics: [f, k1, k2] per unique camera (written once per camera).
    double* ip = &intrinsics_[cam_idx * 3];
    ip[0] = camera.params[0];
    principal_points_[cam_idx * 2 + 0] = camera.params[1];
    principal_points_[cam_idx * 2 + 1] = camera.params[2];
    if (camera.model_id == CameraModelId::kSimpleRadial) {
      ip[1] = camera.params[3]; // k1
      ip[2] = 0.0;
    } else {  // Radial
      ip[1] = camera.params[3];
      ip[2] = camera.params[4];
    }

    constant_pose_mask_[idx] =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());
  }

  // Extract points_3d and constant_point_mask.
  points_3d_.resize(num_points_ * 3);
  constant_point_mask_.resize(num_points_, 0);
  for (const auto& [point3D_id, idx] : point3D_id_to_idx_) {
    if (!reconstruction_.ExistsPoint3D(point3D_id)) continue;
    const auto& point3D = reconstruction_.Point3D(point3D_id);
    points_3d_[idx * 3 + 0] = point3D.xyz.x();
    points_3d_[idx * 3 + 1] = point3D.xyz.y();
    points_3d_[idx * 3 + 2] = point3D.xyz.z();
    constant_point_mask_[idx] =
        !options_.refine_points3D || config_.HasConstantPoint(point3D_id);
  }

  // Extract observations from config images.
  std::unordered_map<image_t, size_t> probe_counts_per_image;
  constexpr size_t kMaxProbeObservations = 12;
  constexpr size_t kMaxProbePerImage = 2;
  for (const image_t image_id : config_.Images()) {
    auto obs_it = image_id_to_idx_.find(image_id);
    if (obs_it == image_id_to_idx_.end()) continue;  // No observations.
    const size_t img_idx = obs_it->second;
    const auto& image = reconstruction_.Image(image_id);
    const auto& camera = reconstruction_.Camera(image.CameraId());
    const double cx = camera.params[1];
    const double cy = camera.params[2];

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const auto& point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) continue;
      if (config_.IsIgnoredPoint(point2D.point3D_id)) continue;
      if (!reconstruction_.ExistsPoint3D(point2D.point3D_id)) continue;
      auto it = point3D_id_to_idx_.find(point2D.point3D_id);
      if (it == point3D_id_to_idx_.end()) continue;
      // Center around principal point (COLMAP convention: obs - cx).
      points_2d_.push_back(point2D.xy.x() - cx);
      points_2d_.push_back(point2D.xy.y() - cy);
      image_indices_.push_back(static_cast<int>(img_idx));
      camera_obs_indices_.push_back(image_camera_idx_[img_idx]);
      point_indices_.push_back(static_cast<int>(it->second));
      if (probes_.size() < kMaxProbeObservations &&
          probe_counts_per_image[image_id] < kMaxProbePerImage) {
        probes_.push_back(ProbeObservation{image_id,
                                           point2D_idx,
                                           point2D.point3D_id,
                                           img_idx,
                                           static_cast<size_t>(
                                               image_camera_idx_[img_idx]),
                                           it->second,
                                           point2D.xy.x() - cx,
                                           point2D.xy.y() - cy});
        ++probe_counts_per_image[image_id];
      }
      ++num_observations_;
    }
  }

  // Handle VariablePoints/ConstantPoints extra residuals.
  // Add observations from external images with frozen poses.
  auto add_external_obs = [&](const point3D_t point3D_id) {
    auto pt_it = point3D_id_to_idx_.find(point3D_id);
    if (pt_it == point3D_id_to_idx_.end()) return;
    if (!reconstruction_.ExistsPoint3D(point3D_id)) return;
    const auto& point3D = reconstruction_.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (config_.HasImage(track_el.image_id)) continue;
      if (!reconstruction_.ExistsImage(track_el.image_id)) continue;
      const auto& ext_image = reconstruction_.Image(track_el.image_id);
      THROW_CHECK(ext_image.IsRefInFrame())
          << "BAE does not support multi-sensor rigs (external image)";
      const auto& ext_camera =
          reconstruction_.Camera(ext_image.CameraId());
      if (ext_camera.model_id != CameraModelId::kSimpleRadial) {
        continue;
      }

      // Add external image as frozen camera if not yet added.
      size_t ext_img_idx;
      auto img_it = image_id_to_idx_.find(track_el.image_id);
      if (img_it == image_id_to_idx_.end()) {
        ext_img_idx = num_images_++;
        image_id_to_idx_[track_el.image_id] = ext_img_idx;

        // Add camera if new.
        camera_t ext_cam_id = ext_image.CameraId();
        if (camera_id_to_idx_.count(ext_cam_id) == 0) {
          camera_id_to_idx_[ext_cam_id] = num_cameras_++;
          intrinsics_.resize(num_cameras_ * 3);
          principal_points_.resize(num_cameras_ * 2);
          double* ip = &intrinsics_[(num_cameras_ - 1) * 3];
          ip[0] = ext_camera.params[0];
          principal_points_[(num_cameras_ - 1) * 2 + 0] =
              ext_camera.params[1];
          principal_points_[(num_cameras_ - 1) * 2 + 1] =
              ext_camera.params[2];
          ip[1] = ext_camera.params[3];
          ip[2] = 0.0;
        }

        // Extrinsics for external image.
        extrinsics_.resize(num_images_ * 12);
        const Rigid3d cam_from_world = ext_image.CamFromWorld();
        double* ep = &extrinsics_[ext_img_idx * 12];
        const Eigen::Matrix3d rotation =
            cam_from_world.rotation().toRotationMatrix();
        const auto et = cam_from_world.translation();
        ep[0] = rotation(0, 0);
        ep[1] = rotation(0, 1);
        ep[2] = rotation(0, 2);
        ep[3] = et.x();
        ep[4] = rotation(1, 0);
        ep[5] = rotation(1, 1);
        ep[6] = rotation(1, 2);
        ep[7] = et.y();
        ep[8] = rotation(2, 0);
        ep[9] = rotation(2, 1);
        ep[10] = rotation(2, 2);
        ep[11] = et.z();

        constant_pose_mask_.push_back(1);  // External images are frozen.
        image_camera_idx_.push_back(
            static_cast<int>(camera_id_to_idx_.at(ext_cam_id)));
        image_camera_ids_.push_back(ext_cam_id);
        image_image_ids_.push_back(track_el.image_id);
        image_frame_ids_.push_back(ext_image.FrameId());
      } else {
        ext_img_idx = img_it->second;
      }

      const double cx = ext_camera.params[1];
      const double cy = ext_camera.params[2];
      const auto& ext_pt2D = ext_image.Point2D(track_el.point2D_idx);
      points_2d_.push_back(ext_pt2D.xy.x() - cx);
      points_2d_.push_back(ext_pt2D.xy.y() - cy);
      image_indices_.push_back(static_cast<int>(ext_img_idx));
      camera_obs_indices_.push_back(image_camera_idx_[ext_img_idx]);
      point_indices_.push_back(static_cast<int>(pt_it->second));
      ++num_observations_;
    }
  };
  for (const auto point3D_id : config_.VariablePoints()) {
    add_external_obs(point3D_id);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    add_external_obs(point3D_id);
  }

  // Diagnostic: connected components of the view graph induced by
  // shared 3D points. Two images are in the same component if they share
  // at least one 3D point (i.e., a track passes through both). Multiple
  // components mean the BA problem decomposes into independent sub-graphs
  // joined only by gauge constraints. Multi-session captures (e.g.,
  // mihama, kushimoto) typically present here as 2-N components with
  // weak inter-session links.
  if (num_images_ > 0 && num_observations_ > 0) {
    std::vector<int> uf_parent(num_images_);
    for (size_t i = 0; i < num_images_; ++i) uf_parent[i] = static_cast<int>(i);
    auto uf_find = [&](int x) {
      while (uf_parent[x] != x) {
        uf_parent[x] = uf_parent[uf_parent[x]];
        x = uf_parent[x];
      }
      return x;
    };
    auto uf_unite = [&](int a, int b) {
      const int ra = uf_find(a);
      const int rb = uf_find(b);
      if (ra != rb) uf_parent[ra] = rb;
    };
    // Group observations by point index, then union all images touching
    // each point. O(num_observations_) total work across both passes.
    std::unordered_map<int, int> first_img_for_point;
    first_img_for_point.reserve(num_points_);
    for (size_t k = 0; k < num_observations_; ++k) {
      const int pt = point_indices_[k];
      const int img = image_indices_[k];
      auto it = first_img_for_point.find(pt);
      if (it == first_img_for_point.end()) {
        first_img_for_point.emplace(pt, img);
      } else {
        uf_unite(it->second, img);
      }
    }
    std::unordered_map<int, int> root_size;
    root_size.reserve(num_images_);
    for (size_t i = 0; i < num_images_; ++i) {
      ++root_size[uf_find(static_cast<int>(i))];
    }
    std::vector<int> sizes;
    sizes.reserve(root_size.size());
    for (const auto& [_, c] : root_size) sizes.push_back(c);
    std::sort(sizes.begin(), sizes.end(), std::greater<int>());
    std::ostringstream ss;
    ss << "[BAE view graph] components=" << sizes.size()
       << " sizes=[";
    const size_t n_show = std::min<size_t>(sizes.size(), 10);
    for (size_t i = 0; i < n_show; ++i) {
      if (i > 0) ss << ",";
      ss << sizes[i];
    }
    if (sizes.size() > n_show) ss << ",...";
    ss << "] total_images=" << num_images_;
    if (sizes.size() > 1) {
      const double largest_frac =
          static_cast<double>(sizes[0]) / static_cast<double>(num_images_);
      ss << " largest_component_frac=" << std::fixed << std::setprecision(3)
         << largest_frac;
    }
    LOG(INFO) << ss.str();
  }

  SelectGaugeConstraints();

  LOG(INFO) << "BAE extraction: " << num_images_ << " images, "
            << num_cameras_ << " cameras, "
            << num_points_ << " points, " << num_observations_
            << " observations";
}

////////////////////////////////////////////////////////////////////////////////
// Factory
////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<BundleAdjuster> CreateDefaultBaeBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction) {
  return std::make_unique<BaeBundleAdjuster>(options, config, reconstruction);
}

}  // namespace colmap
