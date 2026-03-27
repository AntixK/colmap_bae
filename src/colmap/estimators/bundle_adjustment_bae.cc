#include "colmap/estimators/bundle_adjustment_bae.h"

#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>

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
      // Import the BAE solver module.
      // pycolmap mode: import as package submodule (pycolmap._core available).
      // CLI mode: pycolmap.__init__ fails (no _core), load by file path.
      py::module_ bae_solver;
      try {
        bae_solver = py::module_::import("pycolmap.bae_solver");
      } catch (py::error_already_set&) {
#ifdef BAE_SOLVER_MODULE_DIR
        py::module_ importlib_util =
            py::module_::import("importlib.util");
        const std::string solver_path =
            std::string(BAE_SOLVER_MODULE_DIR) +
            "/pycolmap/bae_solver.py";
        auto spec = importlib_util.attr("spec_from_file_location")(
            "bae_solver", solver_path);
        THROW_CHECK(spec.ptr() != nullptr)
            << "Cannot find BAE solver module at " << solver_path;
        bae_solver = importlib_util.attr("module_from_spec")(spec);
        spec.attr("loader").attr("exec_module")(bae_solver);
#else
        throw;
#endif
      }

      // Wrap C++ vectors as numpy arrays (zero-copy views into member data).
      const auto si = [](size_t n) { return static_cast<py::ssize_t>(n); };

      py::array_t<double> cam_arr({si(num_images_), si(10)},
                                  camera_params_.data());
      py::array_t<double> pts3d_arr({si(num_points_), si(3)},
                                    points_3d_.data());
      py::array_t<double> pts2d_arr({si(num_observations_), si(2)},
                                    points_2d_.data());
      py::array_t<int> cam_idx_arr({si(num_observations_)},
                                   camera_indices_.data());
      py::array_t<int> pt_idx_arr({si(num_observations_)},
                                  point_indices_.data());
      py::array_t<uint8_t> const_cam_arr({si(num_images_)},
                                         constant_camera_mask_.data());
      py::array_t<uint8_t> const_pt_arr({si(num_points_)},
                                        constant_point_mask_.data());

      // Build options dict from BaeBundleAdjustmentOptions + refine_* flags.
      py::dict options_dict;
      if (options_.bae) {
        options_dict["max_num_iterations"] =
            options_.bae->max_num_iterations;
        options_dict["use_gpu"] = options_.bae->use_gpu;
        options_dict["gpu_index"] = options_.bae->gpu_index;
      }
      options_dict["refine_focal_length"] = options_.refine_focal_length;
      options_dict["refine_extra_params"] = options_.refine_extra_params;

      // Call the Python BAE solver.
      py::dict result = bae_solver.attr("solve")(cam_arr,
                                                 pts3d_arr,
                                                 pts2d_arr,
                                                 cam_idx_arr,
                                                 pt_idx_arr,
                                                 const_cam_arr,
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

      // Copy optimized parameters back into member arrays for writeback.
      auto opt_cam = result["camera_params"].cast<py::array_t<double>>();
      auto opt_pts = result["points_3d"].cast<py::array_t<double>>();
      std::memcpy(camera_params_.data(),
                  opt_cam.data(),
                  camera_params_.size() * sizeof(double));
      std::memcpy(points_3d_.data(),
                  opt_pts.data(),
                  points_3d_.size() * sizeof(double));
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
  void SetupProblem();

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
  size_t num_points_ = 0;
  size_t num_observations_ = 0;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  std::unordered_map<point3D_t, size_t> point3D_id_to_idx_;

  // (num_images_ * 10): [tx,ty,tz, qx,qy,qz,qw, f, k1, k2] per image.
  std::vector<double> camera_params_;
  // (num_points_ * 3): [x, y, z] per point.
  std::vector<double> points_3d_;
  // (num_observations_ * 2): centered 2D observations.
  std::vector<double> points_2d_;
  // (num_observations_): camera (image) index per observation.
  std::vector<int> camera_indices_;
  // (num_observations_): point index per observation.
  std::vector<int> point_indices_;
  // (num_images_): 1 if camera pose is constant.
  std::vector<uint8_t> constant_camera_mask_;
  // (num_points_): 1 if 3D point is constant.
  std::vector<uint8_t> constant_point_mask_;
  // (num_images_): camera_id for each image (for intrinsics writeback).
  std::vector<camera_t> image_camera_ids_;
};

void BaeBundleAdjuster::SetupProblem() {
  // Validate: all cameras use SimpleRadial or Radial
  // Currently BAE does not support multiRig camera setups
  for (const image_t image_id : config_.Images()) {
    const auto& image = reconstruction_.Image(image_id);
    const auto& camera = reconstruction_.Camera(image.CameraId());
    THROW_CHECK(camera.model_id == CameraModelId::kSimpleRadial ||
                camera.model_id == CameraModelId::kRadial)
        << "BAE only supports SimpleRadial and Radial camera models";
    THROW_CHECK(image.IsRefInFrame())
        << "BAE does not support multi-sensor rigs";
  }

  // Build image_id -> contiguous index map for config images.
  for (const image_t image_id : config_.Images()) {
    image_id_to_idx_[image_id] = num_images_++;
  }

  // Collect point3D_ids observed in config images
  // i.e. for each 2D point in the images, if it has a corresponding 3D point,
  // then add it 3D point for BA optimization.
  // Skip ignored points and points with track length < min_track_length.
  // These points will not be included in the BA optimization.
  for (const image_t image_id : config_.Images()) {
    const auto& image = reconstruction_.Image(image_id);
    for (const auto& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) continue;
      if (config_.IsIgnoredPoint(point2D.point3D_id)) continue;
      const auto& point3D = reconstruction_.Point3D(point2D.point3D_id);
      if (options_.min_track_length > 0 &&
          static_cast<int>(point3D.track.Length()) <
              options_.min_track_length) {
        continue;
      }
      if (point3D_id_to_idx_.count(point2D.point3D_id) == 0) {
        point3D_id_to_idx_[point2D.point3D_id] = num_points_++;
      }
    }
  }

  // Append the extra points from config's VariablePoints and ConstantPoints that are not
  // already included above from the config images.
  // This is crucial when using incremental_mapper
  // For global_mapper, they may be empty
  auto collect_extra_point = [&](const point3D_t point3D_id) {
    if (config_.IsIgnoredPoint(point3D_id)) return;
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
  // Scan through the variable and constant points in the config
  // and add them
  for (const auto point3D_id : config_.VariablePoints()) {
    collect_extra_point(point3D_id);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    collect_extra_point(point3D_id);
  }

  // Extract camera_params (pose + intrinsics) for config images.
  // Initialize placeholders
  camera_params_.resize(num_images_ * 10);
  constant_camera_mask_.resize(num_images_, 0);
  image_camera_ids_.resize(num_images_);

  for (const image_t image_id : config_.Images()) {
    const size_t idx = image_id_to_idx_.at(image_id);
    const auto& image = reconstruction_.Image(image_id);
    const auto& camera = reconstruction_.Camera(image.CameraId());
    const Rigid3d& rig_from_world =
        reconstruction_.Frame(image.FrameId()).RigFromWorld();

    double* p = &camera_params_[idx * 10];
    // Reorder: COLMAP [qx,qy,qz,qw,tx,ty,tz] -> BAE [tx,ty,tz,qx,qy,qz,qw].
    const auto q = rig_from_world.rotation();
    const auto t = rig_from_world.translation();
    p[0] = t.x();
    p[1] = t.y();
    p[2] = t.z();
    p[3] = q.x();
    p[4] = q.y();
    p[5] = q.z();
    p[6] = q.w();
    // Intrinsics: f, k1, k2.
    p[7] = camera.params[0];
    if (camera.model_id == CameraModelId::kSimpleRadial) {
      p[8] = camera.params[3]; // k1
      p[9] = 0.0; // For simple radial, k2 = 0
    } else {  // Radial
      p[8] = camera.params[3];
      p[9] = camera.params[4];
    }

    constant_camera_mask_[idx] =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());
    image_camera_ids_[idx] = image.CameraId();
  }

  // Extract points_3d and constant_point_mask.
  points_3d_.resize(num_points_ * 3);
  constant_point_mask_.resize(num_points_, 0);
  // Gather the 3D points from the point ids collected above
  for (const auto& [point3D_id, idx] : point3D_id_to_idx_) {
    const auto& point3D = reconstruction_.Point3D(point3D_id);
    points_3d_[idx * 3 + 0] = point3D.xyz.x();
    points_3d_[idx * 3 + 1] = point3D.xyz.y();
    points_3d_[idx * 3 + 2] = point3D.xyz.z();
    constant_point_mask_[idx] =
        !options_.refine_points3D || config_.HasConstantPoint(point3D_id);
  }

  // Extract observations (2D points and and their respective indices) from config images.
  for (const image_t image_id : config_.Images()) {
    const size_t cam_idx = image_id_to_idx_.at(image_id);
    const auto& image = reconstruction_.Image(image_id);
    const auto& camera = reconstruction_.Camera(image.CameraId());
    const double cx = camera.params[1];
    const double cy = camera.params[2];

    for (const auto& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) continue;
      if (config_.IsIgnoredPoint(point2D.point3D_id)) continue;
      auto it = point3D_id_to_idx_.find(point2D.point3D_id);
      if (it == point3D_id_to_idx_.end()) continue;
      // Center around principal point (COLMAP convention: obs - cx).
      points_2d_.push_back(point2D.xy.x() - cx);
      points_2d_.push_back(point2D.xy.y() - cy);
      camera_indices_.push_back(static_cast<int>(cam_idx));
      point_indices_.push_back(static_cast<int>(it->second));
      ++num_observations_;
    }
  }

  // C4: handle VariablePoints/ConstantPoints extra residuals.
  // This is mainly to handle cases for local BA. 
  // Note that local BA may include only a set of neighboring images and their corresponding 2D points.
  // But their corresponding 3D points may also be observed by images in the neighborhood set. 
  // So, we still add these points to the optimization set but we do not optimize their
  // camera poses because they are not in the neighborhood set. So, we freeze their camera poses.
  // Add observations from external images with frozen poses.
  auto add_external_obs = [&](const point3D_t point3D_id) {
    auto pt_it = point3D_id_to_idx_.find(point3D_id);
    if (pt_it == point3D_id_to_idx_.end()) return;
    const auto& point3D = reconstruction_.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (config_.HasImage(track_el.image_id)) continue;
      const auto& ext_image = reconstruction_.Image(track_el.image_id);
      THROW_CHECK(ext_image.IsRefInFrame())
          << "BAE does not support multi-sensor rigs (external image)";
      const auto& ext_camera =
          reconstruction_.Camera(ext_image.CameraId());
      if (ext_camera.model_id != CameraModelId::kSimpleRadial &&
          ext_camera.model_id != CameraModelId::kRadial) {
        continue;
      }
      // Add external image as frozen camera if not yet added.
      size_t ext_cam_idx;
      auto img_it = image_id_to_idx_.find(track_el.image_id);
      if (img_it == image_id_to_idx_.end()) {
        ext_cam_idx = num_images_++;
        image_id_to_idx_[track_el.image_id] = ext_cam_idx;
        camera_params_.resize(num_images_ * 10);
        constant_camera_mask_.push_back(1);
        image_camera_ids_.push_back(ext_image.CameraId());
        const Rigid3d cam_from_world = ext_image.CamFromWorld();
        double* ep = &camera_params_[ext_cam_idx * 10];
        const auto eq = cam_from_world.rotation();
        const auto et = cam_from_world.translation();
        ep[0] = et.x();
        ep[1] = et.y();
        ep[2] = et.z();
        ep[3] = eq.x();
        ep[4] = eq.y();
        ep[5] = eq.z();
        ep[6] = eq.w();
        ep[7] = ext_camera.params[0];
        if (ext_camera.model_id == CameraModelId::kSimpleRadial) {
          ep[8] = ext_camera.params[3];
          ep[9] = 0.0;
        } else {
          ep[8] = ext_camera.params[3];
          ep[9] = ext_camera.params[4];
        }
      } else {
        ext_cam_idx = img_it->second;
      }
      const double cx = ext_camera.params[1];
      const double cy = ext_camera.params[2];
      const auto& ext_pt2D = ext_image.Point2D(track_el.point2D_idx);
      points_2d_.push_back(ext_pt2D.xy.x() - cx);
      points_2d_.push_back(ext_pt2D.xy.y() - cy);
      camera_indices_.push_back(static_cast<int>(ext_cam_idx));
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

  LOG(INFO) << "BAE extraction: " << num_images_ << " images, "
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
