#include "colmap/estimators/bundle_adjustment_bae.h"

#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"

#include <unordered_map>

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
    LOG(WARNING) << "BAE bundle adjustment solver not yet implemented";
    return summary;
  }

 private:
  void SetupProblem();

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
      // C2: center around principal point with BAE sign convention.
      points_2d_.push_back(cx - point2D.xy.x());
      points_2d_.push_back(cy - point2D.xy.y());
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
      points_2d_.push_back(cx - ext_pt2D.xy.x());
      points_2d_.push_back(cy - ext_pt2D.xy.y());
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
