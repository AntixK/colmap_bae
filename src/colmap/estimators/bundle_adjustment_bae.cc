#include "colmap/estimators/bundle_adjustment_bae.h"

#include "colmap/util/logging.h"

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
      : BundleAdjuster(options, config), reconstruction_(reconstruction) {}

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    auto summary = std::make_shared<BaeBundleAdjustmentSummary>();
    summary->termination_type = BundleAdjustmentTerminationType::FAILURE;
    LOG(WARNING) << "BAE bundle adjustment is not yet implemented";
    return summary;
  }

 private:
  Reconstruction& reconstruction_;
};

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
