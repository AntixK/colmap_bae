#pragma once

#include "colmap/estimators/bundle_adjustment.h"

#include <string>

namespace colmap {

// BAE-specific bundle adjustment options.
struct BaeBundleAdjustmentOptions {
  // Maximum number of LM iterations.
  int max_num_iterations = 100;

  // Whether to use GPU for BAE optimization.
  bool use_gpu = true;
  // GPU device index. "-1" for automatic selection.
  std::string gpu_index = "0";

  bool Check() const;
};

// BAE-specific bundle adjustment summary.
struct BaeBundleAdjustmentSummary : public BundleAdjustmentSummary {
  int num_iterations = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;

  std::string BriefReport() const override;
};

// Factory function to create a BAE bundle adjuster.
std::unique_ptr<BundleAdjuster> CreateDefaultBaeBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction);

}  // namespace colmap
