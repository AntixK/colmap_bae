#pragma once

#include "colmap/estimators/bundle_adjustment.h"

#include <string>

namespace colmap {

// BAE-specific bundle adjustment options.
struct BaeBundleAdjustmentOptions {
  // Maximum number of LM iterations.
  // 200 is a middle ground: 150 (the original cap, which bridge iter1.full
  // was iter-cap-binding after the §3.31 kernel fix) and 300 (which the
  // 6-dataset cross-benchmark showed was over-budgeting most calls and
  // costing ~3× wall time on bridge with only marginal quality gain).
  // The retri/refinement path still overrides this to 50 in
  // global_mapper.cc.
  int max_num_iterations = 200;

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
