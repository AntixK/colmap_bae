#pragma once

#include "colmap/estimators/bundle_adjustment.h"

#include <string>

namespace colmap {

// BAE-specific bundle adjustment options.
struct BaeBundleAdjustmentOptions {
  // Maximum number of LM iterations.
  // Bumped 150 -> 300 because the post-§3.31 (kernel-correction) bridge
  // run showed iter1.full exiting at 150/150 max_iter with cost still
  // descending (cost_drop_total=3.7%, windowed_imp=nan — budget-bound
  // not tolerance-bound). 300 gives Ceres-comparable iteration headroom
  // on the hard call without affecting easier datasets (most calls
  // exit on func_tol after ~8-20 iters anyway). The retri/refinement
  // path still overrides this to 50 in global_mapper.cc.
  int max_num_iterations = 300;

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
