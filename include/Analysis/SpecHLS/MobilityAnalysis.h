//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_ANALYSIS_MOBILITY_ANALYSIS_H__
#define SPECHLS_ANALYSIS_MOBILITY_ANALYSIS_H__

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <circt/Scheduling/Problems.h>
#include <llvm/Support/LogicalResult.h>

namespace spechls {

struct MobilityAnalysis {
  llvm::DenseMap<spechls::GammaOp, int> mobilities;

  MobilityAnalysis(spechls::TaskOp task, double targetClock);
};

} // namespace spechls

#endif // SPECHLS_ANALYSIS_MOBILITY_ANALYSIS_H__