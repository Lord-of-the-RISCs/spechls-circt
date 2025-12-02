//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_ANALYSIS_SPECHLS_SPECULATION_EXPLORATION_ANALYSIS_H__
#define SPECHLS_ANALYSIS_SPECHLS_SPECULATION_EXPLORATION_ANALYSIS_H__

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/LogicalResult.h>

namespace spechls {

struct SpeculationExplorationAnalysis {
  llvm::SmallVector<int> configuration;
  llvm::SmallVector<int> pidToEid, eidToPid;
  double proba;
  SpeculationExplorationAnalysis(spechls::TaskOp task, double targetClock, double probabilityThreshold,
                                 std::string traceFileName);
};

} // namespace spechls

#endif // SPECHLS_ANALYSIS_SPECHLS_SPECULATION_EXPLORATION_ANALYSIS_H__