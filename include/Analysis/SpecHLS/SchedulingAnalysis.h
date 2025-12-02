//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_ANALYSIS_SPECHLS_SCHEDULING_ANALYSIS_H__
#define SPECHLS_ANALYSIS_SPECHLS_SCHEDULING_ANALYSIS_H__

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <circt/Scheduling/Problems.h>
#include <llvm/Support/LogicalResult.h>

namespace spechls {

struct SchedulingAnalysis {
  llvm::DenseMap<mlir::Operation *, unsigned> startTime;
  llvm::DenseMap<mlir::Operation *, double> startTimeInCycle;
  unsigned ii;

  static circt::scheduling::ChainingCyclicProblem constructProblem(spechls::TaskOp task, double targetClock,
                                                                   llvm::SmallVector<int> configuration);

  SchedulingAnalysis(spechls::TaskOp task, double targetClock);
  SchedulingAnalysis(spechls::TaskOp task, double targetClock, llvm::SmallVector<int> configuration);
};

}; // namespace spechls

#endif // SPECHLS_ANALYSIS_SPECHLS_SCHEDULING_ANALYSIS_H__