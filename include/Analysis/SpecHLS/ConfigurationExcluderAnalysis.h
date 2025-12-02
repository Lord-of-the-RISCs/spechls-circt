//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_ANALYSIS_CONFIGURATION_EXCLUDER_ANALYSIS_H__
#define SPECHLS_ANALYSIS_CONFIGURATION_EXCLUDER_ANALYSIS_H__

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <circt/Scheduling/Problems.h>
#include <llvm/Support/LogicalResult.h>

namespace spechls {

struct ConfigurationExcluderAnalysis {
  bool deadEnd;

  ConfigurationExcluderAnalysis(spechls::TaskOp task, double targetClock, llvm::ArrayRef<int> configuration,
                                llvm::ArrayRef<GammaOp> gammas);
};

} // namespace spechls

#endif // SPECHLS_ANALYSIS_CONFIGURATION_EXCLUDER_ANALYSIS_H__