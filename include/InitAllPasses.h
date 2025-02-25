//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_INIT_ALL_PASSES_H
#define SPECHLS_INCLUDED_INIT_ALL_PASSES_H

#include "Dialect/Schedule/ScheduleDialect.h"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Transforms/Passes.h"
#include "circt/Transforms/Passes.h"

namespace SpecHLS {

inline void registerAllPasses() {
  static bool initOnce = []() {
    registerSpecHLSToCombPass();
    registerSpecHLSLUTToComb();
    registerSpecHLSToSeq();
    registerMergeGammasPass();
    registerMergeLookUpTablesPass();
    registerFactorGammaInputsPass();
    registerEliminateRedundantGammaInputsPass();
    registerGroupControlNodePass();
    registerInlineModules();
    registerYosysOptimizerPass();
    registerMobilityPass();
    registerSchedulePass();
    registerConfigurationExcluderPass();
    registerExportVitisHLS();
    registerUnrollInstrPass();
    registerLongestPathPass();
    return true;
  }();
  (void)initOnce;
}

} // namespace SpecHLS

#endif // SPECHLS_INCLUDED_INIT_ALL_PASSES_H
