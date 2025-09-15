//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_CONVERSION_SCHEDULE_PASSES_TD
#define SPECHLS_CONVERSION_SCHEDULE_PASSES_TD

include "mlir/Pass/PassBase.td"

def ScheduleToSSPPass : Pass<"schedule-to-ssp", "schedule::CircuitOp"> {
  let summary = "Convert Schedule dialect ops to SSP ops";
  let dependentDialects = [
    "::schedule::ScheduleDialect",
    "::circt::ssp::SSPDialect",
  ];
}

#endif // SPECHLS_CONVERSION_SCHEDULE_PASSES_TD
