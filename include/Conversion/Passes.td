//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_CONVERSION_PASSES_TD
#define SPECHLS_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"
include "Dialect/Schedule/IR/Schedule.td"

def scheduleToSSP : Pass<"convert-schedule-to-ssp", "schedule::CircuitOp"> {
  let summary = "Convert schedule dialect to SSP";
  let constructor = "spechls::createScheduleToSSP()";
  let dependentDialects = [
    "circt::ssp::SSPDialect",
    "schedule::ScheduleDialect"
  ];
}

#endif // SPECHLS_CONVERSION_PASSES_TD