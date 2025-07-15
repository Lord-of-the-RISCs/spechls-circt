//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SCHEDULE_TRANSFORMS_PASSES_TD
#define SPECHLS_INCLUDED_DIALECT_SCHEDULE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SchedulePass : Pass<"schedule", "ModuleOp"> {
    let summary = "A CIRCT scheduler wrapper for SpecHLS.";
    let dependentDialects = ["::circt::ssp::SSPDialect"];
}

def MobilityPass : Pass<"mobility", "schedule::CircuitOp"> {
    let summary = "Compute mobility information.";
    let dependentDialects = ["::schedule::ScheduleDialect"];
}

def ConfigurationExcluderPass : Pass<"config-excluder", "schedule::CircuitOp"> {
    let summary = "Exclude speculation configurations that cannot lead to II = 1.";
    let dependentDialects = ["::schedule::ScheduleDialect"];
}

def ExtractCriticalPathPass : Pass<"extract-critical-path", "ModuleOp"> {
    let summary = "Extract a minimal critical path that constrains the II.";
    let dependentDialects = ["::circt::ssp::SSPDialect"];
}

#endif // SPECHLS_INCLUDED_DIALECT_SCHEDULE_TRANSFORMS_PASSES_TD
