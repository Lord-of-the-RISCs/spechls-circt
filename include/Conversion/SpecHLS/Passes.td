//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_CONVERSION_SPECHLS_PASSES_TD
#define SPECHLS_CONVERSION_SPECHLS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SpecHLSToHWPass : Pass<"spechls-to-hw", "spechls::KernelOp"> {
  let summary = "Convert SpecHLS dialect ops to HW and Comb dialect ops";
  let dependentDialects = [
    "::spechls::SpecHLSDialect",
    "::circt::comb::CombDialect",
    "::circt::hw::HWDialect",
  ];
}
def SpecHLSTaskToHWPass : Pass<"spechls-to-hw", "spechls::TaskOp"> {
  let summary = "Convert SpecHLS task op to HW and Comb dialect ops";
  let dependentDialects = [
    "::spechls::SpecHLSDialect",
    "::circt::comb::CombDialect",
    "::circt::hw::HWDialect",
  ];
}

#endif // SPECHLS_CONVERSION_PASSES_TD
