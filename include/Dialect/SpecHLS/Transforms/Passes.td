//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_PASSES_TD
#define SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SimplifyGammasPass : Pass<"simplify-gammas", "ModuleOp"> {
  let summary = "Simplify gamma node patterns.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect",
                           "::mlir::pdl::PDLDialect", "::mlir::pdl_interp::PDLInterpDialect",
  ];
}

def SimplifyLUTsPass : Pass<"simplify-luts", "ModuleOp"> {
  let summary = "Simplify LUT patterns.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect",
                           "::mlir::pdl::PDLDialect", "::mlir::pdl_interp::PDLInterpDialect",
  ];
}

def FactorGammaInputsPass : Pass<"factor-gamma-inputs", "ModuleOp"> {
  let summary = "Simplify LUT patterns.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
}

def ExposeControlFlowSpeculation : Pass<"expose-cf-speculation", "ModuleOp"> {
  let summary = "Insert CF flow speculation for a given GammaConfiguration.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
}


#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_PASSES_TD
