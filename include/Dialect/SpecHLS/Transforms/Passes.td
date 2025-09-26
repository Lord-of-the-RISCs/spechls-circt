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

def OptimizeControlLogicPass: Pass<"optimize-control-logic", "ModuleOp"> {
  let summary = "Optimize control logic of gammas.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect",
    "::circt::comb::CombDialect", "::circt::sv::SVDialect",
    "::circt::llhd::LLHDDialect", "::mlir::cf::ControlFlowDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::math::MathDialect",
    "::circt::sim::SimDialect",
    "::circt::verif::VerifDialect",
    "::circt::moore::MooreDialect",
    "::mlir::func::FuncDialect",
    "::circt::ltl::LTLDialect",
    "::circt::debug::DebugDialect"];
}

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_PASSES_TD
