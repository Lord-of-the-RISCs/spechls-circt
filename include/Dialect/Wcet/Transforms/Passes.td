//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD
#define SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def UnrollInstrPass : Pass<"unroll-instr", "mlir::ModuleOp"> {
    let summary = "Unroll the HW module with the instructions";
    let description = [{ TODO }];

    //let constructor = "wcet::createUnrollInstrPass()";
    let dependentDialects = [
      "circt::hw::HWDialect",
      "SpecHLS::SpecHLSDialect",
      "SpecHLS::WcetDialect",
      "mlir::arith::ArithDialect"
    ];

    let options = [
      ListOption<"instrs", "instrs", "uint32_t", "The basic block instructions">
    ];
  }

def LongestPathPass : Pass<"longest-path", "mlir::ModuleOp"> {
    let summary = "Find the longest path";
    let description = [{ TODO }];

    //let constructor = "wcet::createLongestPathPass()";
    let dependentDialects = [
      "circt::hw::HWDialect",
      "SpecHLS::SpecHLSDialect",
      "SpecHLS::WcetDialect"
      ];
  }

#endif // SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD
