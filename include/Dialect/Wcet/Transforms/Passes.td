//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD
#define SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def UnrollInstrPass : Pass<"unroll-instr", "spechls::KernelOp"> {
    let summary = "Unroll the spechls kernel with the given instructions";
    let description = [{ TODO }];

    let dependentDialects = [
      "circt::hw::HWDialect",
      "spechls::SpecHLSDialect",
      "wcet::WcetDialect",
    ];
    let options = [
      ListOption<"instrs", "instrs", "uint32_t", "The basic block instructions">
    ];
  }

def LongestPathPass : Pass<"longest-path", "spechls::KernelOp"> {
    let summary = "Find the longest wcet path";
    let description = [{ TODO }];

    let dependentDialects = [
      "circt::hw::HWDialect",
      "spechls::SpecHLSDialect",
      "wcet::WcetDialect"
      ];
  }

def MergeDelaysPass : Pass<"merge-delays", "spechls::KernelOp"> {
    let summary = "Merge delays";
    let description = [{ TODO }];

    let dependentDialects = [
      "circt::hw::HWDialect",
      "spechls::SpecHLSDialect",
      ];
  }

#endif // SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD
