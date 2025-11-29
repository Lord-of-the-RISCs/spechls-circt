//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD
#define SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"


def SplitDelaysPass : Pass<"split-delays", "ModuleOp"> {
    let summary = "Split delays";
    let description = [{ TODO }];

    let dependentDialects = [
      "spechls::SpecHLSDialect",
      "::mlir::pdl::PDLDialect",
      "::mlir::pdl_interp::PDLInterpDialect",
      ];
  }

def OutlineCorePass : Pass<"outline-core", "ModuleOp"> {
    let summary = "Outline speculative spechls.task into wcet.core";

    let description = [{
        This pass proceed speculative spechls.task (spechls.task with spechls.speculative attribute) in three steps:
        1) retrieve the speculative task
        2) create a wcet.core with the body of the task and the input of task + one input for the instructions
        3) replace the spechls.commit of the task by wcet.commit
      }];

    let dependentDialects = [
      "spechls::SpecHLSDialect",
      "wcet::WcetDialect"
    ];
  }

def LUTToMuxPass : Pass<"LUT-to-mux", "ModuleOp"> {
    let summary = "Transform LUT to mux tree";
    let description = [{
      TODO
    }];

    let dependentDialects = [
      "spechls::SpecHLSDialect",
      "::circt::comb::CombDialect",
      "::circt::hw::HWDialect",
      "::mlir::pdl::PDLDialect",
      "::mlir::pdl_interp::PDLInterpDialect"
    ];
}

def GammaBinarizationPass : Pass<"gamma-binarization", "ModuleOp"> {
  let summary = "Transform a gamma n-aire to a tree of binary gammas";

  let dependentDialects = [
    "spechls::SpecHLSDialect",
    "::circt::comb::CombDialect",
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect"
  ];
}

def SetupAnalysisPass : Pass<"setup-analysis", "ModuleOp"> {
    let dependentDialects = [
    "wcet::WcetDialect"
    ];
  }

def InsertInstrPass : Pass<"insert-instr", "ModuleOp"> {
    let summary = "";
    let description = [{ TODO }];

    let dependentDialects = [
      "::circt::hw::HWDialect",
      "spechls::SpecHLSDialect",
      "wcet::WcetDialect",
    ];
    let options = [
      ListOption<"instrs", "instrs", "uint32_t", "The basic block instructions">
    ];
  }

def InlineCorePass : Pass<"inline-core", "mlir::ModuleOp"> {
    let summary = "Inline core_instance";
    let description = [{
      Add inliner interface to circt::Comb and spechls dialects then schedule inliner pass
    }];
  }


def LongestPathPass : Pass<"longest-path", "ModuleOp"> {
    let summary = "Find the path between two dummyOp with the most penalties";
    let description = [{ TODO }];

    let dependentDialects = [
      "wcet::WcetDialect"
      ];
  }
#endif // SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_TD
