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

def RemoveIntraRAWPass : Pass<"remove-intra-raw", "ModuleOp"> {
  let summary = "Remove intra-iteration RAW dependencies";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::comb::CombDialect", "::circt::hw::HWDialect", "::mlir::pdl::PDLDialect",
      "::mlir::pdl_interp::PDLInterpDialect",];
}

def ExposeMemorySpeculationPass : Pass<"expose-memory-speculation", "ModuleOp"> {
    let summary = "";
    let dependentDialects = [
      "::spechls::SpecHLSDialect", 
      "::circt::comb::CombDialect", 
      "::circt::hw::HWDialect", 
      "::mlir::pdl::PDLDialect", 
      "::mlir::pdl_interp::PDLInterpDialect"
      ];
}

def InferLSQPass : Pass<"infer-LSQ", "ModuleOp"> {
    let summary = "";
    let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::mlir::pdl::PDLDialect", 
    "::mlir::pdl_interp::PDLInterpDialect"];
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
    "::circt::debug::DebugDialect",
    "::circt::synth::SynthDialect"];
}

def SpecHLSTopoSortPass: Pass<"spechls-toposort", "ModuleOp"> {
  let summary = "Call toposort on kernel and tasks.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
}

def SimplifyCombPass: Pass<"simplify-comb", "::ModuleOp"> {
  let summary = "Simplify unwanted pattern in comb dialect.";
  let dependentDialects = ["::circt::comb::CombDialect", "::circt::hw::HWDialect"];
}



def SpeculationExplorationPass: Pass<"speculation-exploration", "spechls::KernelOp"> {
  let summary = "Exploration configuration configuration and mark it with attributes";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect", "::circt::ssp::SSPDialect"];
  let options = [
    Option<"targetTask", "targetTask", "std::string", "", "Name of task to inline">,
    Option<"targetClock", "targetClock", "double", "10.0", "Target clock for scheduling.">,
    Option<"probabilityThreshold", "probabilityThreshold", "double", "0.0", "Minimum probability of speculation configurations">,
    Option<"traceFileName", "traceFileName", "std::string", "", "Profiling trace file.">
  ];
}

def LowerDelaysPass: Pass<"lower-spechls-delays", "spechls::KernelOp"> {
  let summary = "Lower special delays into delays and their special nodes.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
}

def AnnotateTimingPass : Pass<"annotate-timing", "spechls::KernelOp"> {
  let summary = "Add timing annotation to task operations.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
  let options = [
    Option<"task", "task", "std::string", "", "Name of task to inline">,
    Option<"clockPeriod", "clockPeriod", "double", "10.0", "Target clock for scheduling.">,
    Option<"target", "target", "std::string", "\"xc7a200tisbg484-1L\"", "Target to map timing with.">,
    Option<"targetsFile", "targetsFile", "std::string", "", "Json file with targets timing.">,
  ];
}
def AffectProfilingIDPass : Pass<"affect-profilingID", "spechls::KernelOp"> {
  let summary = "Add unique profile ID to each gamma.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
}


def SplitWithSyncPass : Pass<"split-with-sync", "spechls::KernelOp"> {
  let summary = "Split operations with a combDelay longer than clock period into fake operations to make them takes at most one cycle.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
  let options = [
    Option<"task", "task", "std::string", "", "Name of task to work with">,
    Option<"clockPeriod", "clockPeriod", "double", "10.0", "Target clock for scheduling.">
  ];
}

def LowerComplexOperationsPass: Pass<"lower-complex-operation", "spechls::KernelOp"> {
  let summary = "Lower complex operations such as division into simpler ones.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
}

def ExposeControlFlowSpeculationPass: Pass<"expose-control-flow-speculation", "spechls::KernelOp"> {
  let summary = "Add speculation mechanism into a task.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect", "::circt::ssp::SSPDialect"];
  let options = [
    Option<"targetTask", "targetTask", "std::string", "", "Name of task to inline">,
    Option<"clockPeriod", "clockPeriod", "double", "10.0", "Target clock for scheduling.">,
    Option<"target", "target", "std::string", "\"xc7a200tisbg484-1L\"", "Target to map timing with.">,
    Option<"targetsFile", "targetsFile", "std::string", "", "Json file with targets timing.">,
    Option<"probabilityThreshold", "probabilityThreshold", "double", "0.0", "Minimum probability of speculation configurations">,
    Option<"traceFileName", "traceFileName", "std::string", "", "Profiling trace file.">,
    Option<"wcetModel", "wcetModel", "bool", "false", "Stop speculation insertion at the generation of the wcet module.">
  ];
}

def LinearizeTaskPass: Pass<"linearize-task", "spechls::KernelOp"> {
  let summary = "transform the kernel so its only operations are a chain of task";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
}

def MergeTasksPass: Pass<"merge-tasks", "spechls::KernelOp"> {
  let summary = "Merge not speculative tasks.";
  let dependentDialects = ["::spechls::SpecHLSDialect", "::circt::hw::HWDialect", "::circt::comb::CombDialect"];
}

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_PASSES_TD
