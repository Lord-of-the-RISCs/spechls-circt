//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/SpecHLS/SpeculationExplorationAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/SSP/SSPDialect.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace spechls {
#define GEN_PASS_DEF_SPECULATIONEXPLORATIONPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

class SpeculationExplorationPass : public spechls::impl::SpeculationExplorationPassBase<SpeculationExplorationPass> {
public:
  using SpeculationExplorationPassBase::SpeculationExplorationPassBase;
  void runOnOperation() override {
    auto taskSym = targetTask.getValue();
    spechls::TaskOp task;
    getOperation().walk([&](spechls::TaskOp t) {
      if (t.getSymName() == taskSym) {
        task = t;
      }
    });
    if (!task) {
      llvm::errs() << "No task found.\n";
      return signalPassFailure();
    }
    spechls::SpeculationExplorationAnalysis analysis(task, targetClock.getValue(), probabilityThreshold.getValue(),
                                                     traceFileName.getValue());

    if (analysis.configuration.empty()) {
      return signalPassFailure();
    }
    task.getBodyBlock()->walk([&](spechls::GammaOp gamma) {
      if (gamma->hasAttrOfType<mlir::IntegerAttr>("spechls.profilingId")) {
        unsigned id = gamma->getAttrOfType<mlir::IntegerAttr>("spechls.profilingId").getInt();
        int speculation = analysis.configuration[analysis.pidToEid[id]] + 1;
        if (speculation != 0) {
          gamma->setAttr("spechls.speculation",
                         mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 32), speculation));
        }
      }
    });
  }
};

} // namespace
