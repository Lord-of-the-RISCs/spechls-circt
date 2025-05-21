//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/Passes.h"
#include "Dialect/Schedule/IR/ScheduleOps.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep
#include "circt/Dialect/SSP/SSPOps.h"

#include <circt/Dialect/SSP/SSPDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

namespace schedule {
#define GEN_PASS_DEF_CHECKSCHEDULEPASS
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace schedule

namespace {

class CheckSchedulePass : public schedule::impl::CheckSchedulePassBase<CheckSchedulePass> {
public:
  using CheckSchedulePassBase::CheckSchedulePassBase;
  void runOnOperation() override;
};

} // namespace

void CheckSchedulePass::runOnOperation() {
  auto mod = getOperation();
  OpPassManager pm("builtin.module");
  pm.addNestedPass<schedule::CircuitOp>(schedule::createConfigurationExcluderPass());

  if (failed(runPipeline(pm, mod))) {
    llvm::errs() << "Failed to run ConfigurationExcluder\n";
    return signalPassFailure();
  }

  bool allow = false;
  for (auto &&op : mod) {
    if (auto circuitOp = dyn_cast<schedule::CircuitOp>(op)) {
      allow = circuitOp->getAttrOfType<BoolAttr>("spechls.allow_unit_ii").getValue();
      break;
    }
  }

  pm.clear();
  pm.addNestedPass<schedule::CircuitOp>(schedule::createScheduleToSSPPass());
  pm.addPass(schedule::createSchedulePass());
  if (failed(runPipeline(pm, mod))) {
    llvm::errs() << "Failed to run Scheduler\n";
    return signalPassFailure();
  }

  int64_t ii = -1;
  for (auto &&op : mod) {
    if (auto instanceOp = dyn_cast<circt::ssp::InstanceOp>(op)) {
      ii = instanceOp->getAttrOfType<IntegerAttr>("spechls.ii").getInt();
      break;
    }
  }

  if (!(ii != 1 || allow))
    llvm::outs() << "Inconsistent schedule\n";
}
