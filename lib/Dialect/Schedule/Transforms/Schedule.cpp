//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep

#include <circt/Dialect/SSP/SSPAttributes.h>
#include <circt/Dialect/SSP/SSPOps.h>
#include <circt/Dialect/SSP/SSPPasses.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Pass/PassManager.h>

using namespace mlir;

namespace schedule {
#define GEN_PASS_DEF_SCHEDULEPASS
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace schedule

namespace {

class SchedulePass : public schedule::impl::SchedulePassBase<SchedulePass> {
public:
  using SchedulePassBase::SchedulePassBase;
  void runOnOperation() override;
};

} // namespace

void SchedulePass::runOnOperation() {
  auto moduleOp = getOperation();

  double period = 0.0;
  for (auto &&instance : moduleOp.getOps<circt::ssp::InstanceOp>()) {
    if (auto periodAttr = instance->getAttrOfType<FloatAttr>("spechls.period"))
      period = periodAttr.getValueAsDouble();
    else
      return signalPassFailure();
  }

  OpPassManager dynamicPM("builtin.module");
  auto pass = circt::ssp::createSchedulePass();
  if (failed(pass->initializeOptions("scheduler=lp options=cycle-time=" + std::to_string(period), [](const Twine &msg) {
        llvm::errs() << msg << '\n';
        return failure();
      })))
    return signalPassFailure();
  dynamicPM.addPass(std::move(pass));
  if (failed(runPipeline(dynamicPM, moduleOp)))
    return signalPassFailure();

  // FIXME: The following code only exists to avoid having to implement an ArrayRef wrapper in the Java implementation
  // of SpecHLS, which is not a good reason.
  for (auto &&instance : moduleOp.getOps<circt::ssp::InstanceOp>()) {
    if (auto propertiesAttr = instance->getAttrOfType<ArrayAttr>("sspProperties")) {
      for (auto &&property : propertiesAttr) {
        if (auto iiAttr = dyn_cast<circt::ssp::InitiationIntervalAttr>(property)) {
          instance->setAttr("spechls.ii",
                            IntegerAttr::get(IntegerType::get(moduleOp.getContext(), 32), iiAttr.getValue()));
        }
      }
    }
    for (auto &&sspOp : instance.getDependenceGraph()) {
      if (auto propertiesAttr = sspOp.getAttrOfType<ArrayAttr>("sspProperties")) {
        for (auto &&property : propertiesAttr) {
          if (auto z = dyn_cast<circt::ssp::StartTimeInCycleAttr>(property))
            sspOp.setAttr("spechls.z", z.getValue());
          if (auto t = dyn_cast<circt::ssp::StartTimeAttr>(property))
            sspOp.setAttr("spechls.t", IntegerAttr::get(IntegerType::get(moduleOp.getContext(), 32), t.getValue()));
        }
      }
    }
  }
}
