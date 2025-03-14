//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/IR/ScheduleOps.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace schedule {
#define GEN_PASS_DEF_CONFIGURATIONEXCLUDERPASS
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace schedule

namespace {

class ConfigurationExcluderPass : public schedule::impl::ConfigurationExcluderPassBase<ConfigurationExcluderPass> {
public:
  using ConfigurationExcluderPassBase::ConfigurationExcluderPassBase;
  void runOnOperation() override;
};

} // namespace

// TODO: Move attributes into the operation definition.
void ConfigurationExcluderPass::runOnOperation() {
  auto circuitOp = getOperation();
  double targetClock = circuitOp.getTargetClock().convertToDouble();
  auto &body = circuitOp.getBody();

  int64_t sumDistances = 0;
  SmallVector<Operation *> gammas;
  StringRef gammaAttrName = "SpecHLS.gamma";
  StringRef muAttrName = "SpecHLS.mu";
  StringRef allowUnitIIAttrName = "SpecHLS.allowUnitII";
  StringRef distanceArrayAttrName = "distances";

  for (auto &&op : body.getOps()) {
    if (op.hasAttr(gammaAttrName))
      gammas.push_back(&op);
    for (auto &&attr : op.getAttrOfType<ArrayAttr>(distanceArrayAttrName))
      sumDistances += cast<IntegerAttr>(attr).getInt();
  }
  int64_t iterationCount = 2 * (sumDistances + 1);

  DenseMap<Operation *, SmallVector<int64_t>> startTimes;
  DenseMap<Operation *, SmallVector<double>> startTimesInCycles;

  for (auto &&op : body.getOps()) {
    startTimes[&op] = SmallVector<int64_t>(iterationCount);
    startTimesInCycles[&op] = SmallVector<double>(iterationCount);
  }

  for (int64_t iteration = 0; iteration < iterationCount; ++iteration) {
    for (auto &&op : body.getOps()) {
      auto distanceArray = op.getAttrOfType<ArrayAttr>(distanceArrayAttrName).getValue();

      bool isGamma = op.hasAttr(gammaAttrName);
      bool isMu = op.hasAttr(muAttrName);
      int64_t nextCycle = isGamma ? std::numeric_limits<int64_t>::max() : 0;
      double nextTimeInCycles = 0.0;

      // Compute the unrolled schedule.
      for (size_t predIndex = 0; predIndex < op.getNumOperands(); ++predIndex) {
        int64_t distance = cast<IntegerAttr>(distanceArray[predIndex]).getInt();
        if (iteration - distance < 0) {
          nextCycle = 0;
          nextTimeInCycles = 0.0;
        } else {
          int64_t predEndCycle = 0;
          double predEndTimeInCycles = 0.0;

          Operation *pred = op.getOperand(predIndex).getDefiningOp();
          int64_t predLatency = pred->getAttrOfType<IntegerAttr>("latency").getInt();
          double predInDelay = pred->getAttrOfType<FloatAttr>("inDelay").getValueAsDouble();
          double predOutDelay = pred->getAttrOfType<FloatAttr>("outDelay").getValueAsDouble();

          size_t offset = iteration - distance;
          if (offset < startTimes[pred].size()) {
            if (predLatency > 0) {
              predEndCycle = startTimes[pred][offset] + predLatency;
              predEndTimeInCycles = predOutDelay;
            } else if (startTimesInCycles[pred][offset] + predInDelay + predOutDelay > targetClock) {
              predEndCycle = startTimes[pred][offset] + 1;
              predEndTimeInCycles = predOutDelay;
            } else {
              predEndCycle = startTimes[pred][offset];
              predEndTimeInCycles = startTimesInCycles[pred][offset] + predOutDelay;
            }

            if ((isGamma && nextCycle > predEndCycle) || (predEndCycle > nextCycle)) {
              nextCycle = predEndCycle;
              nextTimeInCycles = predEndTimeInCycles;
            } else if (predEndCycle == nextCycle) {
              nextTimeInCycles = std::min(nextTimeInCycles, predEndTimeInCycles);
            }
          }
        }
      }

      if (isMu && (iteration > sumDistances + 1) && (nextCycle - startTimes[&op][iteration - 1] > 1)) {
        circuitOp->setAttr(allowUnitIIAttrName, BoolAttr::get(circuitOp.getContext(), false));
        return;
      }

      startTimes[&op].push_back(nextCycle);
      startTimesInCycles[&op].push_back(nextTimeInCycles);
    }
  }

  circuitOp->setAttr(allowUnitIIAttrName, BoolAttr::get(circuitOp.getContext(), true));
}
