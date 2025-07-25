//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <mlir/Support/LLVM.h>

#include "Dialect/Schedule/IR/ScheduleOps.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep

#include <queue>

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
  StringRef gammaAttrName = "spechls.gamma";
  StringRef muAttrName = "spechls.mu";
  StringRef allowUnitIIAttrName = "spechls.allow_unit_ii";
  StringRef distanceArrayAttrName = "distances";

  for (auto &&op : body.getOps()) {
    for (auto &&attr : op.getAttrOfType<ArrayAttr>(distanceArrayAttrName))
      sumDistances += cast<IntegerAttr>(attr).getInt();
  }
  int64_t iterationCount = 2 * (sumDistances + 1);

  DenseMap<Operation *, SmallVector<int64_t>> startTimes;
  DenseMap<Operation *, SmallVector<double>> startTimesInCycle;

  for (auto &&op : body.getOps()) {
    startTimes.try_emplace(&op, SmallVector<int64_t>());
    startTimesInCycle.try_emplace(&op, SmallVector<double>());
    // The size of these vectors is used in the following code, so we only reserve and don't allocate.
    startTimes[&op].reserve(iterationCount);
    startTimesInCycle[&op].reserve(iterationCount);
  }

  for (int64_t iteration = 0; iteration < iterationCount; ++iteration) {
    std::queue<Operation *> workList;
    for (auto &&op : body.getOps()) {
      workList.push(&op);
    }

    while (!workList.empty()) {
      bool skip = false;
      Operation *op = workList.front();
      workList.pop();
      auto distanceArray = op->getAttrOfType<ArrayAttr>(distanceArrayAttrName).getValue();

      bool isGamma = op->hasAttr(gammaAttrName);
      bool isMu = op->hasAttr(muAttrName);
      int64_t nextCycle = isGamma ? std::numeric_limits<int64_t>::max() : 0;
      double nextTimeInCycle = 0.0;

      // Compute the unrolled schedule.
      for (size_t predIndex = 0; predIndex < op->getNumOperands(); ++predIndex) {
        int64_t distance = cast<IntegerAttr>(distanceArray[predIndex]).getInt();
        if (iteration - distance < 0) {
          if (isGamma) {
            nextCycle = 0;
            nextTimeInCycle = 0.0;
          }
        } else {
          int64_t predEndCycle = 0;
          double predEndTimeInCycle = 0.0;

          Operation *pred = op->getOperand(predIndex).getDefiningOp();
          int64_t predLatency = pred->getAttrOfType<IntegerAttr>("latency").getInt();
          double predInDelay = pred->getAttrOfType<FloatAttr>("inDelay").getValueAsDouble();
          double predOutDelay = pred->getAttrOfType<FloatAttr>("outDelay").getValueAsDouble();

          size_t offset = iteration - distance;
          if (offset < startTimes[pred].size()) {
            if (predLatency > 0) {
              predEndCycle = startTimes[pred][offset] + predLatency;
              if (startTimesInCycle[pred][offset] + predInDelay > targetClock)
                predEndCycle += 1;
              predEndTimeInCycle = predOutDelay;
            } else if (startTimesInCycle[pred][offset] + predInDelay > targetClock) {
              assert(predInDelay == predOutDelay);
              predEndCycle = startTimes[pred][offset] + 1;
              predEndTimeInCycle = predOutDelay;
            } else {
              predEndCycle = startTimes[pred][offset];
              predEndTimeInCycle = startTimesInCycle[pred][offset] + predOutDelay;
            }

            if (isGamma) {
              if (nextCycle > predEndCycle) {
                nextCycle = predEndCycle;
                nextTimeInCycle = predEndTimeInCycle;
              } else if (predEndCycle == nextCycle) {
                nextTimeInCycle = std::min(nextTimeInCycle, predEndTimeInCycle);
              }
            } else if (predEndCycle > nextCycle) {
              nextCycle = predEndCycle;
              nextTimeInCycle = predEndTimeInCycle;
            } else if (predEndCycle == nextCycle) {
              nextTimeInCycle = std::max(nextTimeInCycle, predEndTimeInCycle);
            }
          } else {
            workList.push(op);
            skip = true;
            break;
          }
        }
      }

      // Move to the next node in the worklist.
      if (skip)
        continue;

      if (isMu && (iteration > sumDistances + 1) && (nextCycle - startTimes[op][iteration - 1] > 1)) {
        circuitOp->setAttr(allowUnitIIAttrName, BoolAttr::get(circuitOp.getContext(), false));
        return;
      }

      startTimes[op].push_back(nextCycle);
      startTimesInCycle[op].push_back(nextTimeInCycle);
    }
  }

  circuitOp->setAttr(allowUnitIIAttrName, BoolAttr::get(circuitOp.getContext(), true));
}
