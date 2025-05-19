//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/IR/ScheduleOps.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
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
  DenseMap<Operation *, SmallVector<double>> startTimesInCycles;

  for (auto &&op : body.getOps()) {
    startTimes.try_emplace(&op, SmallVector<int64_t>());
    startTimesInCycles.try_emplace(&op, SmallVector<double>());
    // The size of these vectors is used in the following code, so we only reserve and don't allocate.
    startTimes[&op].reserve(iterationCount);
    startTimesInCycles[&op].reserve(iterationCount);
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
      double nextTimeInCycles = 0.0;

      // Compute the unrolled schedule.
      for (size_t predIndex = 0; predIndex < op->getNumOperands(); ++predIndex) {
        int64_t distance = cast<IntegerAttr>(distanceArray[predIndex]).getInt();
        if (iteration - distance < 0) {
          if (isGamma) {
            nextCycle = 0;
            nextTimeInCycles = 0.0;
          }
        } else {
          int64_t predEndCycle = 0;
          double predEndTimeInCycles = 0.0;

          Operation *pred = op->getOperand(predIndex).getDefiningOp();
          int64_t predLatency = pred->getAttrOfType<IntegerAttr>("latency").getInt();
          double predInDelay = pred->getAttrOfType<FloatAttr>("inDelay").getValueAsDouble();
          double predOutDelay = pred->getAttrOfType<FloatAttr>("outDelay").getValueAsDouble();

          size_t offset = iteration - distance;
          if (offset < startTimes[pred].size()) {
            if (predLatency > 0) {
              predEndCycle = startTimes[pred][offset] + predLatency;
              if (startTimesInCycles[pred][offset] + predInDelay > targetClock)
                predEndCycle += 1;
              predEndTimeInCycles = predOutDelay;
            } else if (startTimesInCycles[pred][offset] + predInDelay > targetClock) {
              assert(predInDelay == predOutDelay);
              predEndCycle = startTimes[pred][offset] + 1;
              predEndTimeInCycles = predOutDelay;
            } else {
              predEndCycle = startTimes[pred][offset];
              predEndTimeInCycles = startTimesInCycles[pred][offset] + predOutDelay;
            }

            if (isGamma) {
              if (nextCycle > predEndCycle) {
                nextCycle = predEndCycle;
                nextTimeInCycles = predEndTimeInCycles;
              } else if (predEndCycle == nextCycle) {
                nextTimeInCycles = std::min(nextTimeInCycles, predEndTimeInCycles);
              }
            } else if (predEndCycle > nextCycle) {
              nextCycle = predEndCycle;
              nextTimeInCycles = predEndTimeInCycles;
            } else if (predEndCycle == nextCycle) {
              nextTimeInCycles = std::max(nextTimeInCycles, predEndTimeInCycles);
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
        for (auto &&op : body.getOps()) {
          if (auto operation = dyn_cast<schedule::OperationOp>(op)) {
            int depth = 15;
            auto printOp = [&](StringRef name, StringRef prefix = "") {
              if (operation.getSymName() == name) {
                std::string spaces(32 - prefix.size(), ' ');
                llvm::errs() << prefix << spaces;
                for (int i = 0; i < depth; ++i) {
                  llvm::errs() << startTimes[&op][iteration - 1 - i] << ", ";
                }
                llvm::errs() << "\n";
              }
            };
            // printOp("_5", "mu(r): ");
            // printOp("_60", "mux(ctrl_fwdr1): ");
            // printOp("_65", "gamma(r): ");
            // printOp("_67", "r[]: ");
            // printOp("_69", "gamma(forward_r_1): ");
            // printOp("_70", "add: ");
            // printOp("_71", "mul: ");
            // printOp("_72", "sub: ");
            // printOp("_73", "div: ");
            // printOp("_101", "gamma(merge__0): ");
            // printOp("_105", "alpha(r): ");
          }
        }
        circuitOp->setAttr(allowUnitIIAttrName, BoolAttr::get(circuitOp.getContext(), false));
        return;
      }

      startTimes[op].push_back(nextCycle);
      startTimesInCycles[op].push_back(nextTimeInCycles);
    }
  }

  circuitOp->setAttr(allowUnitIIAttrName, BoolAttr::get(circuitOp.getContext(), true));
}
