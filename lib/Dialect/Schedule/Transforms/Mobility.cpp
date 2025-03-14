//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/IR/ScheduleOps.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep

using namespace mlir;

namespace schedule {
#define GEN_PASS_DEF_MOBILITYPASS
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace schedule

namespace {

class MobilityPass : public schedule::impl::MobilityPassBase<MobilityPass> {
public:
  using MobilityPassBase::MobilityPassBase;
  void runOnOperation() override;
};

} // namespace

// TODO: Move attributes into the operation definition.
void MobilityPass::runOnOperation() {
  auto circuitOp = getOperation();
  double targetClock = circuitOp.getTargetClock().convertToDouble();
  auto &body = circuitOp.getBody();

  int64_t sumDistances = 0;
  SmallVector<Operation *> gammas;
  StringRef gammaAttrName = "SpecHLS.gamma";
  StringRef mobilityAttrName = "SpecHLS.mobility";
  StringRef distanceArrayAttrName = "distances";

  for (auto &&op : body.getOps()) {
    if (op.hasAttr(gammaAttrName))
      gammas.push_back(&op);
    for (auto &&attr : op.getAttrOfType<ArrayAttr>(distanceArrayAttrName))
      sumDistances += cast<IntegerAttr>(attr).getInt();
  }
  int64_t iterationCount = 2 * (sumDistances + 1);

  DenseMap<Operation *, SmallVector<int64_t>> startTimesAsap;
  DenseMap<Operation *, SmallVector<int64_t>> startTimesAlap;
  DenseMap<Operation *, SmallVector<double>> startTimesInCyclesAsap;
  DenseMap<Operation *, SmallVector<double>> startTimesInCyclesAlap;

  for (auto &&op : body.getOps()) {
    startTimesAsap.try_emplace(&op, SmallVector<int64_t>());
    startTimesAlap.try_emplace(&op, SmallVector<int64_t>());
    startTimesInCyclesAsap.try_emplace(&op, SmallVector<double>());
    startTimesInCyclesAlap.try_emplace(&op, SmallVector<double>());
    // The size of these vectors is used in the following code, so we only reserve and don't allocate.
    startTimesAsap[&op].reserve(iterationCount);
    startTimesAlap[&op].reserve(iterationCount);
    startTimesInCyclesAsap[&op].reserve(iterationCount);
    startTimesInCyclesAlap[&op].reserve(iterationCount);
  }

  for (int64_t iteration = 0; iteration < iterationCount; ++iteration) {
    for (auto &&op : body.getOps()) {
      auto distanceArray = op.getAttrOfType<ArrayAttr>(distanceArrayAttrName).getValue();

      bool isGamma = op.hasAttr(gammaAttrName);
      int64_t nextCycleAsap = isGamma ? std::numeric_limits<int64_t>::max() : 0;
      int64_t nextCycleAlap = 0;
      double nextTimeInCyclesAsap = 0.0;
      double nextTimeInCyclesAlap = 0.0;

      // Compute the unrolled schedule.
      for (size_t predIndex = 0; predIndex < op.getNumOperands(); ++predIndex) {
        int64_t distance = cast<IntegerAttr>(distanceArray[predIndex]).getInt();
        if (iteration - distance < 0) {
          nextCycleAsap = 0;
          nextTimeInCyclesAsap = 0.0;
        } else {
          int64_t predEndCycleAsap = 0;
          int64_t predEndCycleAlap = 0;
          double predEndTimeInCyclesAsap = 0.0;
          double predEndTimeInCyclesAlap = 0.0;

          Operation *pred = op.getOperand(predIndex).getDefiningOp();
          int64_t predLatency = pred->getAttrOfType<IntegerAttr>("latency").getInt();
          double predInDelay = pred->getAttrOfType<FloatAttr>("inDelay").getValueAsDouble();
          double predOutDelay = pred->getAttrOfType<FloatAttr>("outDelay").getValueAsDouble();

          auto computePredEnd = [=](int predStartCycle, double predStartTimeInCycles) -> std::pair<int, double> {
            if (predLatency > 0)
              return std::make_pair(predStartCycle + predLatency, predOutDelay);
            if (predStartTimeInCycles + predInDelay + predOutDelay > targetClock)
              return std::make_pair(predStartCycle + 1, predOutDelay);
            return std::make_pair(predStartCycle, predStartTimeInCycles + predOutDelay);
          };

          size_t offset = iteration - distance;

          // ASAP.
          if (offset < startTimesAsap[pred].size()) {
            auto [cycle, time] = computePredEnd(startTimesAsap[pred][offset], startTimesInCyclesAsap[pred][offset]);
            predEndCycleAsap = cycle;
            predEndTimeInCyclesAsap = time;

            if (isGamma) {
              if (nextCycleAsap > predEndCycleAsap) {
                nextCycleAsap = predEndCycleAsap;
                nextTimeInCyclesAsap = predEndTimeInCyclesAsap;
              } else if (nextCycleAsap == predEndCycleAsap) {
                nextTimeInCyclesAsap = std::min(nextTimeInCyclesAsap, predEndTimeInCyclesAsap);
              }
            } else if (predEndCycleAsap > nextCycleAsap) {
              nextCycleAsap = predEndCycleAsap;
              nextTimeInCyclesAsap = predEndTimeInCyclesAsap;
            }
          } else {
            nextCycleAsap = 0;
            nextTimeInCyclesAsap = 0.0;
          }

          // ALAP.
          if (offset < startTimesAlap[pred].size()) {
            auto [cycle, time] = computePredEnd(startTimesAlap[pred][offset], startTimesInCyclesAlap[pred][offset]);
            predEndCycleAlap = cycle;
            predEndTimeInCyclesAlap = time;

            if (predEndCycleAlap > nextCycleAlap) {
              nextCycleAlap = predEndCycleAlap;
              nextTimeInCyclesAlap = predEndTimeInCyclesAlap;
            } else if (predEndCycleAlap == nextCycleAlap) {
              nextTimeInCyclesAlap = std::max(nextTimeInCyclesAlap, predEndTimeInCyclesAlap);
            }
          }
        }
      }

      startTimesAsap[&op].push_back(nextCycleAsap);
      startTimesAlap[&op].push_back(nextCycleAlap);
      startTimesInCyclesAsap[&op].push_back(nextTimeInCyclesAsap);
      startTimesInCyclesAlap[&op].push_back(nextTimeInCyclesAlap);
    }
  }

  // Compute mobilities.
  for (auto &&g : gammas) {
    int64_t mobility = 0;
    for (int64_t iteration = sumDistances + 1; iteration < iterationCount; ++iteration) {
      int64_t candidateMobility = (startTimesAlap[g][iteration] - startTimesAlap[g][iteration - 1]) -
                                  (startTimesAsap[g][iteration] - startTimesAsap[g][iteration - 1]);
      mobility = std::max(mobility, candidateMobility);
    }
    g->setAttr(mobilityAttrName, IntegerAttr::get(IntegerType::get(g->getContext(), 32), mobility));
  }
}
