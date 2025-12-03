//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/SpecHLS/MobilityAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <circt/Support/LLVM.h>
#include <limits>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>

#include <cstdint>

using uint64_t = std::uint64_t;

namespace spechls {

MobilityAnalysis::MobilityAnalysis(spechls::TaskOp task, double targetClock) {
  uint64_t sumDistances = 0;
  llvm::SmallVector<spechls::GammaOp> gammas;

  auto *body = task.getBodyBlock();

  struct InitializerWalker {
    llvm::SmallVector<spechls::GammaOp> &gammas;
    uint64_t &sumDistances;

    InitializerWalker(llvm::SmallVector<spechls::GammaOp> &gammas, uint64_t &sumDistances)
        : gammas(gammas), sumDistances(sumDistances) {}

    void operator()(spechls::GammaOp gamma) { gammas.push_back(gamma); }
    void operator()(spechls::MuOp mu) { ++sumDistances; }
    void operator()(spechls::DelayOp delay) { sumDistances += delay.getDepth(); }
  };
  InitializerWalker initWalker(gammas, sumDistances);

  body->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *, void>(op)
        .Case([&](spechls::GammaOp &gamma) { gammas.push_back(gamma); })
        .Case([&](spechls::MuOp &mu) { ++sumDistances; })
        .Case([&](spechls::DelayOp &delay) { sumDistances += delay.getDepth(); })
        .Default([](mlir::Operation *) {});
  });
  uint64_t iterationCount = 2 * (sumDistances + 1);

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint64_t>> startTimesAsap;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>> startTimesInCyclesAsap;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint64_t>> startTimesAlap;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>> startTimesInCyclesAlap;

  body->walk([&](mlir::Operation *op) {
    startTimesAsap.try_emplace(op, llvm::SmallVector<uint64_t>());
    startTimesInCyclesAsap.try_emplace(op, llvm::SmallVector<double>());
    startTimesAlap.try_emplace(op, llvm::SmallVector<uint64_t>());
    startTimesInCyclesAlap.try_emplace(op, llvm::SmallVector<double>());

    startTimesAsap[op].reserve(iterationCount);
    startTimesInCyclesAsap[op].reserve(iterationCount);
    startTimesAlap[op].reserve(iterationCount);
    startTimesInCyclesAlap[op].reserve(iterationCount);
  });

  llvm::DenseMap<mlir::Operation *, std::tuple<double, int, double>> latencyCache;
  auto getTimingInfo = [&](mlir::Operation *op) {
    if (latencyCache.contains(op))
      return latencyCache[op];
    if (auto delayAttr = op->getAttrOfType<mlir::FloatAttr>("spechls.combDelay")) {
      double combDelay = delayAttr.getValueAsDouble();
      int latency = 0;
      double outDelay = combDelay;
      while (combDelay >= targetClock) {
        ++latency;
        combDelay -= targetClock;
        outDelay = 0.0;
      }
      auto result = std::make_tuple(combDelay, latency, outDelay);
      latencyCache.try_emplace(op, result);
      return result;
    }
    auto result = std::make_tuple<double, int, double>(0.0, 0, 0.0);
    latencyCache.try_emplace(op, result);
    return result;
  };

  for (uint64_t iteration = 0; iteration < iterationCount; ++iteration) {
    body->walk([&](mlir::Operation *op) {
      bool isGamma = llvm::isa<spechls::GammaOp>(op);
      unsigned distance = llvm::TypeSwitch<mlir::Operation *, unsigned>(op)
                              .Case([](spechls::MuOp &) { return 1; })
                              .Case([](spechls::DelayOp &delay) { return delay.getDepth(); })
                              .Default([](mlir::Operation *) { return 0; });
      uint64_t nextCycleAsap = isGamma ? std::numeric_limits<uint64_t>::max() : 0;
      uint64_t nextCycleAlap = 0;
      double nextTimeInCyclesAsap = 0.0;
      double nextTimeInCyclesAlap = 0.0;

      for (unsigned predIndex = (isGamma ? 1 : 0); predIndex < op->getNumOperands(); ++predIndex) {
        auto operand = op->getOperand(predIndex);
        if (iteration < distance) {
          nextCycleAsap = 0;
          nextTimeInCyclesAsap = 0.0;
        } else {
          if (auto *pred = operand.getDefiningOp()) {
            uint64_t predEndCycleAsap = 0;
            uint64_t predEndCycleAlap = 0;
            double predEndTimeInCyclesAsap = 0.0;
            double predEndTimeInCyclesAlap = 0.0;

            auto predTiming = getTimingInfo(pred);
            double predInDelay = std::get<0>(predTiming);
            int predLatency = std::get<1>(predTiming);
            double predOutDelay = std::get<2>(predTiming);

            auto computePredEnd = [&](int predStartCycle, double predStartTimeInCycles) -> std::pair<int, double> {
              if (predLatency > 0)
                return std::make_pair(predStartCycle + predLatency, predOutDelay);
              if (predStartTimeInCycles + predInDelay + predOutDelay > targetClock)
                return std::make_pair(predStartCycle + 1, predOutDelay);
              return std::make_pair(predStartCycle, predStartTimeInCycles + predOutDelay);
            };

            uint64_t offset = iteration - distance;

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
          } else {
            nextCycleAsap = 0;
            nextTimeInCyclesAsap = 0.0;
          }
        }
      }
      startTimesAsap[op].push_back(nextCycleAsap);
      startTimesAlap[op].push_back(nextCycleAlap);
      startTimesInCyclesAsap[op].push_back(nextTimeInCyclesAsap);
      startTimesInCyclesAlap[op].push_back(nextTimeInCyclesAlap);
    });
  }

  // Compute mobilities.
  for (auto &&g : gammas) {
    uint64_t mobility = 0;
    for (uint64_t iteration = sumDistances + 1; iteration < iterationCount; ++iteration) {
      uint64_t candidateMobility = (startTimesAlap[g][iteration] - startTimesAlap[g][iteration - 1]) -
                                   (startTimesAsap[g][iteration] - startTimesAsap[g][iteration - 1]);
      mobility = std::max(mobility, candidateMobility);
    }
    mobilities.try_emplace(g, mobility);
  }
}

} // namespace spechls