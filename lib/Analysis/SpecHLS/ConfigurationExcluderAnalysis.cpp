//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/SpecHLS/ConfigurationExcluderAnalysis.h"
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
#include <queue>

using uint64_t = std::uint64_t;

namespace spechls {

ConfigurationExcluderAnalysis::ConfigurationExcluderAnalysis(spechls::TaskOp task, double targetClock,
                                                             llvm::ArrayRef<int> configuration,
                                                             llvm::ArrayRef<GammaOp> gammas) {
  deadEnd = false;
  llvm::DenseMap<spechls::GammaOp, int> mapConfiguration;
  for (unsigned i = 0; i < gammas.size(); ++i) {
    mapConfiguration.try_emplace(gammas[i], configuration[i]);
  }
  auto *body = task.getBodyBlock();
  uint64_t sumDistances = 0;
  body->walk([&sumDistances](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *, void>(op)
        .Case([&sumDistances](spechls::MuOp &) { ++sumDistances; })
        .Case([&sumDistances](spechls::DelayOp &delay) { sumDistances += delay.getDepth(); });
  });
  const uint64_t iterationCount = 2 * (sumDistances + 1);

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint64_t>> startTimes;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>> startTimesInCycle;

  body->walk([&startTimes, &startTimesInCycle, iterationCount](mlir::Operation *op) {
    startTimes.try_emplace(op, llvm::SmallVector<uint64_t>());
    startTimesInCycle.try_emplace(op, llvm::SmallVector<double>());
    // The size of these vectors is used in the following code, so we only reserve and don't allocate.
    startTimes[op].reserve(iterationCount);
    startTimesInCycle[op].reserve(iterationCount);
  });

  llvm::DenseMap<mlir::Operation *, std::tuple<double, int, double>> timingCache;
  body->walk([&timingCache, targetClock](mlir::Operation *op) {
    if (auto delayAttr = op->getAttrOfType<mlir::FloatAttr>("spechls.combDelay")) {
      double inDelay = delayAttr.getValueAsDouble();
      int latency = 0;
      double outDelay = inDelay;
      while (inDelay >= targetClock) {
        inDelay -= targetClock;
        ++latency;
        outDelay = 0;
      }
      timingCache.try_emplace(op, std::make_tuple(inDelay, latency, outDelay));
    } else {
      timingCache.try_emplace(op, std::make_tuple<double, int, double>(0.0, 0, 0.0));
    }
  });

  for (uint64_t iteration = 0; iteration < iterationCount; ++iteration) {
    std::queue<mlir::Operation *> worklist;
    body->walk([&worklist](mlir::Operation *op) { worklist.push(op); });

    while (!worklist.empty()) {
      bool skip = false;
      mlir::Operation *op = worklist.front();
      worklist.pop();
      unsigned distance = llvm::TypeSwitch<mlir::Operation *, unsigned>(op)
                              .Case([](spechls::DelayOp &delay) { return delay.getDepth(); })
                              .Case([](spechls::MuOp &) { return 1; })
                              .Default([](mlir::Operation *) { return 0; });
      bool isGamma = llvm::isa<spechls::GammaOp>(op);
      int forcedEntry = -1;
      if (isGamma) {
        forcedEntry = mapConfiguration[llvm::cast<spechls::GammaOp>(op)];
        if (forcedEntry != -1)
          isGamma = false;
      }
      bool isMu = llvm::isa<spechls::MuOp>(op);
      uint64_t nextCycle = isGamma ? std::numeric_limits<uint64_t>::max() : 0;
      double nextTimeInCycle = 0.0;

      unsigned from = (forcedEntry == -1) ? (isGamma ? 1 : 0) : forcedEntry;
      unsigned to = (forcedEntry == -1) ? op->getNumOperands() : (forcedEntry + 1);

      for (unsigned predIndex = from; predIndex < to; ++predIndex) {
        if (iteration < distance) {
          if (isGamma) {
            nextCycle = 0;
            nextTimeInCycle = 0.0;
          }
        } else {
          uint64_t predEndCycle = 0;
          double predEndTimeInCycle = 0.0;

          if (mlir::Operation *pred = op->getOperand(predIndex).getDefiningOp()) {
            auto [predInDelay, predLatency, predOutDelay] = timingCache[pred];
            uint64_t offset = iteration - distance;
            if (offset < startTimes[pred].size()) {
              if (predLatency > 0) {
                predEndCycle = startTimes[pred][offset] + predLatency;
                if (startTimesInCycle[pred][offset] + predInDelay > targetClock) {
                  ++predEndCycle;
                }
                predEndTimeInCycle = predOutDelay;
              } else if (startTimesInCycle[pred][offset] + predInDelay > targetClock) {
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
              worklist.push(op);
              skip = true;
              break;
            }
          } else {
            if (isGamma) {
              nextCycle = 0;
              nextTimeInCycle = 0.0;
            }
          }
        }
      }

      if (skip)
        continue;

      if (isMu && (iteration > sumDistances + 1) && (nextCycle - startTimes[op][iteration - 1] > 1)) {
        deadEnd = true;
        return;
      }

      startTimes[op].push_back(nextCycle);
      startTimesInCycle[op].push_back(nextTimeInCycle);
    }
  }
}

} // namespace spechls