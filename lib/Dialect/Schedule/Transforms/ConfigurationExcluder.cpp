//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <mlir/Support/LLVM.h>

#include "Dialect/Schedule/IR/ScheduleOps.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep
#include "llvm/Support/raw_ostream.h"

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

      if (isMu) {
        int64_t distance = cast<IntegerAttr>(distanceArray[1]).getInt();
        int64_t offset = iteration - distance;
        if (offset >= 0) {
          auto *alpha = op->getOperand(1).getDefiningOp();
          alpha->print(llvm::errs());
          llvm::errs() << ": " << startTimes[alpha].back() << "\n";
          auto *gamma_merge0 = alpha->getOperand(1).getDefiningOp();
          llvm::errs() << "  ";
          gamma_merge0->print(llvm::errs());
          llvm::errs() << ": " << startTimes[gamma_merge0].back() << "\n";
          for (auto &&operand : gamma_merge0->getOperands()) {
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
          }
          llvm::errs() << "---\n";
        }
      }

      if (cast<schedule::OperationOp>(op).getSymName() == "_92") {
        llvm::errs() << "@_92:\n";
        for (auto &&operand : op->getOperands()) {
          if (startTimes[operand.getDefiningOp()].size() >= 2)
            llvm::errs() << startTimes[operand.getDefiningOp()][startTimes[operand.getDefiningOp()].size() - 2] << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_104") {
        llvm::errs() << "@_104:\n";
        for (auto &&operand : op->getOperands()) {
          if (startTimes[operand.getDefiningOp()].size() >= 2)
            llvm::errs() << startTimes[operand.getDefiningOp()][startTimes[operand.getDefiningOp()].size() - 2] << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_10") {
        llvm::errs() << "@_10:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_51") {
        llvm::errs() << "@_51:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_52") {
        llvm::errs() << "@_52:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_53") {
        llvm::errs() << "@_53:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_54") {
        llvm::errs() << "@_54:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_68") {
        llvm::errs() << "@_68:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_80") {
        llvm::errs() << "@_80:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_84") {
        llvm::errs() << "@_84:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_81") {
        llvm::errs() << "@_81:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_86") {
        llvm::errs() << "@_86:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_91") {
        llvm::errs() << "@_91:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_99") {
        llvm::errs() << "@_99:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_100") {
        llvm::errs() << "@_100:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      } else if (cast<schedule::OperationOp>(op).getSymName() == "_101") {
        llvm::errs() << "@_101:\n";
        for (auto &&operand : op->getOperands()) {
          if (!startTimes[operand.getDefiningOp()].empty())
            llvm::errs() << startTimes[operand.getDefiningOp()].back() << "\n";
        }
        llvm::errs() << "---\n";
      }

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
#if 0
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

            printOp("_5", "mu(r): ");
            printOp("_60", "mux(ctrl_fwdr1): ");
            printOp("_65", "gamma(r): ");
            printOp("_67", "r[]: ");
            printOp("_69", "gamma(forward_r_1): ");
            printOp("_70", "add: ");
            printOp("_71", "mul: ");
            printOp("_72", "sub: ");
            printOp("_73", "div: ");
            printOp("_101", "gamma(merge__0): ");
            printOp("_105", "alpha(r): ");
          }
        }
#endif
        circuitOp->setAttr(allowUnitIIAttrName, BoolAttr::get(circuitOp.getContext(), false));
        return;
      }

      startTimes[op].push_back(nextCycle);
      startTimesInCycle[op].push_back(nextTimeInCycle);
    }
  }

  circuitOp->setAttr(allowUnitIIAttrName, BoolAttr::get(circuitOp.getContext(), true));
}
