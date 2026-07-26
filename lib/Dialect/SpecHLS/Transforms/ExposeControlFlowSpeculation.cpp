//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/SpecHLS/OperationDelayAnalysis.h"
#include "Analysis/SpecHLS/SchedulingAnalysis.h"
#include "Analysis/SpecHLS/SpeculationExplorationAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "Dialect/SpecHLS/Transforms/TopologicalSort.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/SSP/SSPDialect.h>
#include <circt/Scheduling/Algorithms.h>
#include <circt/Scheduling/Problems.h>
#include <limits>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <queue>
#include <string>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_EXPOSECONTROLFLOWSPECULATIONPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

circt::scheduling::ChainingCyclicProblem duplicateProblem(circt::scheduling::ChainingCyclicProblem &problem,
                                                          circt::ssp::DependenceGraphOp &graph) {
  circt::scheduling::ChainingCyclicProblem result(graph);

  // Initialize operations & operators
  graph.getBodyBlock()->walk([&](circt::ssp::OperationOp op) {
    result.insertOperation(op);
    auto opr = *problem.getLinkedOperatorType(op);
    auto inc = *problem.getIncomingDelay(opr);
    auto lat = *problem.getLatency(opr);
    auto out = *problem.getOutgoingDelay(opr);
    result.setIncomingDelay(opr, inc);
    result.setLatency(opr, lat);
    result.setOutgoingDelay(opr, out);
    result.setLinkedOperatorType(op, opr);
  });

  // Initialize dependences
  graph.getBodyBlock()->walk([&](circt::ssp::OperationOp op) {
    for (auto dep : problem.getDependences(op)) {
      if (dep.isAuxiliary()) {
        if (failed(result.insertDependence(dep))) {
          return mlir::WalkResult::interrupt();
        }
        result.setDistance(dep, *problem.getDistance(dep));
      }
    }
    return mlir::WalkResult::advance();
  });

  return result;
}

llvm::SmallVector<mlir::Operation *> getTransitivePredecessors(mlir::Operation *op,
                                                               llvm::DenseSet<mlir::Operation *> pipelinedOperations) {
  llvm::SmallVector<mlir::Operation *> result;
  std::queue<mlir::Operation *> workingList;
  workingList.push(op);
  while (!workingList.empty()) {
    mlir::Operation *current = workingList.front();
    workingList.pop();
    if (!pipelinedOperations.contains(current) && !spechls::topologicalSortCriterion(nullptr, current)) {
      result.push_back(current);
      for (auto operand : current->getOperands()) {
        if (auto *pred = operand.getDefiningOp()) {
          workingList.push(pred);
        }
      }
    }
  }
  return result;
}

namespace {

llvm::SmallVector<int64_t> uniquifyDepths(llvm::ArrayRef<int64_t> depths) {
  llvm::SmallVector<int64_t> result;
  result.reserve(depths.size());
  auto contains = [&](int64_t depth) {
    for (auto d : result)
      if (d == depth)
        return true;
    return false;
  };
  for (auto d : depths) {
    if ((d != 0) && !contains(d)) {
      result.push_back(d);
    }
  }
  return result;
}

} // namespace

struct ExposeControlFlowSpeculationPass
    : public spechls::impl::ExposeControlFlowSpeculationPassBase<ExposeControlFlowSpeculationPass> {
  using ExposeControlFlowSpeculationPassBase::ExposeControlFlowSpeculationPassBase;

  llvm::DenseMap<spechls::GammaOp, llvm::DenseSet<spechls::GammaOp>> poisonMap;

  void initPoisonMap(llvm::ArrayRef<spechls::GammaOp> gammas) {
    for (spechls::GammaOp gamma : gammas) {
      llvm::DenseSet<spechls::GammaOp> poisoned;
      auto preds = getTransitivePredecessors(gamma, llvm::DenseSet<mlir::Operation *>());
      for (auto *op : preds) {
        if (auto gamma = llvm::dyn_cast<spechls::GammaOp>(op)) {
          poisoned.insert(gamma);
        }
      }
      poisonMap.try_emplace(gamma, poisoned);
    }
  }

  llvm::SmallVector<spechls::GammaOp>
  sortGammas(llvm::ArrayRef<spechls::GammaOp> gammas, llvm::DenseMap<spechls::GammaOp, unsigned> resolveDelays,
             llvm::DenseMap<spechls::GammaOp, mlir::SmallVector<unsigned>> inputLatencies) {
    auto sorter = [&](spechls::GammaOp g1, spechls::GammaOp g2) {
      int resolve1 = resolveDelays[g1], resolve2 = resolveDelays[g2];
      if (resolve1 > resolve2)
        return true;

      if (resolve2 > resolve1)
        return false;

      if (poisonMap[g2].contains(g1))
        return false;

      if (poisonMap[g1].contains(g2))
        return true;

      int maxInputDelays1 = 0;
      for (int lat : inputLatencies[g1])
        maxInputDelays1 = std::max(maxInputDelays1, lat);
      int maxInputDelays2 = 0;
      for (int lat : inputLatencies[g2])
        maxInputDelays2 = std::max(maxInputDelays2, lat);

      if (maxInputDelays1 > maxInputDelays2)
        return false;
      return true;
    };
    llvm::SmallVector<spechls::GammaOp> result;
    for (spechls::GammaOp gamma : gammas)
      result.push_back(gamma);
    llvm::sort(result.begin(), result.end(), sorter);
    return result;
  }

  struct ResourceConstraintsInfo {
    llvm::DenseMap<mlir::StringAttr, int> readPorts, writePorts;
  };

  ResourceConstraintsInfo generateResourceConstraint(spechls::TaskOp &task) {
    ResourceConstraintsInfo result;
    task.getBodyBlock()->walk([&](spechls::MuOp mu) {
      if (llvm::isa<spechls::ArrayType>(mu->getResult(0).getType())) {
        if (auto constr = mu->getAttrOfType<mlir::IntegerAttr>("spechls.readPortContraint")) {
          result.readPorts.try_emplace(mu.getSymNameAttr(), constr.getInt());
        }
        if (auto constr = mu->getAttrOfType<mlir::IntegerAttr>("spechls.writePortContraint")) {
          result.writePorts.try_emplace(mu.getSymNameAttr(), constr.getInt());
        }
      }
    });
    return result;
  }

  struct UnpipelineCoordinate {
    bool pipelined;
    unsigned unpipelineStage;
  };

  struct UnpipelineAnalysisResult {
    llvm::DenseMap<spechls::GammaOp, llvm::SmallVector<llvm::DenseMap<spechls::DelayOp, UnpipelineCoordinate>>>
        operationsCoordinate;
    llvm::DenseSet<mlir::Operation *> alwaysPipelined;
    llvm::DenseMap<spechls::DelayOp, unsigned> pipelineDelays;
  };

  unsigned getTypeBitWidth(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, unsigned>(type)
        .Case<mlir::IntegerType, mlir::FloatType>([](mlir::Type type) { return type.getIntOrFloatBitWidth(); })
        .Case<spechls::StructType>([&](spechls::StructType type) { return type.getBitWidth(); })
        .Default([](mlir::Type type) {
          llvm::errs() << "Unhandled type for `getTypeBitWidth`.\n";
          type.dump();
          exit(1);
          return 0;
        });
  }

  llvm::DenseSet<spechls::DelayOp> extractSlowDelays(spechls::GammaOp &gamma, unsigned input,
                                                     UnpipelineAnalysisResult &unpipelineInfo) {
    llvm::DenseSet<spechls::DelayOp> result;
    llvm::DenseSet<mlir::Operation *> seen;

    std::queue<mlir::Operation *> workList;

    if (auto *pred = gamma.getInputs()[input].getDefiningOp()) {
      if (!unpipelineInfo.alwaysPipelined.contains(pred)) {
        seen.insert(pred);
        workList.push(pred);
        if (auto d = llvm::dyn_cast<spechls::DelayOp>(pred)) {
          result.insert(d);
        }
      }
    }

    while (!workList.empty()) {
      auto *current = workList.front();
      workList.pop();
      for (unsigned idx = 0; idx < current->getNumOperands(); ++idx) {
        if (auto *pred = current->getOperand(idx).getDefiningOp()) {
          if (!unpipelineInfo.alwaysPipelined.contains(pred) && !seen.contains(pred)) {
            seen.insert(pred);
            workList.push(pred);
            if (auto d = llvm::dyn_cast<spechls::DelayOp>(pred)) {
              result.insert(d);
            }
          }
        }
      }
    }

    return result;
  }

  std::optional<UnpipelineAnalysisResult>
  computeUnpipelineAnalysis(spechls::TaskOp &task, llvm::DenseMap<spechls::GammaOp, unsigned> speculations,
                            double clockPeriod, spechls::TimingAnalyser &analyser, mlir::PassManager &canonicalizePm,
                            ResourceConstraintsInfo constrs,
                            llvm::DenseMap<spechls::GammaOp, llvm::SmallVector<unsigned>> &inputLatencies,
                            llvm::DenseMap<mlir::Operation *, unsigned> &startTimes,
                            llvm::DenseMap<mlir::Operation *, double> &startTimesInCycle,
                            llvm::DenseMap<spechls::GammaOp, unsigned> &gammaCond, mlir::MLIRContext *ctx) {
    UnpipelineAnalysisResult result;

    // Schedule the 'fast' cycle
    mlir::OpBuilder builder(ctx);

    auto module = builder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx));
    builder.setInsertionPointToStart(module.getBody());
    mlir::IRMapping mapper;
    auto newTask = llvm::dyn_cast<spechls::TaskOp>(builder.clone(*task, mapper));
    llvm::SmallVector<mlir::Value> newArgs;
    for (auto arg : newTask.getArgs()) {
      auto argType = arg.getType();
      auto cst = builder.create<spechls::DummyOp>(mlir::UnknownLoc::get(ctx), argType);
      newArgs.push_back(builder.create<circt::hw::BitcastOp>(mlir::UnknownLoc::get(ctx), argType, cst));
    }
    newTask.getArgsMutable().assign(newArgs);
    for (auto [gamma, spec] : speculations) {
      llvm::errs() << gamma.getSymName() << " -> " << spec << "\n";
      auto mappedGamma = llvm::dyn_cast<spechls::GammaOp>(mapper.lookup(gamma).getDefiningOp());
      auto operand = mappedGamma->getOperand(spec);
      mappedGamma.getInputsMutable().clear();
      builder.setInsertionPointAfter(mappedGamma);
      mappedGamma.getInputsMutable().assign(operand);
    }
    if (failed(canonicalizePm.run(module)))
      return std::nullopt;

    builder.setInsertionPointToStart(module.getBody());
    auto graph = builder.create<circt::ssp::DependenceGraphOp>(mlir::UnknownLoc::get(ctx));
    builder.setInsertionPointToStart(graph.getBodyBlock());

    circt::scheduling::ChainingCyclicProblem problem(graph);

    llvm::DenseMap<mlir::Operation *, circt::ssp::OperationOp> association;
    // create operations
    unsigned idx = 0;
    if (newTask.getBodyBlock()
            ->walk([&](mlir::Operation *op) {
              auto timing = analyser.computeOperationTiming(op, clockPeriod, ctx);
              if (!timing)
                return mlir::WalkResult::interrupt();

              auto [in, lat, out] = *timing;

              auto sspOp =
                  builder.create<circt::ssp::OperationOp>(op->getLoc(), 1, llvm::SmallVector<mlir::Value>(),
                                                          mlir::StringAttr::get(ctx, "op_" + std::to_string(idx)));
              circt::scheduling::ChainingCyclicProblem::OperatorType operatorType(
                  mlir::StringAttr::get(ctx, "opr_" + std::to_string(idx)));

              association.try_emplace(op, sspOp);

              problem.insertOperation(sspOp);
              problem.setLinkedOperatorType(sspOp, operatorType);

              problem.setLatency(operatorType, lat);
              problem.setIncomingDelay(operatorType, in);
              problem.setOutgoingDelay(operatorType, out);

              ++idx;
              return mlir::WalkResult::advance();
            })
            .wasInterrupted())
      return std::nullopt;

    // Create dependences
    if (newTask.getBodyBlock()
            ->walk([&](mlir::Operation *op) {
              auto sspOp = association[op];
              return llvm::TypeSwitch<mlir::Operation *, mlir::WalkResult>(op)
                  .Case([&](spechls::MuOp mu) {
                    for (auto predVal : op->getOperands()) {
                      if (auto *pred = predVal.getDefiningOp()) {
                        circt::scheduling::ChainingCyclicProblem::Dependence dep(association[pred], sspOp);
                        if (failed(problem.insertDependence(dep)))
                          return mlir::WalkResult::interrupt();
                        problem.setDistance(dep, 1);
                      }
                    }
                    return mlir::WalkResult::advance();
                  })
                  .Case<spechls::DelayOp, spechls::CancellableDelayOp, spechls::RollbackableDelayOp>([&](auto op) {
                    auto depth = op.getDepth();
                    for (auto predVal : op->getOperands()) {
                      if (auto *pred = predVal.getDefiningOp()) {
                        circt::scheduling::ChainingCyclicProblem::Dependence dep(association[pred], sspOp);
                        if (failed(problem.insertDependence(dep)))
                          return mlir::WalkResult::interrupt();
                        problem.setDistance(dep, depth);
                      }
                    }
                    return mlir::WalkResult::advance();
                  })
                  .Default([&](mlir::Operation *op) {
                    llvm::SmallVector<mlir::Value> operands;
                    for (auto predVal : op->getOperands()) {
                      if (mlir::Operation *pred = predVal.getDefiningOp()) {
                        operands.push_back(association[pred].getResult(0));
                      }
                    }
                    sspOp.getOperandsMutable().assign(operands);
                    return mlir::WalkResult::advance();
                  });
            })
            .wasInterrupted())
      return std::nullopt;

    auto terminator = association[newTask.getBodyBlock()->getTerminator()];

    assert(mlir::succeeded(circt::scheduling::scheduleSimplex(problem, terminator, clockPeriod)));
    assert(mlir::succeeded(problem.verify()));
    assert(problem.getInitiationInterval() == 1);

    task.getBodyBlock()->walk([&](mlir::Operation *op) {
      if (auto *mappedOp = mapper.lookupOrNull(op)) {
        if (association.contains(mappedOp)) {
          auto sspOp = association[mappedOp];
          result.alwaysPipelined.insert(op);
          startTimes.try_emplace(op, *problem.getStartTime(sspOp));
          startTimesInCycle.try_emplace(op, *problem.getStartTimeInCycle(sspOp));
        }
      }
    });

    // Compute slow start times.
    llvm::DenseSet<mlir::Operation *> alreadyComputed;
    std::queue<mlir::Operation *> workingList;
    task.getBodyBlock()->walk([&](mlir::Operation *op) {
      auto *mappedOp = mapper.lookupOrNull(op);
      if (mappedOp && association.contains(mappedOp) && startTimes.contains(op)) {
        alreadyComputed.insert(op);
      } else {
        workingList.push(op);
      }
    });

    while (!workingList.empty()) {
      auto *op = workingList.front();
      workingList.pop();
      bool isReady = true;
      auto timing = analyser.computeOperationTiming(op, clockPeriod, ctx);
      if (!timing)
        return std::nullopt;
      auto [in, lat, out] = *timing;

      int potentialStartTime = 0;
      double potentialStartTimeInCycle = 0.0;
      for (auto predVal : op->getOperands()) {
        if (auto *pred = predVal.getDefiningOp()) {
          if (!alreadyComputed.contains(pred)) {
            workingList.push(op);
            isReady = false;
            break;
          }

          auto predTiming = analyser.computeOperationTiming(pred, clockPeriod, ctx);
          if (!predTiming)
            return std::nullopt;
          auto [predIn, predLat, predOut] = *predTiming;
          int predStartTime = startTimes[pred];
          double predStartTimeInCycle = startTimesInCycle[pred];
          int predEndTime;
          double predEndTimeInCycle;

          if (predLat == 0) {
            predEndTime = predStartTime;
            predEndTimeInCycle = predStartTimeInCycle + predOut;
          } else {
            predEndTime = predStartTime + predLat;
            predEndTimeInCycle = predOut;
          }

          if (predEndTime > potentialStartTime) {
            potentialStartTime = predEndTime;
            potentialStartTimeInCycle = predEndTimeInCycle;
          } else if (predEndTime == potentialStartTime) {
            potentialStartTimeInCycle = std::max(potentialStartTimeInCycle, predEndTimeInCycle);
          }
        }
      }
      if (!isReady)
        continue;

      if (potentialStartTimeInCycle + in > clockPeriod) {
        ++potentialStartTime;
        potentialStartTimeInCycle = 0;
      }
      startTimes[op] = potentialStartTime;
      startTimesInCycle[op] = potentialStartTimeInCycle;
      alreadyComputed.insert(op);
    }

    // Update input latencies
    for (auto &[gamma, input] : speculations) {
      if (!gamma->hasAttr("spechls.memspec")) {
        for (unsigned i = 1; i < gamma->getNumOperands(); ++i) {
          if (i != input) {
            if (auto *pred = gamma.getOperand(i).getDefiningOp()) {

              auto timing = analyser.computeOperationTiming(pred, clockPeriod, ctx);
              if (!timing)
                return std::nullopt;
              // auto t = *timing;

              inputLatencies[gamma][i - 1] = std::max(startTimes[pred] + 1, gammaCond[gamma] + 1);
            } else {
              inputLatencies[gamma][i - 1] = gammaCond[gamma] + 1;
            }
          }
        }
      }
    }

    builder.setInsertionPointToStart(task.getBodyBlock());

    // Add pipeline delays
    mlir::Value trueCst = builder.create<circt::hw::ConstantOp>(builder.getUnknownLoc(), builder.getI1Type(), 1);

    task.getBodyBlock()->walk([&](mlir::Operation *op) {
      if (result.alwaysPipelined.contains(op))
        return;
      if (llvm::isa<spechls::MuOp>(op))
        return;

      unsigned opStartTime = startTimes[op];
      for (unsigned predIdx = 0; predIdx < op->getNumOperands(); ++predIdx) {
        auto pred = op->getOperand(predIdx);
        if (auto *predOp = pred.getDefiningOp()) {
          unsigned predStartTime = result.alwaysPipelined.contains(predOp) ? 0u : startTimes[predOp];
          auto current = pred;
          for (unsigned i = predStartTime; i < opStartTime; ++i) {
            auto delay =
                builder.create<spechls::DelayOp>(builder.getUnknownLoc(), pred.getType(), current, 1, trueCst, nullptr);
            result.pipelineDelays.try_emplace(delay, i);
            current = delay.getResult();
          }
          op->setOperand(predIdx, current);
        }
      }
    });

    // Add delays for slow faster than conds
    for (auto [gamma, cond] : gammaCond) {
      if (!gamma->hasAttr("spechls.memspec")) {
        auto spec = speculations[gamma];
        auto latencies = inputLatencies[gamma];
        for (unsigned input = 1; input < gamma.getNumOperands(); ++input) {
          if (input != spec) {
            auto pred = gamma->getOperand(input);
            if (auto *predOp = pred.getDefiningOp()) {
              auto predStartTime = startTimes[predOp];
              mlir::Value current = pred;
              for (unsigned i = predStartTime; i < cond; ++i) {
                auto delay = builder.create<spechls::DelayOp>(builder.getUnknownLoc(), pred.getType(), current, 1,
                                                              trueCst, nullptr);
                result.pipelineDelays.try_emplace(delay, i);
                current = delay.getResult();
              }
              gamma->setOperand(input, current);
            }
          }
        }
      }
    }

    return result;
  }

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();
    auto canonicalizePm = mlir::PassManager::on<mlir::ModuleOp>(ctx);
    canonicalizePm.addPass(mlir::createCanonicalizerPass());
    if (failed(canonicalizePm.run(getOperation()->getParentOfType<mlir::ModuleOp>())))
      return signalPassFailure();
    spechls::KernelOp kernel = getOperation();

    if (failed(spechls::timingAnalyserFactory.registerAnalyers(targetsFile)))
      return signalPassFailure();

    auto annotatePm = mlir::PassManager::on<spechls::KernelOp>(ctx);
    annotatePm.addPass(spechls::createLowerComplexOperationsPass());
    annotatePm.addPass(spechls::createAnnotateTimingPass(
        spechls::AnnotateTimingPassOptions{targetTask, clockPeriod, target, targetsFile}));
    annotatePm.addPass(spechls::createSplitWithSyncPass(spechls::SplitWithSyncPassOptions{targetTask, clockPeriod}));
    if (failed(annotatePm.run(kernel)))
      return signalPassFailure();

    if (targetTask != "") {
      if (failed(runOnTask(kernel, canonicalizePm, ctx, targetTask)))
        return signalPassFailure();
    } else {
      llvm::SmallVector<std::string> tasksName;
      kernel.walk([&](spechls::TaskOp task) { tasksName.push_back(task.getSymName().str()); });
      for (auto task : tasksName) {
        if (failed(runOnTask(kernel, canonicalizePm, ctx, task)))
          return signalPassFailure();
      }
    }
  }

  llvm::LogicalResult runOnTask(spechls::KernelOp kernel, mlir::PassManager &canonicalizePm, mlir::MLIRContext *ctx,
                                std::string taskName) {

    spechls::TaskOp task = nullptr;
    kernel.getBody().walk([&](spechls::TaskOp t) {
      if (t.getSymName() == taskName) {
        task = t;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (task == nullptr) {
      llvm::errs() << "Target task not found.\n";
      return llvm::failure();
    }

    auto constraint = generateResourceConstraint(task);
    llvm::SmallVector<spechls::GammaOp> gammas;
    llvm::DenseMap<spechls::GammaOp, unsigned> specGammas, resolveDelays;
    llvm::DenseMap<spechls::GammaOp, mlir::SmallVector<unsigned>> inputLatencies;
    llvm::SmallVector<int64_t> rollbackDepths;
    llvm::SmallVector<int64_t> rewindDepths;

    if (useExistingSpeculation) {
      bool invalidSelection = false;
      task.getBodyBlock()->walk([&](spechls::GammaOp gamma) {
        gammas.push_back(gamma);
        auto selection = gamma->getAttrOfType<mlir::IntegerAttr>("spechls.speculation");
        if (!selection)
          return;
        int64_t input = selection.getInt();
        if (input <= 0 || input >= gamma->getNumOperands()) {
          gamma.emitOpError("has an invalid spechls.speculation input index");
          invalidSelection = true;
          return;
        }
        specGammas.try_emplace(gamma, static_cast<unsigned>(input));
      });
      if (invalidSelection)
        return llvm::failure();
    } else {
      spechls::SpeculationExplorationAnalysis exploration(task, clockPeriod, probabilityThreshold, traceFileName);
      task.getBodyBlock()->walk([&](spechls::GammaOp gamma) {
        gammas.push_back(gamma);
        int pid = gamma->getAttrOfType<mlir::IntegerAttr>("spechls.profilingId").getInt();
        int confElt = exploration.configuration[exploration.pidToEid[pid]];
        if (confElt != 0)
          specGammas.try_emplace(gamma, confElt);
      });
    }
    if (specGammas.empty())
      return llvm::success();

    task->setAttr("spechls.speculativeTask", mlir::UnitAttr::get(ctx));

    llvm::DenseMap<circt::ssp::OperationOp, mlir::Operation *> reverseAssociation;
    llvm::DenseMap<mlir::Operation *, circt::ssp::OperationOp> association;

    mlir::OpBuilder builder(ctx);
    auto graph = builder.create<circt::ssp::DependenceGraphOp>(mlir::UnknownLoc::get(ctx));
    builder.setInsertionPointToStart(graph.getBodyBlock());
    unsigned idx = 0, oprId = 0;

    circt::scheduling::ChainingCyclicProblem problem(graph);
    task.getBodyBlock()->walk([&](mlir::Operation *op) {
      auto sspOp =
          builder.create<circt::ssp::OperationOp>(mlir::UnknownLoc::get(ctx), 1, llvm::SmallVector<mlir::Value>(),
                                                  mlir::StringAttr::get(ctx, "op_" + std::to_string(idx++)));
      reverseAssociation.try_emplace(sspOp, op);
      association.try_emplace(op, sspOp);
      problem.insertOperation(sspOp);
      circt::scheduling::ChainingCyclicProblem::OperatorType operatorType(
          mlir::StringAttr::get(ctx, "opr_" + std::to_string(oprId++)));
      unsigned latency = 0;
      double combDelay = 0.0;
      if (op->hasAttrOfType<mlir::FloatAttr>("spechls.combDelay")) {
        combDelay = op->getAttrOfType<mlir::FloatAttr>("spechls.combDelay").getValueAsDouble();
        while (combDelay >= clockPeriod) {
          combDelay -= clockPeriod;
          ++latency;
        }
      }
      if (latency == 0) {
        problem.setLatency(operatorType, 0);
        problem.setIncomingDelay(operatorType, combDelay);
        problem.setOutgoingDelay(operatorType, combDelay);
      } else {
        problem.setLatency(operatorType, latency);
        problem.setIncomingDelay(operatorType, combDelay);
        problem.setOutgoingDelay(operatorType, 0);
      }
      problem.setLinkedOperatorType(sspOp, operatorType);
    });

    auto terminator = association[task.getBodyBlock()->getTerminator()];

    task.getBodyBlock()->walk([&](mlir::Operation *op) {
      auto sspOp = association[op];
      if (!llvm::isa<spechls::GammaOp>(op) || !specGammas.contains(llvm::dyn_cast<spechls::GammaOp>(op))) {
        if (llvm::isa<spechls::MuOp>(op)) {
          for (auto operand : op->getOperands()) {
            if (auto *pred = operand.getDefiningOp()) {
              circt::scheduling::ChainingCyclicProblem::Dependence dependence(association[pred], sspOp);
              assert(mlir::succeeded(problem.insertDependence(dependence)));
              problem.setDistance(dependence, 1);
            }
          }
        } else {
          auto treatDelay = [&](auto d) {
            for (auto operand : d->getOperands()) {
              if (auto *pred = operand.getDefiningOp()) {
                circt::scheduling::ChainingCyclicProblem::Dependence dependence(association[pred], sspOp);
                assert(mlir::succeeded(problem.insertDependence(dependence)));
                problem.setDistance(dependence, d.getDepth());
              }
            }
          };
          if (auto delay = llvm::dyn_cast<spechls::DelayOp>(op)) {
            treatDelay(delay);
          } else if (auto delay = llvm::dyn_cast<spechls::RollbackableDelayOp>(op)) {
            treatDelay(delay);
          } else if (auto delay = llvm::dyn_cast<spechls::CancellableDelayOp>(op)) {
            treatDelay(delay);
          } else {
            llvm::SmallVector<mlir::Value> operands;
            for (auto operand : op->getOperands()) {
              if (auto *pred = operand.getDefiningOp()) {
                sspOp.getOperandsMutable().append(association[pred]->getResult(0));
              }
            }
            for (auto &opOp : sspOp->getOpOperands()) {
              circt::scheduling::ChainingCyclicProblem::Dependence dependence(&opOp);
              assert(mlir::succeeded(problem.insertDependence(dependence)));
            }
          }
        }
      }
    });
    // Compute resolve latencies of speculated Gammas
    for (spechls::GammaOp specGamma : specGammas.keys()) {
      for (spechls::GammaOp gamma : specGammas.keys()) {
        auto sspOp = association[gamma];
        auto predVal = gamma->getOperand((gamma == specGamma) ? 0 : (specGammas[gamma]));
        if (auto *pred = predVal.getDefiningOp()) {
          sspOp.getOperandsMutable().append(association[pred].getResult(0));
        }
      }
      assert(mlir::succeeded(circt::scheduling::scheduleSimplex(problem, terminator, clockPeriod)));
      assert(mlir::succeeded(problem.verify()));

      int resolveDelay = 1;
      if (problem.getInitiationInterval() != 1) {
        auto sspGamma = association[specGamma];
        sspGamma.getOperandsMutable().clear();

        // Condition has a non-null definingOp, else it would have a resolveDelay of 0.
        auto sspCond = association[specGamma.getOperand(0).getDefiningOp()];
        auto newProblem = duplicateProblem(problem, graph);
        circt::scheduling::ChainingCyclicProblem::Dependence dependence(sspCond, sspGamma);
        if (failed(newProblem.insertDependence(dependence))) {
          return llvm::failure();
        }
        do {
          ++resolveDelay;
          newProblem.setDistance(dependence, resolveDelay - 1);
          assert(mlir::succeeded(circt::scheduling::scheduleSimplex(newProblem, terminator, clockPeriod)));
          assert(mlir::succeeded(newProblem.verify()));
        } while (*newProblem.getInitiationInterval() != 1);
      }

      resolveDelays.try_emplace(specGamma, resolveDelay);
      rollbackDepths.push_back(resolveDelay);
      rewindDepths.push_back(resolveDelay);

      for (spechls::GammaOp gamma : specGammas.keys())
        association[gamma].getOperandsMutable().clear();
    }

    // Compute input latencies
    for (spechls::GammaOp specGamma : specGammas.keys()) {
      for (spechls::GammaOp gamma : specGammas.keys()) {
        if (gamma != specGamma) {
          auto sspOp = association[gamma];
          auto predVal = gamma->getOperand(specGammas[gamma]);
          if (auto *pred = predVal.getDefiningOp()) {
            sspOp.getOperandsMutable().append(association[pred].getResult(0));
          }
        }
      }

      auto sspOp = association[specGamma];
      llvm::SmallVector<unsigned> latencies;

      for (unsigned index = 1; index < specGamma->getNumOperands(); ++index) {
        if (index == static_cast<unsigned>(specGammas[specGamma])) {
          latencies.push_back(0);
        } else {
          if (auto *pred = specGamma->getOperand(index).getDefiningOp()) {
            auto sspPred = association[pred];
            // Try latency of 0
            sspOp.getOperandsMutable().append(sspPred.getResult(0));

            assert(mlir::succeeded(circt::scheduling::scheduleSimplex(problem, terminator, clockPeriod)));
            assert(mlir::succeeded(problem.verify()));
            sspOp.getOperandsMutable().clear();
            if (problem.getInitiationInterval() == 1) {
              latencies.push_back(resolveDelays[specGamma]);
            } else {
              // Try successives distances to compute latency
              unsigned distance = 1;
              auto newProblem = duplicateProblem(problem, graph);
              circt::scheduling::ChainingCyclicProblem::Dependence dependence(sspPred, sspOp);
              if (failed(newProblem.insertDependence(dependence)))
                return llvm::failure();
              do {
                ++distance;
                newProblem.setDistance(dependence, distance - 1);
                assert(mlir::succeeded(circt::scheduling::scheduleSimplex(newProblem, terminator, clockPeriod)));
                assert(mlir::succeeded(newProblem.verify()));
              } while (newProblem.getInitiationInterval() != 1);
              distance = std::max<int>(resolveDelays[specGamma], distance);
              latencies.push_back(distance);
              rewindDepths.push_back(distance);
            }

          } else {
            latencies.push_back(resolveDelays[specGamma]);
          }
        }
      }
      inputLatencies.try_emplace(specGamma, latencies);

      for (auto gamma : specGammas.keys()) {
        association[gamma].getOperandsMutable().clear();
      }
    }

    graph.erase();
    builder.setInsertionPointToStart(task.getBodyBlock());
    auto trueCst = builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getIntegerType(1), true);

    llvm::DenseSet<spechls::DelayOp> condDelayNodes;
    // Add delays to gamma conditions
    // Add delays to memspec gammas
    for (auto &[gamma, resolve] : resolveDelays) {
      gamma->setAttr("spechls.resolveDelay", builder.getI32IntegerAttr(resolve));
      if (resolve > 1) {
        auto delay = builder.create<spechls::DelayOp>(mlir::UnknownLoc::get(ctx), gamma->getOperand(0).getType(),
                                                      gamma->getOperand(0), resolve - 1, trueCst, nullptr);
        condDelayNodes.insert(delay);
        gamma->setOperand(0, delay.getResult());
      }
      if (gamma->hasAttr("spechls.memspec")) {
        // Big Scotch :(
        mlir::Value lastValue = gamma.getOperand(gamma->getNumOperands() - 1);
        // // recompute input latencies
        // llvm::SmallVector<unsigned> latencies;
        // unsigned current = gamma->getNumOperands() - 1;
        // for (unsigned i = 1; i < gamma.getNumOperands(); ++i) {
        //   latencies.push_back(current--);
        // }
        // inputLatencies[gamma] = latencies;
        // rewire inputs
        for (unsigned idx = 1; idx < gamma.getNumOperands() - 1; ++idx) {
          gamma.setOperand(idx, lastValue);
        }
      }
    }

    llvm::DenseMap<mlir::Operation *, unsigned int> startTimes;
    llvm::DenseMap<mlir::Operation *, double> startTimesInCycle;
    auto analyser = spechls::timingAnalyserFactory.get(target);
    auto constrs = generateResourceConstraint(task);
    auto unpipelineInfo = computeUnpipelineAnalysis(task, specGammas, clockPeriod, analyser, canonicalizePm, constrs,
                                                    inputLatencies, startTimes, startTimesInCycle, resolveDelays, ctx);
    if (!unpipelineInfo.has_value()) {
      return llvm::failure();
    }
    initPoisonMap(gammas);
    auto sortedGammaNodes =
        sortGammas(llvm::SmallVector<spechls::GammaOp>(specGammas.keys()), resolveDelays, inputLatencies);

    // This standard MLIR attribute is the handoff between configuration
    // selection and FSM construction. It deliberately uses builtin
    // dictionaries/arrays so xDSL and native MLIR can consume the same IR.
    llvm::DenseMap<spechls::GammaOp, unsigned> speculationIds;
    for (unsigned index = 0; index < sortedGammaNodes.size(); ++index)
      speculationIds.try_emplace(sortedGammaNodes[index], index);
    llvm::SmallVector<mlir::Attribute> configurationEntries;
    for (auto gamma : sortedGammaNodes) {
      llvm::SmallVector<mlir::Attribute> slowPaths;
      unsigned fastSelector = specGammas[gamma] - 1;
      for (unsigned selector = 0; selector < inputLatencies[gamma].size(); ++selector) {
        if (selector == fastSelector)
          continue;
        slowPaths.push_back(mlir::DictionaryAttr::get(
            ctx, {mlir::NamedAttribute(builder.getStringAttr("selector"), builder.getI64IntegerAttr(selector)),
                  mlir::NamedAttribute(builder.getStringAttr("latency"),
                                       builder.getI64IntegerAttr(inputLatencies[gamma][selector])),
                  mlir::NamedAttribute(builder.getStringAttr("rewind"), builder.getI64IntegerAttr(0)),
                  mlir::NamedAttribute(builder.getStringAttr("rbwe"), builder.getI64IntegerAttr(1)),
                  mlir::NamedAttribute(builder.getStringAttr("rollback_mu_ids"), builder.getArrayAttr({})),
                  mlir::NamedAttribute(builder.getStringAttr("rollback_array_ids"), builder.getArrayAttr({})),
                  mlir::NamedAttribute(builder.getStringAttr("rollback_gamma_ids"), builder.getArrayAttr({})),
                  mlir::NamedAttribute(builder.getStringAttr("rollback_depth"),
                                       builder.getI64IntegerAttr(inputLatencies[gamma][selector]))}));
      }
      llvm::SmallVector<mlir::Attribute> poisoned;
      for (auto dependency : poisonMap[gamma])
        if (dependency != gamma)
          if (auto id = speculationIds.find(dependency); id != speculationIds.end())
          poisoned.push_back(builder.getI64IntegerAttr(id->second));
      configurationEntries.push_back(mlir::DictionaryAttr::get(
          ctx, {mlir::NamedAttribute(builder.getStringAttr("gamma_id"), builder.getStringAttr(gamma.getSymName())),
                mlir::NamedAttribute(builder.getStringAttr("cond_latency"),
                                     builder.getI64IntegerAttr(resolveDelays[gamma])),
                mlir::NamedAttribute(builder.getStringAttr("fast_selector"), builder.getI64IntegerAttr(fastSelector)),
                mlir::NamedAttribute(builder.getStringAttr("poison_speculation_ids"),
                                     builder.getArrayAttr(poisoned)),
                mlir::NamedAttribute(builder.getStringAttr("slow_paths"), builder.getArrayAttr(slowPaths))}));
    }
    task->setAttr("spechls.speculation_config", builder.getArrayAttr(configurationEntries));

    builder.setInsertionPointToStart(task.getBodyBlock());

    llvm::SmallVector<mlir::Attribute> gammaNames;
    llvm::SmallVector<int64_t> condDelays, fastIndices;
    llvm::SmallVector<mlir::Attribute> inputDelays;
    llvm::SmallVector<mlir::Value> mispecInputs;
    llvm::SmallVector<std::string> fsmCmdTypeFieldNames;
    llvm::SmallVector<mlir::Type> fsmCmdTypeFieldTypes;
    std::string fsmName = "fsm_";
    llvm::StringMap<int> namesId;
    unsigned maxCond = 0;
    for (auto &[gamma, resolve] : resolveDelays)
      maxCond = std::max(maxCond, resolve);

    fsmCmdTypeFieldNames.push_back("nextInput");
    fsmCmdTypeFieldTypes.push_back(builder.getI1Type());
    fsmCmdTypeFieldNames.push_back("commit");
    fsmCmdTypeFieldTypes.push_back(builder.getI1Type());
    fsmCmdTypeFieldNames.push_back("muRollBack");
    fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
    fsmCmdTypeFieldNames.push_back("arrayRollBack");
    fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
    fsmCmdTypeFieldNames.push_back("rewind");
    fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
    fsmCmdTypeFieldNames.push_back("rbwe");
    fsmCmdTypeFieldTypes.push_back(builder.getI1Type());

    llvm::DenseMap<spechls::GammaOp, std::string> gammaNameMap;

    for (spechls::GammaOp g : sortedGammaNodes) {
      std::string name = g.getSymName().str();
      if (!namesId.contains(name)) {
        namesId.try_emplace(name, 0);
        name = name + "0";
      } else {
        int id = namesId[name];
        namesId.try_emplace(name, id + 1);
        name = name + std::to_string(id);
      }
      gammaNameMap.try_emplace(g, name);
      gammaNames.push_back(builder.getStringAttr(name));
      fsmName += name;

      fsmCmdTypeFieldNames.push_back("gammaRollBack_" + name);
      fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
      fsmCmdTypeFieldNames.push_back("selSlowPath_" + name);
      fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
      for (unsigned index = 0; index < g.getInputs().size(); ++index) {
        fsmCmdTypeFieldNames.push_back("stall_" + name + "_" + std::to_string(index));
        fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
      }

      condDelays.push_back(resolveDelays[g]);
      fastIndices.push_back(specGammas[g] - 1);
      llvm::SmallVector<mlir::Attribute> inputs;
      for (auto lat : inputLatencies[g]) {
        inputs.push_back(builder.getIntegerAttr(builder.getI64Type(), lat));
      }
      inputDelays.push_back(builder.getArrayAttr(inputs));
      mispecInputs.push_back(g.getSelect());
    }
    auto module = kernel->getParentOfType<mlir::ModuleOp>();
    if (SymbolTable::lookupSymbolIn(module, fsmName))
      fsmName += "_" + task.getSymName().str();
    auto initialState = builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0);
    auto fsmStateMu = builder.create<spechls::MuOp>(mlir::UnknownLoc::get(ctx),
                                                     builder.getStringAttr(fsmName + "State"), initialState, initialState);
    llvm::SmallVector<mlir::Type> machineInputs{builder.getI32Type()};
    for (auto input : mispecInputs)
      machineInputs.push_back(input.getType());
    llvm::SmallVector<mlir::Type> machineResults{builder.getI32Type()};
    machineResults.append(fsmCmdTypeFieldTypes);
    auto machineType = builder.getFunctionType(machineInputs, machineResults);

    // The controller declaration is reusable and state-free. Its state lives
    // in the task-local mu and all selectors are independent input ports.
    OpBuilder moduleBuilder(ctx);
    moduleBuilder.setInsertionPoint(kernel);
    auto fsm = moduleBuilder.create<spechls::FSMOp>(mlir::UnknownLoc::get(ctx), builder.getStringAttr(fsmName), machineType);
    builder.create<spechls::FSMInstanceOp>(mlir::UnknownLoc::get(ctx), builder.getStringAttr(fsmName),
                                           FlatSymbolRefAttr::get(ctx, fsmName));
    llvm::SmallVector<mlir::Value> triggerInputs{fsmStateMu.getResult()};
    triggerInputs.append(mispecInputs);
    auto fsmTrigger = builder.create<spechls::FSMTriggerOp>(mlir::UnknownLoc::get(ctx), machineResults,
                                                             builder.getStringAttr(fsmName), triggerInputs);
    fsmStateMu.getLoopValueMutable().assign(fsmTrigger.getResult(0));

    // Retain the state table in the declaration without coupling its ports to
    // speculation-specific aggregate types.
    auto &fsmBody = fsm.getBody();
    fsmBody.emplaceBlock();
    OpBuilder fsmBuilder(ctx);
    fsmBuilder.setInsertionPointToStart(&fsmBody.front());
    llvm::StringMap<spechls::FSMStateOp> fsmStates;
    auto commandValues = [&](bool nextInput, bool commit) {
      llvm::SmallVector<int64_t> values(fsmCmdTypeFieldNames.size(), 0);
      for (unsigned index = 0; index < fsmCmdTypeFieldNames.size(); ++index) {
        if (fsmCmdTypeFieldNames[index] == "nextInput")
          values[index] = nextInput;
        else if (fsmCmdTypeFieldNames[index] == "commit")
          values[index] = commit;
      }
      return values;
    };
    auto addState = [&](llvm::StringRef name, llvm::ArrayRef<int64_t> commands) {
      fsmBuilder.setInsertionPointToEnd(&fsmBody.front());
      auto state = fsmBuilder.create<spechls::FSMStateOp>(mlir::UnknownLoc::get(ctx), fsmBuilder.getStringAttr(name));
      state.getOutput().emplaceBlock();
      fsmBuilder.setInsertionPointToStart(&state.getOutput().front());
      llvm::SmallVector<mlir::Attribute> outputNames;
      for (auto &fieldName : fsmCmdTypeFieldNames)
        outputNames.push_back(fsmBuilder.getStringAttr(fieldName));
      fsmBuilder.create<spechls::FSMOutputOp>(mlir::UnknownLoc::get(ctx), fsmBuilder.getArrayAttr(outputNames),
                                              fsmBuilder.getDenseI64ArrayAttr(commands));
      state.getTransitions().emplaceBlock();
      fsmStates[name] = state;
    };
    auto addTransition = [&](llvm::StringRef source, llvm::StringRef target, llvm::ArrayRef<int64_t> inputIds,
                              llvm::ArrayRef<int64_t> selectors, llvm::StringRef kind = "normal") {
      auto sourceState = fsmStates.lookup(source);
      assert(sourceState && "FSM transition source must be created before its transition");
      fsmBuilder.setInsertionPointToEnd(&sourceState.getTransitions().front());
      auto transition = fsmBuilder.create<spechls::FSMTransitionOp>(mlir::UnknownLoc::get(ctx),
                                                                      fsmBuilder.getStringAttr(target),
                                                                      fsmBuilder.getStringAttr(kind));
      if (inputIds.empty())
        return;
      auto &guard = transition.getGuard();
      guard.emplaceBlock();
      fsmBuilder.setInsertionPointToStart(&guard.front());
      mlir::Value predicate;
      for (auto [inputId, selector] : llvm::zip_equal(inputIds, selectors)) {
        auto inputType = llvm::cast<mlir::IntegerType>(machineInputs[inputId + 1]);
        auto input = fsmBuilder.create<spechls::FSMInputOp>(mlir::UnknownLoc::get(ctx), inputType,
                                                              fsmBuilder.getI64IntegerAttr(inputId + 1));
        auto selected = fsmBuilder.create<mlir::arith::ConstantIntOp>(mlir::UnknownLoc::get(ctx), selector,
                                                                        inputType.getWidth());
        auto match = fsmBuilder.create<mlir::arith::CmpIOp>(mlir::UnknownLoc::get(ctx),
                                                             mlir::arith::CmpIPredicate::eq, input, selected);
        if (predicate)
          predicate = fsmBuilder.create<circt::comb::AndOp>(mlir::UnknownLoc::get(ctx), predicate,
                                                             match.getResult()).getResult();
        else
          predicate = match.getResult();
      }
      fsmBuilder.create<spechls::FSMReturnOp>(mlir::UnknownLoc::get(ctx), predicate);
    };

    unsigned initCount = std::max(1u, maxCond);
    for (unsigned index = 0; index < initCount; ++index) {
      auto name = "Init_" + std::to_string(index);
      addState(name, commandValues(false, false));
      addTransition(name, index + 1 < initCount ? "Init_" + std::to_string(index + 1) : "Proceed", {}, {});
    }
    addState("Proceed", commandValues(true, true));
    addTransition("Proceed", "Proceed", {}, {});
    struct BodyPath {
      unsigned gammaIndex;
      unsigned pathIndex;
      unsigned selector;
      unsigned latency;
      unsigned condLatency;
      std::string gammaName;
    };
    auto orderedBefore = [&](const BodyPath &left, const BodyPath &right) {
      auto leftGamma = sortedGammaNodes[left.gammaIndex];
      auto rightGamma = sortedGammaNodes[right.gammaIndex];
      // Mirror the Java recovery worklist's poison-aware ordering before
      // falling back to condition latency and stable speculation ID order.
      if (poisonMap[leftGamma].contains(rightGamma))
        return true;
      if (poisonMap[rightGamma].contains(leftGamma))
        return false;
      if (left.condLatency != right.condLatency)
        return left.condLatency < right.condLatency;
      return left.gammaIndex < right.gammaIndex;
    };
    llvm::SmallVector<llvm::SmallVector<BodyPath>> alternatives(sortedGammaNodes.size());
    for (unsigned gammaIndex = 0; gammaIndex < sortedGammaNodes.size(); ++gammaIndex) {
      auto gamma = sortedGammaNodes[gammaIndex];
      unsigned pathIndex = 0;
      for (unsigned selector = 0; selector < gamma.getInputs().size(); ++selector) {
        if (selector == fastIndices[gammaIndex])
          continue;
        alternatives[gammaIndex].push_back(
            {gammaIndex, pathIndex++, selector, inputLatencies[gamma][selector], resolveDelays[gamma], gammaNameMap[gamma]});
      }
    }

    // Enumerate the Java-style canonical combinations. The same limit used by
    // the xDSL builder prevents an exponential controller table from escaping
    // into the IR.
    llvm::SmallVector<llvm::SmallVector<BodyPath>> plans(1);
    for (auto &paths : alternatives) {
      auto previous = plans;
      for (auto &plan : previous)
        for (auto &path : paths) {
          auto combined = plan;
          combined.push_back(path);
          llvm::sort(combined, orderedBefore);
          plans.push_back(std::move(combined));
          if (plans.size() > 257)
            break;
        }
      if (plans.size() > 257)
        break;
    }
    if (plans.size() > 257)
      return llvm::failure();

    auto planKey = [](llvm::ArrayRef<BodyPath> plan) {
      std::string key;
      for (auto &path : plan)
        key += std::to_string(path.gammaIndex) + ":" + std::to_string(path.selector) + ";";
      return key;
    };
    auto commandsFor = [&](llvm::ArrayRef<BodyPath> plan, bool rollback) {
      auto commands = commandValues(false, false);
      for (auto &path : plan) {
        for (unsigned index = 0; index < fsmCmdTypeFieldNames.size(); ++index) {
          if (fsmCmdTypeFieldNames[index] == "selSlowPath_" + path.gammaName)
            commands[index] = path.selector;
          if (rollback && (fsmCmdTypeFieldNames[index] == "rbwe" ||
                           fsmCmdTypeFieldNames[index] == "gammaRollBack_" + path.gammaName))
            commands[index] = 1;
        }
      }
      return commands;
    };
    llvm::StringMap<std::string> planStarts;
    llvm::SmallVector<std::string> recoveryStates;
    auto addRecoveryChain = [&](llvm::StringRef prefix, llvm::ArrayRef<BodyPath> plan) {
      auto tail = plan.back();
      unsigned stalls = tail.latency > tail.condLatency ? tail.latency - tail.condLatency - 1 : 0;
      unsigned fills = tail.condLatency > 0 ? tail.condLatency - 1 : 0;
      llvm::SmallVector<std::string> names;
      for (unsigned index = 0; index < stalls; ++index)
        names.push_back(prefix.str() + "_Stall_" + std::to_string(index));
      names.push_back(prefix.str() + "_Rollback");
      for (unsigned index = 0; index < fills; ++index)
        names.push_back(prefix.str() + "_Fill_" + std::to_string(index));
      for (auto &name : names) {
        bool rollback = name.find("Rollback") != std::string::npos;
        addState(name, commandsFor(plan, rollback));
        recoveryStates.push_back(name);
      }
      for (unsigned index = 0; index + 1 < names.size(); ++index)
        addTransition(names[index], names[index + 1], {}, {});
      addTransition(names.back(), "Proceed", {}, {});
      return names.front();
    };

    unsigned combinedIndex = 0;
    for (auto &plan : plans) {
      if (plan.empty())
        continue;
      auto prefix = plan.size() == 1 ? std::to_string(plan.front().gammaIndex) + "_" + std::to_string(plan.front().pathIndex)
                                     : "Combined_" + std::to_string(combinedIndex++);
      auto start = addRecoveryChain(prefix, plan);
      planStarts[planKey(plan)] = start;
      llvm::SmallVector<int64_t> inputs, selectors;
      for (auto &path : plan) {
        inputs.push_back(path.gammaIndex);
        selectors.push_back(path.selector);
      }
      addTransition("Proceed", start, inputs, selectors);
    }

    // A recovery can observe a later mis-speculation. Materialize both the
    // explicit NEW_MISPEC hop and the direct CANCELED redirect used by the
    // Java worklist; both preserve exact selector guards in the IR table.
    unsigned newMispecIndex = 0;
    for (auto &plan : plans) {
      if (plan.empty())
        continue;
      llvm::DenseSet<unsigned> active;
      for (auto &path : plan)
        active.insert(path.gammaIndex);
      for (auto &paths : alternatives)
        for (auto &path : paths) {
          if (active.contains(path.gammaIndex))
            continue;
          auto destinationPlan = plan;
          destinationPlan.push_back(path);
          llvm::sort(destinationPlan, orderedBefore);
          auto destination = planStarts.lookup(planKey(destinationPlan));
          if (destination.empty())
            continue;
          auto newState = "NewMispec_" + std::to_string(newMispecIndex++);
          addState(newState, commandsFor(llvm::ArrayRef<BodyPath>(&path, 1), false));
          addTransition(newState, destination, {}, {});
          for (auto &state : recoveryStates) {
            addTransition(state, newState, {static_cast<int64_t>(path.gammaIndex)},
                          {static_cast<int64_t>(path.selector)}, "new_mispec");
            addTransition(state, destination, {static_cast<int64_t>(path.gammaIndex)},
                          {static_cast<int64_t>(path.selector)}, "canceled");
          }
        }
    }

    auto command = [&](llvm::StringRef name) -> mlir::Value {
      for (auto [index, fieldName] : llvm::enumerate(fsmCmdTypeFieldNames))
        if (fieldName == name)
          return fsmTrigger.getResult(index + 1);
      llvm_unreachable("unknown FSM command port");
    };
    auto nextInput = command("nextInput");
    auto commitEnable = command("commit");
    auto muRollBack = command("muRollBack");
    auto arrayRollBack = command("arrayRollBack");
    auto rewind = command("rewind");
    auto rbwe = command("rbwe");
    llvm::DenseMap<spechls::GammaOp, llvm::SmallVector<mlir::Value>> gammaStallFields;
    for (unsigned id = 0; id < sortedGammaNodes.size(); ++id) {
      auto gamma = sortedGammaNodes[id];
      auto name = llvm::dyn_cast<mlir::StringAttr>(gammaNames[id]).str();
      llvm::SmallVector<mlir::Value> fields;
      for (unsigned i = 0; i < gamma.getInputs().size(); ++i) {
        fields.push_back(command("stall_" + name + "_" + std::to_string(i)));
      }
      gammaStallFields.try_emplace(gamma, fields);
    }

    rollbackDepths = uniquifyDepths(rollbackDepths);
    long maxDepth = 0;
    for (auto &d : rollbackDepths) {
      maxDepth = std::max(maxDepth, d);
    }
    for (auto [_, lat] : inputLatencies) {
      for (auto d : lat)
        maxDepth = std::max(maxDepth, static_cast<long>(d));
    }
    llvm::SmallVector<int64_t> arrayRollbackDepths;
    for (long i = 1; i < maxDepth; ++i)
      arrayRollbackDepths.push_back(i);

    // Rewire gamma condition and add gamma rollbacks
    for (auto gamma : sortedGammaNodes) {
      auto gammaName = gammaNameMap[gamma];
      gamma.getSelectMutable().assign(command("selSlowPath_" + gammaName));
      if (!llvm::isa<spechls::ArrayType>(gamma.getResult().getType())) {
        auto rb = builder.create<spechls::RollbackOp>(mlir::UnknownLoc::get(ctx),
                                                       builder.getDenseI64ArrayAttr(rollbackDepths), 0, gamma.getResult(),
                                                       command("gammaRollBack_" + gammaName), rbwe);
        gamma.getResult().replaceAllUsesExcept(rb.getResult(), rb);
      }
    }

    // Rewire rollbackable/cancellable delays
    task.getBodyBlock()->walk([&](spechls::RollbackableDelayOp delay) {
      delay.setRollbackDepths(arrayRollbackDepths);
      delay.getRollbackMutable().assign(arrayRollBack);
    });
    task.getBodyBlock()->walk([&](spechls::CancellableDelayOp delay) {
      delay.getCancelMutable().assign(arrayRollBack);
      delay.getCancelWeMutable().assign(trueCst.getResult());
    });

    // Add mu rollbacks
    task.getBodyBlock()->walk([&](spechls::MuOp mu) {
      if (mu == fsmStateMu)
        return;
      spechls::RollbackOp rb;
      if (llvm::isa<spechls::ArrayType>(mu.getResult().getType())) {
        auto trueCst = builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI1Type(), 1);
        rb = builder.create<spechls::RollbackOp>(mlir::UnknownLoc::get(ctx),
                                                 builder.getDenseI64ArrayAttr(arrayRollbackDepths), 0, mu.getResult(),
                                                  arrayRollBack, trueCst.getResult());
      } else {
        rb = builder.create<spechls::RollbackOp>(mlir::UnknownLoc::get(ctx),
                                                 builder.getDenseI64ArrayAttr(rollbackDepths), 0, mu.getResult(),
                                                  muRollBack, rbwe);
      }
      mu.getResult().replaceAllUsesExcept(rb.getResult(), rb);
    });

    // rewire commit condition
    auto commit = llvm::dyn_cast<spechls::CommitOp>(task.getBodyBlock()->getTerminator());
    commit.setOperand(0, commitEnable);

    for (auto [_, latencies] : inputLatencies) {
      for (auto lat : latencies)
        rewindDepths.push_back(lat);
    }
    rewindDepths = uniquifyDepths(rewindDepths);

    // Insert rewind operations
    for (auto arg : task.getBodyBlock()->getArguments()) {
      if (!task.getArgs()[arg.getArgNumber()].getDefiningOp())
        continue;
      auto rewindOp = builder.create<spechls::RewindOp>(mlir::UnknownLoc::get(ctx), arg.getType(),
                                                         builder.getDenseI64ArrayAttr(rewindDepths), arg, rewind, nextInput);
      arg.replaceUsesWithIf(rewindOp.getResult(), [&](mlir::OpOperand &operand) -> bool {
        if (operand.getOwner() == rewindOp) {
          return false;
        }
        if (llvm::isa<spechls::MuOp>(operand.getOwner())) {
          return false;
        }
        if (llvm::isa<spechls::DelayOp>(operand.getOwner()) &&
            (operand.getOperandNumber()) == spechls::DelayOp::odsIndex_init) {
          return false;
        }
        if (llvm::isa<spechls::RollbackableDelayOp>(operand.getOwner()) &&
            (operand.getOperandNumber()) == spechls::RollbackableDelayOp::odsIndex_init) {
          return false;
        }
        if (llvm::isa<spechls::CancellableDelayOp>(operand.getOwner()) &&
            (operand.getOperandNumber()) == spechls::CancellableDelayOp::odsIndex_init) {
          return false;
        }

        return true;
      });
    }

    // Add delays before commits
    int maxResolveDelay = 0;
    for (int resolve : resolveDelays.values()) {
      maxResolveDelay = (maxResolveDelay < resolve) ? resolve : maxResolveDelay;
    }
    if (maxResolveDelay > 0) {
      for (unsigned i = 1; i < commit->getNumOperands(); ++i) {
        auto op = commit.getOperand(i);
        if (llvm::isa_and_nonnull<circt::hw::ConstantOp>(op.getDefiningOp()))
          continue;
        commit.setOperand(
            i, builder.create<spechls::DelayOp>(
                   mlir::UnknownLoc::get(ctx), op.getType(), op, maxResolveDelay,
                   builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI1Type(), 1), nullptr));
      }
    }

    // Compute unpipeline delay conditions
    llvm::DenseSet<spechls::DelayOp> initializedDelays;
    for (auto &[gamma, spec] : specGammas) {
      --spec;
      auto cond = resolveDelays[gamma];
      if (!gamma->hasAttr("spechls.memspec")) {
        for (unsigned i = 0; i < gamma.getInputs().size(); ++i) {
          if (i != spec) {
            auto stallField = gammaStallFields[gamma][i];
            auto slows = extractSlowDelays(gamma, i, *unpipelineInfo);
            for (auto &d : slows) {
              auto step = unpipelineInfo->pipelineDelays[d];
              mlir::Operation *newCond = builder.create<circt::comb::ICmpOp>(
                  builder.getUnknownLoc(), circt::comb::ICmpPredicate::eq, stallField,
                  builder.create<circt::hw::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(),
                                                        (step < (cond - 1)) ? 0 : (step - cond + 1)));
              if (initializedDelays.contains(d)) {
                newCond =
                    builder.create<circt::comb::OrOp>(builder.getUnknownLoc(), d.getEnable(), newCond->getResult(0));
              } else {
                initializedDelays.insert(d);
              }
              d.getEnableMutable().clear();
              d.getEnableMutable().append(newCond->getResult(0));
            }
          }
        }
      }
    }

    return llvm::success();
  }
};

} // namespace
