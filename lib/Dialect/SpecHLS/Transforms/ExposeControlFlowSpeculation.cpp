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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/SSP/SSPDialect.h>
#include <circt/Scheduling/Algorithms.h>
#include <circt/Scheduling/Problems.h>
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
  graph.getBodyBlock()->walk([&](circt::ssp::OperationOp op) {
    result.insertOperation(op);
    auto opr = *problem.getLinkedOperatorType(op);
    result.setLinkedOperatorType(op, opr);
    result.setIncomingDelay(opr, *problem.getIncomingDelay(opr));
    result.setLatency(opr, *problem.getLatency(opr));
    result.setOutgoingDelay(opr, *problem.getOutgoingDelay(opr));
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
  sortGammas(llvm::ArrayRef<spechls::GammaOp> gammas, llvm::DenseMap<spechls::GammaOp, int> resolveDelays,
             llvm::DenseMap<spechls::GammaOp, mlir::SmallVector<int>> inputLatencies) {
    auto sorter = [&](spechls::GammaOp g1, spechls::GammaOp g2) {
      int resolve1 = resolveDelays[g1], resolve2 = resolveDelays[g2];
      if (resolve1 > resolve2)
        return -1;

      if (resolve2 > resolve1)
        return 1;

      if (poisonMap[g2].contains(g1))
        return -1;

      if (poisonMap[g1].contains(g2))
        return 1;

      int maxInputDelays1 = 0;
      for (int lat : inputLatencies[g1])
        maxInputDelays1 = std::max(maxInputDelays1, lat);
      int maxInputDelays2 = 0;
      for (int lat : inputLatencies[g2])
        maxInputDelays2 = std::max(maxInputDelays2, lat);

      if (maxInputDelays1 > maxInputDelays2)
        return -1;
      if (maxInputDelays1 < maxInputDelays2)
        return 1;
      return 0;
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
    unsigned pipelineStage, unpipelineStage;
  };

  struct UnpipelineAnalysisResult {
    llvm::DenseMap<spechls::GammaOp, llvm::SmallVector<llvm::DenseMap<mlir::Operation *, UnpipelineCoordinate>>>
        operationsCoordinate;
    llvm::DenseMap<mlir::Operation *, unsigned> pipelinedCycles;
  };

  std::optional<UnpipelineAnalysisResult>
  computeUnpipelineAnalysis(spechls::TaskOp &task, llvm::DenseMap<spechls::GammaOp, int> speculations,
                            double clockPeriod, spechls::TimingAnalyser &analyser, mlir::PassManager &canonicalizePm,
                            ResourceConstraintsInfo constrs,
                            llvm::DenseMap<spechls::GammaOp, llvm::SmallVector<int>> &inputLatencies,
                            llvm::DenseMap<mlir::Operation *, int> &startTimes,
                            llvm::DenseMap<mlir::Operation *, double> &startTimesInCycle, mlir::MLIRContext *ctx) {
    UnpipelineAnalysisResult result;

    // Schedule the 'fast' cycle
    mlir::OpBuilder builder(ctx);

    auto module = builder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx));
    builder.setInsertionPointToStart(module.getBody());
    mlir::IRMapping mapper;
    auto newTask = llvm::dyn_cast<spechls::TaskOp>(builder.clone(*task, mapper));
    llvm::SmallVector<mlir::Value> newArgs;
    for (auto arg : newTask.getArgs()) {
      newArgs.push_back(builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), arg.getType(), 0));
    }
    newTask.getArgsMutable().assign(newArgs);
    for (auto [gamma, spec] : speculations) {
      auto mappedGamma = llvm::dyn_cast<spechls::GammaOp>(mapper.lookup(gamma).getDefiningOp());
      auto operand = mappedGamma->getOperand(spec);
      mappedGamma.getInputsMutable().clear();
      mappedGamma.getInputsMutable().assign(operand);
    }
    if (failed(canonicalizePm.run(module)))
      return std::nullopt;

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
                  .Case([&](spechls::MuOp) {
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

    task.getBodyBlock()->walk([&](mlir::Operation *op) {
      if (auto *mappedOp = mapper.lookupOrNull(op)) {
        if (association.contains(mappedOp)) {
          auto sspOp = association[mappedOp];
          startTimes.try_emplace(op, *problem.getStartTime(sspOp));
          startTimesInCycle.try_emplace(op, *problem.getStartTimeInCycle(sspOp));
        }
      }
    });

    // Compute slow start times.
    llvm::DenseSet<mlir::Operation *> alreadyComputed, pipelineCycle;
    std::queue<mlir::Operation *> workingList;
    task.getBodyBlock()->walk([&](mlir::Operation *op) {
      auto *mappedOp = mapper.lookupOrNull(op);
      if (mappedOp && association.contains(mappedOp) && startTimes.contains(op)) {
        alreadyComputed.insert(op);
        result.pipelinedCycles.try_emplace(op, startTimes[op]);
        pipelineCycle.insert(op);
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

    // Get gamma condition compute time
    llvm::DenseMap<spechls::GammaOp, int> gammaCond;
    task.getBodyBlock()->walk([&](spechls::GammaOp gamma) {
      if (mlir::Operation *cond = gamma.getSelect().getDefiningOp()) {
        if (auto delay = llvm::dyn_cast<spechls::DelayOp>(cond)) {
          if (mlir::Operation *cond2 = delay.getInput().getDefiningOp()) {
            cond = cond2;
          } else {
            gammaCond.try_emplace(gamma, 0);
            return;
          }
        }
        gammaCond.try_emplace(gamma, startTimes[cond]);
      }
    });

    // Compute unpipeline coordinates of "slows" nodes
    task.getBodyBlock()->walk([&](spechls::GammaOp gamma) {
      unsigned condTime = gammaCond[gamma];
      llvm::SmallVector<llvm::DenseMap<mlir::Operation *, UnpipelineCoordinate>> gammaCoordinates;
      gammaCoordinates.resize(gamma.getInputs().size());
      for (unsigned input = 0; input < gamma.getInputs().size(); ++input) {
        if (input == static_cast<unsigned>(speculations[gamma]) - 1)
          continue;
        if (auto *pred = gamma.getInputs()[input].getDefiningOp()) {
          auto transitivePredecessors = getTransitivePredecessors(pred, pipelineCycle);
          for (auto *op : transitivePredecessors) {
            unsigned startTime = startTimes[op];
            if (startTime > condTime) {
              unsigned unpipelineStage = startTime - condTime;
              gammaCoordinates[input].try_emplace(op, UnpipelineCoordinate{condTime, unpipelineStage});
            } else {
              gammaCoordinates[input].try_emplace(op, UnpipelineCoordinate{startTime, 0});
            }
          }
        }
      }
      result.operationsCoordinate.try_emplace(gamma, gammaCoordinates);
    });

    // Update input latencies
    for (auto &[gamma, input] : speculations) {
      if (!gamma->hasAttr("spechls.memspec")) {
        for (unsigned i = 1; i < gamma->getNumOperands(); ++i) {
          if (static_cast<int>(i) != input) {
            if (auto *pred = gamma.getOperand(i).getDefiningOp()) {
              inputLatencies[gamma][i - 1] = std::max(startTimes[pred], gammaCond[gamma]);
            } else {
              inputLatencies[gamma][i - 1] = std::max(0, gammaCond[gamma]);
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
      llvm::errs() << "Target tas knot found.\n";
      return llvm::failure();
    }

    spechls::SpeculationExplorationAnalysis exploration(task, clockPeriod, probabilityThreshold, traceFileName);

    bool needSpeculation = false;
    for (int config : exploration.configuration) {
      if (config > 0) {
        needSpeculation = true;
        break;
      }
    }
    if (!needSpeculation)
      return llvm::success();

    task->setAttr("spechls.speculativeTask", mlir::UnitAttr::get(ctx));

    auto constraint = generateResourceConstraint(task);
    llvm::SmallVector<spechls::GammaOp> gammas;
    llvm::DenseMap<spechls::GammaOp, int> specGammas, resolveDelays;
    llvm::DenseMap<spechls::GammaOp, mlir::SmallVector<int>> inputLatencies;
    llvm::SmallVector<int64_t> rollbackDepths;
    llvm::SmallVector<int64_t> rewindDepths;

    task.getBodyBlock()->walk([&](spechls::GammaOp gamma) {
      gammas.push_back(gamma);
      int pid = gamma->getAttrOfType<mlir::IntegerAttr>("spechls.profilingId").getInt();
      int confElt = exploration.configuration[exploration.pidToEid[pid]];
      if (confElt != 0) {
        specGammas.try_emplace(gamma, confElt);
      }
    });

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
        } else if (auto delay = llvm::dyn_cast<spechls::DelayOp>(op)) {
          for (auto operand : op->getOperands()) {
            if (auto *pred = operand.getDefiningOp()) {
              circt::scheduling::ChainingCyclicProblem::Dependence dependence(association[pred], sspOp);
              assert(mlir::succeeded(problem.insertDependence(dependence)));
              problem.setDistance(dependence, delay.getDepth());
            }
          }
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

      resolveDelays.try_emplace(specGamma, *problem.getInitiationInterval());
      rollbackDepths.push_back(*problem.getInitiationInterval());
      rewindDepths.push_back(*problem.getInitiationInterval());

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
      llvm::SmallVector<int> latencies;

      for (unsigned index = 1; index < specGamma->getNumOperands(); ++index) {
        if (auto *pred = specGamma->getOperand(index).getDefiningOp()) {
          auto sspPred = association[pred];
          // Try latency of 0
          sspOp.getOperandsMutable().append(sspPred.getResult(0));

          assert(mlir::succeeded(circt::scheduling::scheduleSimplex(problem, terminator, clockPeriod)));
          assert(mlir::succeeded(problem.verify()));
          if (problem.getInitiationInterval() == 1) {
            latencies.push_back(0);
          } else {
            // Try successives distances to compute latency
            unsigned distance = 0;
            auto newProblem = duplicateProblem(problem, graph);
            circt::scheduling::ChainingCyclicProblem::Dependence dependence(sspPred, sspOp);
            do {
              ++distance;
              newProblem.setDistance(dependence, distance);
              assert(mlir::succeeded(circt::scheduling::scheduleSimplex(newProblem, terminator, clockPeriod)));
              assert(mlir::succeeded(newProblem.verify()));
            } while (newProblem.getInitiationInterval() != 1);
            latencies.push_back(distance);
            rewindDepths.push_back(distance);
          }

        } else {
          latencies.push_back(0);
        }
      }
      inputLatencies.try_emplace(specGamma, latencies);
    }

    graph.erase();

    auto trueCst = builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getIntegerType(1), true);

    // Add delays to gamma conditions
    // Add delays to memspec gammas
    for (auto &[gamma, resolve] : resolveDelays) {
      if (resolve > 2) {
        auto delay = builder.create<spechls::DelayOp>(mlir::UnknownLoc::get(ctx), gamma->getOperand(0).getType(),
                                                      gamma->getOperand(0), resolve - 1, trueCst, nullptr);
        gamma->setOperand(0, delay.getResult());
      }

      if (gamma->hasAttr("spechls.memspec")) {
        auto &latencies = inputLatencies[gamma];
        int resolve = resolveDelays[gamma];
        for (unsigned index = 1; index < gamma->getNumOperands(); ++index) {
          auto numDelay = std::max(latencies[index] - 1, resolve);
          if (numDelay > 0) {
            auto delay = builder.create<spechls::DelayOp>(mlir::UnknownLoc::get(ctx), gamma.getOperand(index).getType(),
                                                          gamma.getOperand(index), numDelay, trueCst, nullptr);
            gamma.setOperand(index, delay);
          }
        }
      }
    }

    initPoisonMap(gammas);
    auto sortedGammaNodes = sortGammas(gammas, resolveDelays, inputLatencies);

    auto analyser = spechls::timingAnalyserFactory.get(target);
    llvm::DenseMap<mlir::Operation *, int> startTimes;
    llvm::DenseMap<mlir::Operation *, double> startTimesInCycle;
    auto unpipelineInfo = computeUnpipelineAnalysis(task, specGammas, clockPeriod, analyser, canonicalizePm, constraint,
                                                    inputLatencies, startTimes, startTimesInCycle, ctx);
    if (!unpipelineInfo)
      return llvm::failure();

    auto tmp = *unpipelineInfo;

    builder.setInsertionPointToStart(task.getBodyBlock());

    llvm::SmallVector<mlir::Attribute> gammaNames;
    llvm::SmallVector<int64_t> condDelays, fastIndices;
    llvm::SmallVector<mlir::Attribute> inputDelays;
    llvm::SmallVector<mlir::Value> mispecInputs, fsmInitialValues;
    llvm::SmallVector<std::string> mispecTypeFieldNames, fsmTypeFieldNames, fsmCmdTypeFieldNames;
    llvm::SmallVector<mlir::Type> mispecTypeFieldTypes, fsmTypeFieldTypes, fsmCmdTypeFieldTypes;
    std::string fsmName = "fsm_";
    llvm::StringMap<int> namesId;

    fsmTypeFieldNames.push_back("array_rollback");
    fsmTypeFieldTypes.push_back(builder.getI32Type());
    fsmInitialValues.push_back(
        builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));
    fsmTypeFieldNames.push_back("mu_rollback");
    fsmTypeFieldTypes.push_back(builder.getI32Type());
    fsmInitialValues.push_back(
        builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));
    fsmTypeFieldNames.push_back("rewindCpt");
    fsmTypeFieldTypes.push_back(builder.getI32Type());
    fsmInitialValues.push_back(
        builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));
    fsmTypeFieldNames.push_back("rewind");
    fsmTypeFieldTypes.push_back(builder.getI32Type());
    fsmInitialValues.push_back(
        builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));
    fsmTypeFieldNames.push_back("state");
    fsmTypeFieldTypes.push_back(builder.getI32Type());
    fsmInitialValues.push_back(
        builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));
    fsmTypeFieldNames.push_back("rbwe");
    fsmTypeFieldTypes.push_back(builder.getI1Type());
    fsmInitialValues.push_back(
        builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI1Type(), 1));
    fsmTypeFieldNames.push_back("rewindDepth");
    fsmTypeFieldTypes.push_back(builder.getI32Type());
    fsmInitialValues.push_back(
        builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));

    int maxCond = 0;
    for (auto &[gamma, resolve] : resolveDelays) {
      maxCond = std::max(maxCond, resolve);
    }

    for (int i = 0; i < maxCond; ++i) {
      fsmTypeFieldNames.push_back("delayed_commit_" + std::to_string(i));
      fsmTypeFieldTypes.push_back(builder.getI1Type());
      fsmInitialValues.push_back(
          builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI1Type(), 0));
    }

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
      gammaNames.push_back(builder.getStringAttr(name));
      mispecTypeFieldNames.push_back("mispec_" + name);
      fsmName += name;

      fsmTypeFieldNames.push_back("commit_" + name);
      fsmTypeFieldTypes.push_back(builder.getI1Type());
      fsmInitialValues.push_back(
          builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI1Type(), 0));
      fsmTypeFieldNames.push_back("selSlowPath_" + name);
      fsmTypeFieldTypes.push_back(builder.getI32Type());
      fsmInitialValues.push_back(
          builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), specGammas[g]));
      fsmTypeFieldNames.push_back("rollback_" + name);
      fsmTypeFieldTypes.push_back(builder.getI32Type());
      fsmInitialValues.push_back(
          builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));
      fsmTypeFieldNames.push_back("slowPath_" + name);
      fsmTypeFieldTypes.push_back(builder.getI32Type());
      fsmInitialValues.push_back(
          builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));

      fsmCmdTypeFieldNames.push_back("gammaRollBack_" + name);
      fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
      fsmCmdTypeFieldNames.push_back("selSlowPath_" + name);
      fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
      for (unsigned index = 0; index < g.getInputs().size(); ++index) {
        fsmCmdTypeFieldNames.push_back("stall_" + name + "_" + std::to_string(index));
        fsmCmdTypeFieldTypes.push_back(builder.getI32Type());
        fsmTypeFieldNames.push_back("stall_" + name + "_" + std::to_string(index));
        fsmTypeFieldTypes.push_back(builder.getI32Type());
        fsmInitialValues.push_back(
            builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));
      }

      condDelays.push_back(resolveDelays[g]);
      fastIndices.push_back(specGammas[g]);
      llvm::SmallVector<mlir::Attribute> inputs;
      for (auto lat : inputLatencies[g]) {
        inputs.push_back(builder.getIntegerAttr(builder.getI64Type(), lat));
      }
      inputDelays.push_back(builder.getArrayAttr(inputs));
      mispecInputs.push_back(g.getSelect());
      mispecTypeFieldTypes.push_back(g.getSelect().getType());
    }
    auto mispecType = spechls::StructType::get(ctx, fsmName + "_mispec", mispecTypeFieldNames, mispecTypeFieldTypes);
    auto mispecPack = builder.create<spechls::PackOp>(mlir::UnknownLoc::get(ctx), mispecType, mispecInputs);
    auto fsmType = spechls::StructType::get(ctx, fsmName + "_state", fsmTypeFieldNames, fsmTypeFieldTypes);
    auto fsmInit = builder.create<spechls::PackOp>(mlir::UnknownLoc::get(ctx), fsmType, fsmInitialValues);

    auto fsmStateMu = builder.create<spechls::MuOp>(mlir::UnknownLoc::get(ctx),
                                                    builder.getStringAttr(fsmName + "State"), fsmInit, fsmInit);
    builder.create<spechls::DelayOp>(mlir::UnknownLoc::get(ctx), fsmType, fsmInit, 1, nullptr, fsmInit);

    auto fsm = builder.create<spechls::FSMOp>(
        mlir::UnknownLoc::get(ctx), fsmType, builder.getStringAttr(fsmName), builder.getArrayAttr(gammaNames),
        builder.getDenseI64ArrayAttr(condDelays), builder.getDenseI64ArrayAttr(fastIndices),
        builder.getArrayAttr(inputDelays), mispecPack, fsmStateMu);

    fsmStateMu.getLoopValueMutable().assign(fsm.getResult());

    auto fsmCmdType = spechls::StructType::get(ctx, fsmName + "_cmdType", fsmCmdTypeFieldNames, fsmCmdTypeFieldTypes);

    auto fsmCmd = builder.create<spechls::FSMCommandOp>(mlir::UnknownLoc::get(ctx), fsmCmdType,
                                                        builder.getStringAttr(fsmName), fsmStateMu.getResult());

    auto nextInputField = builder.create<spechls::FieldOp>(mlir::UnknownLoc::get(ctx), "nextInput", fsmCmd.getResult());
    auto commitField = builder.create<spechls::FieldOp>(mlir::UnknownLoc::get(ctx), "commit", fsmCmd.getResult());
    auto muRollBackField =
        builder.create<spechls::FieldOp>(mlir::UnknownLoc::get(ctx), "muRollBack", fsmCmd.getResult());
    auto arrayRollBackField =
        builder.create<spechls::FieldOp>(mlir::UnknownLoc::get(ctx), "arrayRollBack", fsmCmd.getResult());
    auto rewindField = builder.create<spechls::FieldOp>(mlir::UnknownLoc::get(ctx), "rewind", fsmCmd.getResult());
    auto rbweField = builder.create<spechls::FieldOp>(mlir::UnknownLoc::get(ctx), "rbwe", fsmCmd.getResult());

    llvm::DenseMap<spechls::GammaOp, llvm::SmallVector<spechls::FieldOp>> gammaStallFields;
    for (unsigned id = 0; id < sortedGammaNodes.size(); ++id) {
      auto gamma = sortedGammaNodes[id];
      auto name = llvm::dyn_cast<mlir::StringAttr>(gammaNames[id]).str();
      llvm::SmallVector<spechls::FieldOp> fields;
      for (unsigned i = 0; i < gamma.getInputs().size(); ++i) {
        fields.push_back(builder.create<spechls::FieldOp>(
            mlir::UnknownLoc::get(ctx), "stall_" + name + "_" + std::to_string(i), fsmCmd.getResult()));
      }
      gammaStallFields.try_emplace(gamma, fields);
    }

    if (wcetModel)
      return llvm::success();

    // Add delays
    task.getBodyBlock()->walk([&](mlir::Operation *op) {
      if (!(llvm::isa<spechls::GammaOp>(op) || llvm::isa<spechls::DelayOp>(op) ||
            llvm::isa<spechls::CancellableDelayOp>(op) || llvm::isa<spechls::RollbackableDelayOp>(op) ||
            llvm::isa<spechls::MuOp>(op) || llvm::isa<spechls::CommitOp>(op))) {
        if (startTimes.contains(op) && !llvm::isa<spechls::GammaOp>(op)) {
          for (auto &opOperand : op->getOpOperands()) {
            if (auto *pred = opOperand.get().getDefiningOp()) {
              if (startTimes.contains(pred)) {
                unsigned expectedDelays = startTimes[op] - startTimes[pred];
                if (expectedDelays > 0) {
                  llvm::SmallVector<mlir::Value> delayConditions;
                  // Compute delays write-enable conditions
                  delayConditions.resize(expectedDelays);
                  bool first = true;
                  for (auto &[gamma, inputsInfo] : unpipelineInfo->operationsCoordinate) {
                    for (unsigned input = 0; input < gamma.getInputs().size(); ++input) {
                      if (inputsInfo[input].contains(pred) && inputsInfo[input].contains(op)) {
                        auto predInfo = inputsInfo[input][pred];
                        auto opInfo = inputsInfo[input][op];
                        int pipelineDelayId;
                        for (pipelineDelayId = 0; pipelineDelayId < (static_cast<int>(opInfo.pipelineStage) -
                                                                     static_cast<int>(predInfo.pipelineStage));
                             ++pipelineDelayId) {
                          mlir::Value cond = builder.create<circt::comb::ICmpOp>(
                              mlir::UnknownLoc::get(ctx), circt::comb::ICmpPredicate::eq,
                              gammaStallFields[gamma][input],
                              builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(),
                                                                    0));
                          if (first) {
                            delayConditions[pipelineDelayId] = cond;
                          } else {
                            delayConditions[pipelineDelayId] = builder.create<circt::comb::OrOp>(
                                mlir::UnknownLoc::get(ctx), delayConditions[pipelineDelayId], cond);
                          }
                        }
                        for (int unpipelineDelayId = 0;
                             unpipelineDelayId <
                             (static_cast<int>(opInfo.unpipelineStage) - static_cast<int>(predInfo.unpipelineStage));
                             ++unpipelineDelayId) {
                          mlir::Value cond = builder.create<circt::comb::ICmpOp>(
                              mlir::UnknownLoc::get(ctx), circt::comb::ICmpPredicate::eq,
                              gammaStallFields[gamma][input],
                              builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(),
                                                                    unpipelineDelayId + predInfo.unpipelineStage));
                          if (first) {
                            delayConditions[pipelineDelayId + unpipelineDelayId] = cond;
                          } else {
                            delayConditions[pipelineDelayId + unpipelineDelayId] = builder.create<circt::comb::OrOp>(
                                mlir::UnknownLoc::get(ctx), delayConditions[pipelineDelayId + unpipelineDelayId], cond);
                          }
                        }
                      } else {
                        mlir::Value allCond = builder.create<circt::comb::ICmpOp>(
                            mlir::UnknownLoc::get(ctx), circt::comb::ICmpPredicate::eq, gammaStallFields[gamma][input],
                            builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI32Type(), 0));
                        if (first) {
                          for (unsigned idx = 0; idx < expectedDelays; ++idx) {
                            delayConditions[idx] = allCond;
                          }
                        } else {
                          for (unsigned idx = 0; idx < expectedDelays; ++idx) {
                            delayConditions[idx] = builder.create<circt::comb::OrOp>(mlir::UnknownLoc::get(ctx),
                                                                                     delayConditions[idx], allCond);
                          }
                        }
                      }
                    }
                    first = false;
                  }

                  mlir::Value current = opOperand.get();
                  auto type = current.getType();
                  for (auto we : delayConditions) {
                    current =
                        builder.create<spechls::DelayOp>(mlir::UnknownLoc::get(ctx), type, current, 1, we, nullptr);
                  }
                  opOperand.assign(current);
                }
              }
            }
          }
        }
      }
    });
    return llvm::success();

    llvm::unique(rollbackDepths);

    // Rewire gamma condition and add gamma rollbacks
    for (auto gamma : sortedGammaNodes) {

      gamma.getSelectMutable().assign(
          builder.create<spechls::FieldOp>(mlir::UnknownLoc::get(ctx), "selSlowPath_", fsmCmd.getResult()));
      auto rb = builder.create<spechls::RollbackOp>(
          mlir::UnknownLoc::get(ctx), builder.getDenseI64ArrayAttr(rollbackDepths), 0, gamma.getResult(),
          builder.create<spechls::FieldOp>(mlir::UnknownLoc::get(ctx), "gammaRollBack_", fsmCmd.getResult()),
          rbweField);
      gamma.getResult().replaceAllUsesExcept(rb.getResult(), rb);
    }

    // Rewire rollbackable/cancellable delays
    task.getBodyBlock()->walk(
        [&](spechls::RollbackableDelayOp delay) { delay.getRollbackMutable().assign(arrayRollBackField); });
    task.getBodyBlock()->walk(
        [&](spechls::CancellableDelayOp delay) { delay.getCancelMutable().assign(arrayRollBackField); });

    // Add mu rollbacks
    task.getBodyBlock()->walk([&](spechls::MuOp mu) {
      auto rb =
          builder.create<spechls::RollbackOp>(mlir::UnknownLoc::get(ctx), builder.getDenseI64ArrayAttr(rollbackDepths),
                                              0, mu.getResult(), muRollBackField, rbweField);
      mu.getResult().replaceAllUsesExcept(rb.getResult(), rb);
    });

    // Add delays before commits and rewire commit condition
    int maxResolveDelay = 0;
    for (int resolve : resolveDelays.values()) {
      maxResolveDelay = (maxResolveDelay < resolve) ? resolve : maxResolveDelay;
    }
    auto commit = llvm::dyn_cast<spechls::CommitOp>(task.getBodyBlock()->getTerminator());
    commit.setOperand(0, commitField);
    if (maxResolveDelay > 0) {
      for (unsigned i = 1; i < commit->getNumOperands(); ++i) {
        auto op = commit.getOperand(i);
        commit.setOperand(
            i, builder.create<spechls::DelayOp>(
                   mlir::UnknownLoc::get(ctx), op.getType(), op, maxResolveDelay,
                   builder.create<circt::hw::ConstantOp>(mlir::UnknownLoc::get(ctx), builder.getI1Type(), 1), nullptr));
      }
    }

    llvm::unique(rewindDepths);

    // Insert rewind operations
    for (auto arg : task.getBodyBlock()->getArguments()) {
      auto rewind = builder.create<spechls::RewindOp>(mlir::UnknownLoc::get(ctx), arg.getType(),
                                                      builder.getDenseI64ArrayAttr(rewindDepths), arg, rewindField,
                                                      nextInputField);
      arg.replaceAllUsesExcept(rewind.getResult(), rewind);
    }
    return llvm::success();
  }
};

} // namespace