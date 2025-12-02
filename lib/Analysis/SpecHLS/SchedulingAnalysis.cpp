//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/SpecHLS/SchedulingAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include <circt/Scheduling/Algorithms.h>
#include <circt/Scheduling/Problems.h>

namespace spechls {

namespace {

llvm::SmallVector<int> defaultConfiguration(spechls::TaskOp task) {
  unsigned maxProfilingId = 0;
  task.walk([&](spechls::GammaOp gamma) {
    if (gamma->hasAttrOfType<mlir::IntegerAttr>("spechls.profilingId")) {
      unsigned id = gamma->getAttrOfType<mlir::IntegerAttr>("spechls.profilingId").getInt();
      if (id > maxProfilingId)
        maxProfilingId = id;
    }
  });
  return llvm::SmallVector<int>(maxProfilingId + 1, -1);
}

} // namespace

SchedulingAnalysis::SchedulingAnalysis(spechls::TaskOp task, double targetClock)
    : SchedulingAnalysis(task, targetClock, defaultConfiguration(task)) {}

circt::scheduling::ChainingCyclicProblem SchedulingAnalysis::constructProblem(spechls::TaskOp task, double targetClock,
                                                                              llvm::SmallVector<int> configuration) {

  // Declare operation and operator type.
  // Do not declare dependence yet as other operation may not be declared yet.
  circt::scheduling::ChainingCyclicProblem problem(task);
  task.getBodyBlock()->walk([&](mlir::Operation *op) {
    problem.insertOperation(op);
    circt::scheduling::ChainingCyclicProblem::OperatorType operatorType;
    unsigned latency = 0;
    double combDelay = 0.0;
    if (op->hasAttrOfType<mlir::FloatAttr>("spechls.combDelay")) {
      combDelay = op->getAttrOfType<mlir::FloatAttr>("spechls.combDelay").getValueAsDouble();
      while (combDelay > targetClock) {
        combDelay -= targetClock;
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

    problem.setLinkedOperatorType(op, operatorType);
  });

  // Declare dependencies
  task.getBodyBlock()->walk([&](mlir::Operation *op) {
    unsigned distance = 0;
    if (auto delay = llvm::dyn_cast<spechls::DelayOp>(op)) {
      distance = delay.getDepth();
    } else if (llvm::isa<spechls::MuOp>(op)) {
      distance = 1;
    }
    if (op->hasAttrOfType<mlir::IntegerAttr>("spechls.profilingId")) {
      int id = op->getAttrOfType<mlir::IntegerAttr>("spechls.profilingId").getInt();
      if (configuration[id] != -1) {
        auto operand = op->getOperand(configuration[id] + 1);
        if (auto *pred = operand.getDefiningOp()) {
          circt::scheduling::ChainingCyclicProblem::Dependence dependence(pred, op);
          assert(mlir::succeeded(problem.insertDependence(dependence)));
        }
        return;
      }
    }
    for (auto operand : op->getOperands()) {
      if (auto *pred = operand.getDefiningOp()) {
        circt::scheduling::ChainingCyclicProblem::Dependence dependence(pred, op);
        assert(mlir::succeeded(problem.insertDependence(dependence)));
        if (distance > 0) {
          problem.setDistance(dependence, distance);
        }
      }
    }
  });

  // Check the problem
  assert(mlir::succeeded(problem.check()));
  return problem;
}

SchedulingAnalysis::SchedulingAnalysis(spechls::TaskOp task, double targetClock, llvm::SmallVector<int> configuration) {
  auto problem = constructProblem(task, targetClock, configuration);
  // solve and verify the problem
  assert(
      mlir::succeeded(circt::scheduling::scheduleSimplex(problem, task.getBodyBlock()->getTerminator(), targetClock)));
  assert(mlir::succeeded(problem.verify()));

  // write back schedule info
  task.getBodyBlock()->walk([&](mlir::Operation *op) {
    startTime[op] = *problem.getStartTime(op);
    startTimeInCycle[op] = *problem.getStartTimeInCycle(op);
  });
  ii = *problem.getInitiationInterval();
}

} // namespace spechls
