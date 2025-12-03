//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/SpecHLS/SpeculationExplorationAnalysis.h"
#include "Analysis/SpecHLS/ConfigurationExcluderAnalysis.h"
#include "Analysis/SpecHLS/MobilityAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <circt/Dialect/SSP/SSPOps.h>
#include <circt/Scheduling/Algorithms.h>
#include <circt/Scheduling/Problems.h>
#include <circt/Support/LLVM.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <fstream>
#include <queue>
#include <string>

namespace spechls {

namespace {

bool areSameConfig(llvm::ArrayRef<int> config1, llvm::ArrayRef<int> config2) {
  // It is presupposed that the length of the vector are the same.
  for (unsigned i = 0; i < config1.size(); ++i) {
    if (config1[i] != config2[i])
      return false;
  }
  return true;
}

bool areCompatibleConfig(llvm::ArrayRef<int> base, llvm::ArrayRef<int> config, llvm::ArrayRef<bool> memspecGammas) {
  // It is presupposed that the lengths of the vectors are the same.
  for (unsigned i = 0; i < base.size(); ++i) {
    if (config[i] != -1) {
      if (memspecGammas[i]) {
        if (base[i] < config[i]) {
          return false;
        }
      } else {
        if (base[i] != config[i]) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace

namespace {

circt::scheduling::ChainingCyclicProblem
constructBaseProblem(spechls::TaskOp task, double targetClock, mlir::OpBuilder &builder,
                     circt::ssp::DependenceGraphOp &graph, mlir::MLIRContext *ctx,
                     llvm::DenseMap<mlir::Operation *, circt::ssp::OperationOp> &association) {
  builder.setInsertionPointToStart(graph.getBodyBlock());
  unsigned idx = 0;
  unsigned oprId = 0;

  // Declare operation and operator type.
  // Do not declare dependence yet as other operation may not be declared yet.
  circt::scheduling::ChainingCyclicProblem problem(task);
  task.getBodyBlock()->walk([&](mlir::Operation *op) {
    llvm::SmallVector<mlir::Value> operands;
    auto sspOp = builder.create<circt::ssp::OperationOp>(op->getLoc(), 1, operands,
                                                         mlir::StringAttr::get(ctx, "op_" + std::to_string(idx++)));
    association.try_emplace(op, sspOp.getOperation());
    problem.insertOperation(sspOp);
    circt::scheduling::ChainingCyclicProblem::OperatorType operatorType(
        mlir::StringAttr::get(ctx, "opr_" + std::to_string(oprId++)));
    unsigned latency = 0;
    double combDelay = 0.0;
    if (op->hasAttrOfType<mlir::FloatAttr>("spechls.combDelay")) {
      combDelay = op->getAttrOfType<mlir::FloatAttr>("spechls.combDelay").getValueAsDouble();
      while (combDelay >= targetClock) {
        combDelay -= targetClock;
        ++latency;
      }
    }
    sspOp->setAttr("lat", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), latency));
    sspOp->setAttr("in", mlir::FloatAttr::get(mlir::Float32Type::get(ctx), combDelay));
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

  // Declare dependencies
  task.getBodyBlock()->walk([&](mlir::Operation *op) {
    auto sspOp = association[op];
    if (!llvm::isa<spechls::GammaOp>(op)) {
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
    } else {
      sspOp->setAttr("isGamma", mlir::UnitAttr::get(ctx));
    }
  });
  return problem;
}

bool hasUnitII(llvm::ArrayRef<int> configuration, mlir::Operation *terminator,
               circt::scheduling::ChainingCyclicProblem &problem, llvm::ArrayRef<spechls::GammaOp> gammas,
               double targetClock, llvm::DenseMap<mlir::Operation *, circt::ssp::OperationOp> &association,
               circt::ssp::DependenceGraphOp &graph) {

  // insert gammas inputs according to configuration
  for (unsigned i = 0; i < gammas.size(); ++i) {
    auto gamma = gammas[i];
    auto sspOp = association[gamma];
    llvm::SmallVector<mlir::Value> operands;
    if (configuration[i] == 0) {
      for (auto operand : gamma->getOperands()) {
        if (auto *op = operand.getDefiningOp()) {
          operands.push_back(association[op].getResult(0));
        }
      }
    } else {
      if (auto *op = gamma->getOperand(configuration[i]).getDefiningOp()) {
        operands.push_back(association[op].getResult(0));
      }
    }
    sspOp.getOperandsMutable().append(operands);
  }

  // solve and verify the problem
  assert(mlir::succeeded(circt::scheduling::scheduleSimplex(problem, terminator, targetClock)));
  assert(mlir::succeeded(problem.verify()));

  // reset gammas inputs
  for (auto gamma : gammas) {
    association[gamma].getOperandsMutable().clear();
  }

  return problem.getInitiationInterval() == 1;
}

}; // namespace

SpeculationExplorationAnalysis::SpeculationExplorationAnalysis(spechls::TaskOp task, double targetClock,
                                                               double probabilityThreshold, std::string traceFileName) {

  MobilityAnalysis mobility(task, targetClock);

  mlir::MLIRContext *ctx = task->getContext();
  llvm::SmallVector<spechls::GammaOp> gammas(mobility.mobilities.keys());
  llvm::SmallVector<bool> memspecGammas;

  std::sort(gammas.begin(), gammas.end(), [&](spechls::GammaOp &g1, spechls::GammaOp &g2) {
    return mobility.mobilities[g1] < mobility.mobilities[g2];
  });

  unsigned numGamma = gammas.size();
  unsigned index = 0;

  for (auto &gamma : gammas) {
    unsigned profilingId = gamma->getAttrOfType<mlir::IntegerAttr>("spechls.profilingId").getInt();
    if (pidToEid.size() < (profilingId + 1)) {
      pidToEid.resize(profilingId + 1);
    }
    memspecGammas.push_back(gamma->hasAttr("spechls.memspec"));
    pidToEid[profilingId] = index++;
    eidToPid.push_back(profilingId);
  }

  struct ProbabilityInformation {
    llvm::DenseMap<llvm::SmallVector<int>, int> configurations;
    unsigned numConfiguration = 0;

    void addConfiguration(llvm::ArrayRef<int> configuration, unsigned numGamma, llvm::ArrayRef<int> eidToPid) {
      assert(configuration.size() >= numGamma);
      ++numConfiguration;
      llvm::SmallVector<int> translatedConfig(numGamma);
      for (unsigned i = 0; i < numGamma; ++i) {
        translatedConfig[i] = configuration[eidToPid[i]];
      }
      for (auto &[k, v] : configurations) {
        if (areSameConfig(k, translatedConfig)) {
          ++v;
          return;
        }
      }
      configurations.try_emplace(translatedConfig, 1);
    }
    double getProbability(llvm::ArrayRef<int> configuration, llvm::ArrayRef<bool> memspecGammas) {
      if (numConfiguration == 0)
        return 1.0;
      unsigned count = 0;
      for (auto &[k, v] : configurations) {
        if (areCompatibleConfig(k, configuration, memspecGammas))
          count += v;
      }
      return static_cast<double>(count) / static_cast<double>(numConfiguration);
    }
  };
  ProbabilityInformation probaInfo;

  if (traceFileName != "") {
    std::ifstream traceFile(traceFileName);
    assert(traceFile.is_open());
    std::string traceLine;
    while (std::getline(traceFile, traceLine)) {
      llvm::SmallVector<int> config;
      std::string assoc;
      std::stringstream lineStream(traceLine);
      while (std::getline(lineStream, assoc, ',')) {
        config.push_back(std::stoul(assoc) + 1);
      }
      probaInfo.addConfiguration(config, numGamma, eidToPid);
    }
    traceFile.close();
  }

  mlir::OpBuilder builder(ctx);
  llvm::DenseMap<mlir::Operation *, circt::ssp::OperationOp> association;
  auto graph = builder.create<circt::ssp::DependenceGraphOp>(task->getLoc());

  auto baseSchedulingProblem = constructBaseProblem(task, targetClock, builder, graph, ctx, association);

  // We create a default configuration. We force to speculate on memspec gammas.
  llvm::SmallVector<int> defaultConfiguration(numGamma, 0);
  for (unsigned index = 0; index < numGamma; ++index) {
    auto &gamma = gammas[index];
    if (gamma->hasAttr("spechls.memspec")) {
      defaultConfiguration[index] = gamma->getNumOperands() - 1;
    }
  }

  struct ConfigurationType {
    llvm::SmallVector<int> config;
    unsigned int lastGamma;
    unsigned int numSpeculation;
    double proba;
  };

  struct ConfigurationComparator {
    bool operator()(ConfigurationType &config1, ConfigurationType &config2) {
      // A config is smaller than the other one if it has less speculations.
      // For configs with the same number of speculation, they are sorted by probability.
      return ((config1.numSpeculation < config2.numSpeculation) ||
              ((config1.numSpeculation == config2.numSpeculation) && (config1.proba < config2.proba)));
    }
  };
  mlir::Operation *terminator = association[task.getBodyBlock()->getTerminator()];

  if (hasUnitII(defaultConfiguration, terminator, baseSchedulingProblem, gammas, targetClock, association, graph)) {
    this->proba = probaInfo.getProbability(defaultConfiguration, memspecGammas);
    this->configuration = std::move(defaultConfiguration);
    return;
  }
  std::priority_queue<ConfigurationType, llvm::SmallVector<ConfigurationType>, ConfigurationComparator> configs,
      resultConfigs;

  for (unsigned index = 0; index < numGamma; ++index) {
    if (defaultConfiguration[index] == 0) {
      for (unsigned input = 1; input < gammas[index]->getNumOperands(); ++input) {
        llvm::SmallVector<int> config(defaultConfiguration.begin(), defaultConfiguration.end());
        config[index] = input;
        double proba = probaInfo.getProbability(config, memspecGammas);
        if (proba >= probabilityThreshold) {
          if (hasUnitII(config, terminator, baseSchedulingProblem, gammas, targetClock, association, graph)) {
            resultConfigs.push(ConfigurationType{
                .config = std::move(config), .lastGamma = index, .numSpeculation = 1, .proba = proba});
          } else {
            if (!ConfigurationExcluderAnalysis(task, targetClock, config, gammas).deadEnd)
              configs.push(ConfigurationType{
                  .config = std::move(config), .lastGamma = index, .numSpeculation = 1, .proba = proba});
          }
        }
      }
    }
  }

  if (!resultConfigs.empty()) {
    this->proba = resultConfigs.top().proba;
    this->configuration = std::move(resultConfigs.top().config);
    return;
  }

  while (!configs.empty()) {
    auto config = configs.top();
    configs.pop();

    for (unsigned index = config.lastGamma + 1; index < numGamma; ++index) {
      if (defaultConfiguration[index] == 0) {
        llvm::SmallVector<int> newConfig(config.config);
        for (unsigned input = 1; input < gammas[index].getNumOperands(); ++input) {
          newConfig[index] = input;
          double proba = probaInfo.getProbability(newConfig, memspecGammas);
          if (proba >= probabilityThreshold) {
            if (hasUnitII(newConfig, terminator, baseSchedulingProblem, gammas, targetClock, association, graph)) {
              this->proba = proba;
              this->configuration = std::move(newConfig);
              return;
            }

            if (!ConfigurationExcluderAnalysis(task, targetClock, newConfig, gammas).deadEnd)
              configs.push(ConfigurationType{.config = std::move(newConfig),
                                             .lastGamma = index,
                                             .numSpeculation = config.numSpeculation + 1,
                                             .proba = proba});
          }
        }
      }
    }
  }
  llvm::errs() << "No configuration found.\n";
  this->configuration = llvm::SmallVector<int>();
  this->proba = -1.0;
}

} // namespace spechls