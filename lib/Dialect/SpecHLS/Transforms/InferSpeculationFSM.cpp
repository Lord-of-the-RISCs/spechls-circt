//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_INFERSPECULATIONFSMPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

namespace {

struct SlowPath {
  int64_t selector;
  int64_t latency;
};

struct Speculation {
  std::string gammaId;
  int64_t condLatency;
  int64_t fastSelector;
  llvm::SmallVector<int64_t> poisonSpeculationIds;
  llvm::SmallVector<SlowPath> slowPaths;
};

static FailureOr<llvm::SmallVector<Speculation>> parseConfig(spechls::TaskOp task) {
  auto config = task->getAttrOfType<ArrayAttr>("spechls.speculation_config");
  if (!config)
    return failure();

  llvm::SmallVector<Speculation> result;
  for (auto attribute : config) {
    auto entry = dyn_cast<DictionaryAttr>(attribute);
    auto gammaId = entry ? entry.getAs<StringAttr>("gamma_id") : nullptr;
    auto condLatency = entry ? entry.getAs<IntegerAttr>("cond_latency") : nullptr;
    auto fastSelector = entry ? entry.getAs<IntegerAttr>("fast_selector") : nullptr;
    auto poisonIds = entry ? entry.getAs<ArrayAttr>("poison_speculation_ids") : nullptr;
    auto slowPaths = entry ? entry.getAs<ArrayAttr>("slow_paths") : nullptr;
    if (!gammaId || !condLatency || !fastSelector || !poisonIds || !slowPaths)
      return failure();

    Speculation speculation{gammaId.str(), condLatency.getInt(), fastSelector.getInt(), {}, {}};
    for (auto idAttribute : poisonIds) {
      auto id = dyn_cast<IntegerAttr>(idAttribute);
      if (!id)
        return failure();
      speculation.poisonSpeculationIds.push_back(id.getInt());
    }
    for (auto pathAttribute : slowPaths) {
      auto path = dyn_cast<DictionaryAttr>(pathAttribute);
      auto selector = path ? path.getAs<IntegerAttr>("selector") : nullptr;
      auto latency = path ? path.getAs<IntegerAttr>("latency") : nullptr;
      if (!selector || !latency)
        return failure();
      speculation.slowPaths.push_back({selector.getInt(), latency.getInt()});
    }
    result.push_back(std::move(speculation));
  }
  return result;
}

struct InferSpeculationFSMPass
    : public spechls::impl::InferSpeculationFSMPassBase<InferSpeculationFSMPass> {
  void runOnOperation() override {
    auto kernel = getOperation();
    bool invalidConfig = false;
    kernel.walk([&](spechls::TaskOp task) {
      if (task->hasAttr("spechls.speculation_config") && failed(buildFSM(task)))
        invalidConfig = true;
    });
    if (invalidConfig)
      signalPassFailure();
  }

private:
  LogicalResult buildFSM(spechls::TaskOp task) {
    // Do not duplicate the controller when this pass is run after the wiring pass.
    bool hasFSM = false;
    task.getBodyBlock()->walk([&](spechls::FSMInstanceOp) { hasFSM = true; });
    if (hasFSM)
      return success();

    auto configurations = parseConfig(task);
    if (failed(configurations)) {
      task.emitError("spechls.speculation_config must contain gamma_id, cond_latency, fast_selector, "
                     "poison_speculation_ids, and slow_paths entries");
      return failure();
    }

    llvm::StringMap<spechls::GammaOp> gammas;
    task.getBodyBlock()->walk([&](spechls::GammaOp gamma) { gammas[gamma.getSymName().str()] = gamma; });
    for (const auto &configuration : *configurations) {
      if (!gammas.contains(configuration.gammaId)) {
        task.emitError("spechls.speculation_config refers to unknown gamma '")
            << configuration.gammaId << "'";
        return failure();
      }
    }

    MLIRContext *ctx = task.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(task.getBodyBlock());
    std::string name = "fsm_";
    llvm::SmallVector<Attribute> gammaNames, inputDelays;
    llvm::SmallVector<int64_t> condDelays, fastIndices;
    llvm::SmallVector<Value> mispecInputs;
    int64_t maxCondLatency = 1;

    for (const auto &configuration : *configurations) {
      auto gamma = gammas.lookup(configuration.gammaId);
      std::string gammaName = configuration.gammaId + "0";
      name += gammaName;
      gammaNames.push_back(builder.getStringAttr(gammaName));
      condDelays.push_back(configuration.condLatency);
      fastIndices.push_back(configuration.fastSelector);
      maxCondLatency = std::max(maxCondLatency, configuration.condLatency);
      mispecInputs.push_back(gamma.getSelect());

      int64_t lastSelector = configuration.fastSelector;
      for (const auto &path : configuration.slowPaths)
        lastSelector = std::max(lastSelector, path.selector);
      llvm::SmallVector<Attribute> delays(lastSelector + 1, builder.getI64IntegerAttr(0));
      for (const auto &path : configuration.slowPaths)
        delays[path.selector] = builder.getI64IntegerAttr(path.latency);
      inputDelays.push_back(builder.getArrayAttr(delays));
    }

    auto module = task->getParentOfType<mlir::ModuleOp>();
    if (SymbolTable::lookupSymbolIn(module, name))
      name += "_" + task.getSymName().str();
    auto initialState = builder.create<circt::hw::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(), 0);
    auto state = builder.create<spechls::MuOp>(builder.getUnknownLoc(), builder.getStringAttr(name + "State"),
                                               initialState, initialState);
    llvm::SmallVector<Type> machineInputs{builder.getI32Type()};
    for (auto input : mispecInputs)
      machineInputs.push_back(input.getType());
    auto machineType = builder.getFunctionType(machineInputs, ArrayRef<Type>{builder.getI32Type()});
    OpBuilder moduleBuilder(ctx);
    moduleBuilder.setInsertionPoint(task->getParentOfType<spechls::KernelOp>());
    auto fsm = moduleBuilder.create<spechls::FSMOp>(builder.getUnknownLoc(), builder.getStringAttr(name), machineType);
    builder.create<spechls::FSMInstanceOp>(builder.getUnknownLoc(), builder.getStringAttr(name),
                                           FlatSymbolRefAttr::get(ctx, name));
    llvm::SmallVector<Value> triggerInputs{state.getResult()};
    triggerInputs.append(mispecInputs);
    auto trigger = builder.create<spechls::FSMTriggerOp>(builder.getUnknownLoc(), ArrayRef<Type>{builder.getI32Type()},
                                                          builder.getStringAttr(name), triggerInputs);
    state.getLoopValueMutable().assign(trigger.getResult(0));

    auto &body = fsm.getBody();
    body.emplaceBlock();
    OpBuilder fsmBuilder(ctx);
    fsmBuilder.setInsertionPointToStart(&body.front());
    llvm::StringMap<spechls::FSMStateOp> fsmStates;
    auto addState = [&](StringRef stateName) {
      fsmBuilder.setInsertionPointToEnd(&body.front());
      auto fsmState = fsmBuilder.create<spechls::FSMStateOp>(builder.getUnknownLoc(),
                                                              fsmBuilder.getStringAttr(stateName));
      fsmState.getOutput().emplaceBlock();
      fsmBuilder.setInsertionPointToStart(&fsmState.getOutput().front());
      fsmBuilder.create<spechls::FSMOutputOp>(builder.getUnknownLoc(), fsmBuilder.getArrayAttr({}),
                                              fsmBuilder.getDenseI64ArrayAttr({}));
      fsmState.getTransitions().emplaceBlock();
      fsmStates[stateName] = fsmState;
    };
    auto addTransition = [&](StringRef source, StringRef target, ArrayRef<int64_t> inputs = {},
                              ArrayRef<int64_t> selectors = {}, StringRef kind = "normal") {
      auto sourceState = fsmStates.lookup(source);
      assert(sourceState && "FSM transition source must be created before its transition");
      fsmBuilder.setInsertionPointToEnd(&sourceState.getTransitions().front());
      auto transition = fsmBuilder.create<spechls::FSMTransitionOp>(builder.getUnknownLoc(),
                                                                      fsmBuilder.getStringAttr(target),
                                                                      fsmBuilder.getStringAttr(kind));
      if (inputs.empty())
        return;
      auto &guard = transition.getGuard();
      guard.emplaceBlock();
      fsmBuilder.setInsertionPointToStart(&guard.front());
      Value predicate;
      for (auto [input, selector] : llvm::zip_equal(inputs, selectors)) {
        auto inputType = llvm::cast<IntegerType>(machineInputs[input + 1]);
        auto port = fsmBuilder.create<spechls::FSMInputOp>(builder.getUnknownLoc(), inputType,
                                                            fsmBuilder.getI64IntegerAttr(input + 1));
        auto selected = fsmBuilder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), selector,
                                                                 inputType.getWidth());
        auto match = fsmBuilder.create<arith::CmpIOp>(builder.getUnknownLoc(), arith::CmpIPredicate::eq, port,
                                                       selected);
        if (predicate)
          predicate = fsmBuilder.create<circt::comb::AndOp>(builder.getUnknownLoc(), predicate,
                                                             match.getResult()).getResult();
        else
          predicate = match.getResult();
      }
      fsmBuilder.create<spechls::FSMReturnOp>(builder.getUnknownLoc(), predicate);
    };
    for (int64_t index = 0; index < maxCondLatency; ++index) {
      auto current = "Init_" + std::to_string(index);
      addState(current);
      addTransition(current, index + 1 == maxCondLatency ? "Proceed" : "Init_" + std::to_string(index + 1));
    }
    addState("Proceed");
    addTransition("Proceed", "Proceed");
    struct Path {
      unsigned gammaIndex;
      unsigned pathIndex;
      int64_t selector;
    };
    llvm::SmallVector<llvm::SmallVector<Path>> alternatives(configurations->size());
    for (unsigned gammaIndex = 0; gammaIndex < configurations->size(); ++gammaIndex)
      for (unsigned pathIndex = 0; pathIndex < (*configurations)[gammaIndex].slowPaths.size(); ++pathIndex)
        alternatives[gammaIndex].push_back(
            {gammaIndex, pathIndex, (*configurations)[gammaIndex].slowPaths[pathIndex].selector});

    auto orderedBefore = [&](const Path &left, const Path &right) {
      const auto &leftPoison = (*configurations)[left.gammaIndex].poisonSpeculationIds;
      const auto &rightPoison = (*configurations)[right.gammaIndex].poisonSpeculationIds;
      if (llvm::is_contained(leftPoison, static_cast<int64_t>(right.gammaIndex)))
        return true;
      if (llvm::is_contained(rightPoison, static_cast<int64_t>(left.gammaIndex)))
        return false;
      if ((*configurations)[left.gammaIndex].condLatency != (*configurations)[right.gammaIndex].condLatency)
        return (*configurations)[left.gammaIndex].condLatency < (*configurations)[right.gammaIndex].condLatency;
      return left.gammaIndex < right.gammaIndex;
    };
    llvm::SmallVector<llvm::SmallVector<Path>> plans(1);
    for (const auto &paths : alternatives) {
      auto previous = plans;
      for (const auto &plan : previous)
        for (const auto &path : paths) {
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
    if (plans.size() > 257) {
      task.emitError("spechls.speculation_config produces more than 256 recovery plans");
      return failure();
    }
    auto planKey = [](ArrayRef<Path> plan) {
      std::string key;
      for (const auto &path : plan)
        key += std::to_string(path.gammaIndex) + ":" + std::to_string(path.selector) + ";";
      return key;
    };
    llvm::StringMap<std::string> planStarts;
    llvm::SmallVector<std::string> recoveryStates;
    unsigned combinedIndex = 0;
    for (const auto &plan : plans) {
      if (plan.empty())
        continue;
      auto prefix = plan.size() == 1 ? std::to_string(plan.front().gammaIndex) + "_" +
                                           std::to_string(plan.front().pathIndex)
                                     : "Combined_" + std::to_string(combinedIndex++);
      auto rollback = prefix + "_Rollback";
      addState(rollback);
      llvm::SmallVector<int64_t> inputs, selectors;
      for (const auto &path : plan) {
        inputs.push_back(path.gammaIndex);
        selectors.push_back(path.selector);
      }
      addTransition("Proceed", rollback, inputs, selectors);
      addTransition(rollback, "Proceed");
      planStarts[planKey(plan)] = rollback;
      recoveryStates.push_back(rollback);
    }
    unsigned newMispecIndex = 0;
    for (const auto &plan : plans) {
      if (plan.empty())
        continue;
      for (const auto &paths : alternatives)
        for (const auto &path : paths) {
          if (llvm::any_of(plan, [&](const Path &active) { return active.gammaIndex == path.gammaIndex; }))
            continue;
          auto destinationPlan = plan;
          destinationPlan.push_back(path);
          llvm::sort(destinationPlan, orderedBefore);
          auto destination = planStarts.lookup(planKey(destinationPlan));
          if (destination.empty())
            continue;
          auto newMispec = "NewMispec_" + std::to_string(newMispecIndex++);
          addState(newMispec);
          addTransition(newMispec, destination);
          for (const auto &recovery : recoveryStates) {
            addTransition(recovery, newMispec, {static_cast<int64_t>(path.gammaIndex)}, {path.selector}, "new_mispec");
            addTransition(recovery, destination, {static_cast<int64_t>(path.gammaIndex)}, {path.selector}, "canceled");
          }
        }
    }
    return success();
  }
};

} // namespace
