//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/Wcet/LinearWcetAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "Support/WcetUtils.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>

using namespace mlir;
namespace llvm {
template <>
struct DenseMapInfo<wcet::state> {
  static inline wcet::state getEmptyKey() { return wcet::StateStruct(-1, 0, {}); }

  static inline wcet::state getTombstoneKey() { return wcet::StateStruct(-2, 0, {}); }

  static unsigned getHashValue(const wcet::state &key) {
    return (hash_value(key.layers)) ^ (hash_value(key.pen) << 1) ^ (hash_value(key.st.size()) << 2);
  }

  static bool isEqual(const wcet::state &lhs, const wcet::state &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace {

int64_t retrieveWcet(spechls::FSMOp &fsm) {
  auto packOp = dyn_cast_or_null<spechls::PackOp>(fsm.getMispec().getDefiningOp());
  auto penArr = fsm.getInputDelays();
  if (!packOp)
    return 0;

  int64_t wcet = 0;

  for (auto in : llvm::enumerate(packOp.getInputs())) {
    auto inPen = cast<mlir::ArrayAttr>(penArr[in.index()]);
    auto inOp = in.value().getDefiningOp<circt::hw::ConstantOp>();
    int64_t max = 0;
    if (!inOp) {
      for (auto attr : inPen) {
        int64_t current = cast<IntegerAttr>(attr).getInt();
        if (current > max)
          max = current;
      }
    } else {
      auto idx = inOp.getValueAttr().getInt();
      max = cast<IntegerAttr>(inPen[idx]).getInt();
    }
    wcet += max;
  }

  return wcet;
}

std::optional<wcet::state> stateAnalysis(IRRewriter &rewriter, wcet::state &state, SmallVector<Type> &stTypes,
                                         wcet::CoreOp &analyzedCore) {
  ModuleOp mod = analyzedCore->getParentOfType<ModuleOp>();
  auto core = createAnalyseCore(rewriter, mod, analyzedCore, state.st, stTypes);

  auto pm = PassManager::on<ModuleOp>(mod->getContext());
  pm.addPass(wcet::createInlineCorePass());
  if (failed(pm.run(mod))) {
    llvm::errs() << "inlining failed\n";
    return {};
  }

  auto top = ModuleOp::create(rewriter.getUnknownLoc());
  rewriter.setInsertionPointToEnd(top.getBody());
  auto newCore = cast<wcet::CoreOp>(rewriter.clone(*core));
  core->erase();

  auto topPm = mlir::PassManager::on<mlir::ModuleOp>(top->getContext());
  topPm.addPass(createCanonicalizerPass());
  if (failed(topPm.run(top))) {
    llvm::errs() << "canonicalize failed\n";
    return {};
  }

  spechls::FSMOp fsm;
  newCore->walk([&](spechls::FSMOp f) { fsm = f; });
  int64_t currentWcet = retrieveWcet(fsm);
  wcet::DummyOp lastDum = nullptr;
  for (auto d : newCore.getOps<wcet::DummyOp>()) {
    if (!d->hasAttr("wcet.next"))
      continue;
    lastDum = d;
    break;
  }
  if (!lastDum) {
    llvm::errs() << "no last Dum\n";
    return {};
  }

  SmallVector<std::optional<IntegerAttr>> dumResult;
  for (auto d : lastDum.getInputs()) {
    auto lastResultOp = dyn_cast_or_null<circt::hw::ConstantOp>(d.getDefiningOp());
    if (lastResultOp) {
      dumResult.push_back(lastResultOp.getValueAttr());
      continue;
    }
    dumResult.push_back({});
  }

  SmallVector<std::optional<IntegerAttr>> st;

  st = generateNextState(rewriter, analyzedCore, stTypes, dumResult, currentWcet);

  wcet::state res = wcet::StateStruct(currentWcet, state.layers + 1, st);
  // top->dumpPretty();
  top->erase();
  return res;
}

} // namespace

namespace wcet {

LinearAnalysis::LinearAnalysis(ModuleOp mod, SmallVector<size_t> instrs) {
  wcet = 0;
  size_t step = 4;
  ModuleOp workingMod = mod.clone();
  DenseMap<state, SmallVector<state>> succs;
  DenseMap<state, SmallVector<state>> preds;
  IRRewriter rewriter(workingMod->getContext());

  // Retrieve the wcet core to analyze
  wcet::CoreOp analyzedCore = nullptr;
  workingMod->walk([&](wcet::CoreOp c) {
    if (c->hasAttr("wcet.cpuCore")) {
      analyzedCore = c;
      IntegerAttr pcStep = dyn_cast_or_null<IntegerAttr>(c->getAttr("wcet.pcStep"));
      if (pcStep)
        step = (size_t)pcStep.getInt();
    }
  });

  // Replace fetchs by array reads
  SmallVector<int64_t> i64instrs;
  for (auto i : instrs) {
    i64instrs.push_back(i);
  }

  analyzedCore->walk([&](Operation *op) {
    if (op->hasAttr("wcet.fetch")) {
      auto oldFetch = dyn_cast_or_null<spechls::LoadOp>(op);
      if (!oldFetch)
        return;
      rewriter.setInsertionPoint(oldFetch);
      wcet::ConstArrayRead newFetch = wcet::ConstArrayRead::create(
          rewriter, rewriter.getUnknownLoc(), oldFetch.getType(), oldFetch.getIndex(), i64instrs, step);
      rewriter.replaceAllOpUsesWith(oldFetch, newFetch);
      // newFetch->dumpPretty();
      rewriter.eraseOp(oldFetch);
    }
  });

  // Setup the initial state
  SmallVector<std::optional<IntegerAttr>> st;
  SmallVector<Type> stTypes;

  auto resultsType = analyzedCore.getResultTypes();
  stTypes.push_back(resultsType[0]);
  st.push_back(rewriter.getIntegerAttr(resultsType[0], 0)); // pc
  for (size_t i = 1; i < resultsType.size(); i++) {
    Type type = analyzedCore.getResultTypes()[i];
    stTypes.push_back(type);
    st.push_back({});
  }
  state initState = StateStruct(0, 0, st);
  succs.try_emplace(initState);
  preds.try_emplace(initState);

  // analysis' core
  SmallVector<state> layers = {initState};
  wcet::state cState = initState;
  DenseMap<wcet::state, int64_t> dists;
  dists[initState] = 0;
  while (cState.st.size() > 0) {
    auto nextSt = stateAnalysis(rewriter, cState, stTypes, analyzedCore);
    if (!nextSt) {
      wcet = 0;
      return;
    }
    if (nextSt == cState)
      break;
    succs[cState].push_back(nextSt.value());
    preds[nextSt.value()].push_back(cState);
    dists[nextSt.value()] = dists[cState] + nextSt.value().pen;
    wcet += nextSt.value().pen;
    if (nextSt.value().st.size() == 0) {
      cState = nextSt.value();
      break;
    }
    if ((size_t)nextSt.value().st[0].value().getInt() < (size_t)cState.st[0].value().getInt()) {
      cState = nextSt.value();
      break;
    }
    cState = nextSt.value();
    if ((size_t)nextSt.value().st[0].value().getInt() >= instrs.size() * step) {
      break;
    }
  }

  // Dump Penalty Graph
  auto lastState = StateStruct(0, cState.layers + 1, {});
  preds[lastState].push_back(cState);
  succs[cState].push_back(lastState);
  succs[lastState] = {};

  dumpGraph(succs, instrs, dists);
  workingMod->erase();
}

} // namespace wcet
