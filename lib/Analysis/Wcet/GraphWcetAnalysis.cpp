//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/Wcet/GraphWcetAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "Support/WcetUtils.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <stack>

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

void visitedState(wcet::state &st, SmallVector<wcet::state> &visited,
                  DenseMap<wcet::state, SmallVector<wcet::state>> &outs) {
  if (std::find(visited.begin(), visited.end(), st) != visited.end())
    return;

  visited.push_back(st);

  for (auto nextSt : outs[st]) {
    visitedState(nextSt, visited, outs);
  }
}

void topologicalSort(wcet::state &st, std::stack<wcet::state> &stack, DenseMap<wcet::state, bool> &visited,
                     DenseMap<wcet::state, SmallVector<wcet::state>> outs) {
  visited[st] = true;
  for (auto suc : outs[st]) {
    if (!visited[suc]) {
      topologicalSort(suc, stack, visited, outs);
    }
  }
  for (auto suc : outs[st]) {
    stack.push(suc);
  }
}

int64_t longestPath(wcet::state &start, wcet::state &end, DenseMap<wcet::state, SmallVector<wcet::state>> outs,
                    DenseMap<wcet::state, int64_t> &dists) {
  DenseMap<wcet::state, bool> visited;
  std::stack<wcet::state> stack;
  SmallVector<wcet::state> states;
  visitedState(start, states, outs);
  for (auto st : states) {
    visited[st] = false;
    dists[st] = 0;
  }
  for (auto st : states) {
    if (!visited[st]) {
      topologicalSort(st, stack, visited, outs);
    }
  }
  while (!stack.empty()) {
    wcet::state st = stack.top();
    stack.pop();

    int64_t delay = st.pen;
    int64_t dist = dists[st];

    for (auto suc : outs[st]) {
      int64_t sucDist = dists[suc];
      if (sucDist < (dist + delay))
        dists[suc] = dist + delay;
    }
  }
  return dists[end];
}

void mergeSameState(SmallVector<wcet::state> &states, DenseMap<wcet::state, SmallVector<wcet::state>> succs,
                    DenseMap<wcet::state, SmallVector<wcet::state>> preds) {
  SmallVector<wcet::state> result;
  for (auto cSt : states) {
    wcet::state *sameState = nullptr;
    for (auto resultSt : result) {
      if (resultSt == cSt) {
        sameState = &resultSt;
        break;
      }
    }
    if (sameState) {
      preds[*sameState].append(preds[cSt]);
      preds.erase(preds.find(cSt));
    } else {
      result.push_back(cSt);
    }
  }
  states = result;
}

mlir::SmallVector<wcet::state> stateAnalysis(mlir::IRRewriter &rewriter, size_t instr, wcet::state &st,
                                             mlir::SmallVector<mlir::Type> &stTypes, wcet::CoreOp &analyzedCore) {
  auto mod = analyzedCore->getParentOfType<mlir::ModuleOp>();
  auto core = createAnalyseCore(rewriter, mod, analyzedCore, st.st, stTypes, instr);

  auto inlinePM = mlir::PassManager::on<mlir::ModuleOp>(mod->getContext());
  inlinePM.addPass(wcet::createInlineCorePass());
  if (failed(inlinePM.run(mod))) {
    return {};
  }

  auto top = mlir::ModuleOp::create(rewriter.getUnknownLoc());
  rewriter.setInsertionPointToEnd(top.getBody());
  auto newCore = mlir::cast<wcet::CoreOp>(rewriter.clone(*core));
  core->erase();

  auto topPm = mlir::PassManager::on<mlir::ModuleOp>(top->getContext());
  topPm.addPass(mlir::createCanonicalizerPass());
  if (failed(topPm.run(top))) {
    return {};
  }

  spechls::FSMOp fsm;
  newCore->walk([&](spechls::FSMOp f) { fsm = f; });
  mlir::SmallVector<int64_t> penalties = retrieveMultWcet(fsm);

  wcet::DummyOp lastDum = nullptr;
  for (auto d : newCore.getOps<wcet::DummyOp>()) {
    if (!d->hasAttr("wcet.next"))
      continue;
    lastDum = d;
    break;
  }
  if (!lastDum) {
    llvm::errs() << "instruction fail " << instr << "\n";
    return {};
  }

  mlir::SmallVector<std::optional<mlir::IntegerAttr>> dumResult;
  for (auto d : lastDum.getInputs()) {
    auto lastResultOp = dyn_cast_or_null<circt::hw::ConstantOp>(d.getDefiningOp());
    if (lastResultOp) {
      dumResult.push_back(lastResultOp.getValueAttr());
      continue;
    }
    dumResult.push_back({});
  }

  SmallVector<wcet::state> result;
  for (auto pen : penalties) {
    mlir::SmallVector<std::optional<mlir::IntegerAttr>> state =
        generateNextState(rewriter, analyzedCore, stTypes, dumResult, pen);
    result.push_back(wcet::StateStruct(pen, st.layers + 1, state));
  }
  return result;
}
} // namespace
namespace wcet {

GraphAnalysis::GraphAnalysis(mlir::ModuleOp mod, mlir::SmallVector<size_t> instrs) {
  wcet = 0;

  // llvm::errs() << instrs.size() << "\n";
  // Setup analysis
  mlir::DenseMap<state, mlir::SmallVector<state>> succs;
  mlir::DenseMap<state, mlir::SmallVector<state>> preds;
  mlir::IRRewriter rewriter(mod->getContext());

  wcet::CoreOp analyzedCore = nullptr;
  mod->walk([&](wcet::CoreOp c) {
    if (c->hasAttr("wcet.cpuCore"))
      analyzedCore = c;
  });

  mlir::SmallVector<std::optional<mlir::IntegerAttr>> st;
  mlir::SmallVector<mlir::Type> stTypes;

  for (auto type : analyzedCore.getResultTypes()) {
    stTypes.push_back(type);
    st.push_back({});
  }
  state initState = StateStruct(0, 0, st);
  succs.try_emplace(initState);
  preds.try_emplace(initState);

  mlir::SmallVector<state> layers = {initState};

  // size_t lId = 0;
  // Build the penalties graphs
  for (auto instr : instrs) {
    // llvm::errs() << lId++ << ":";
    mergeSameState(layers, succs, preds);
    SmallVector<wcet::state> nextLayers;
    for (auto state : layers) {
      // for (auto s : state.st) {
      //   llvm::errs() << " " << s;
      // }
      // llvm::errs() << "\n";
      mlir::SmallVector<wcet::state> succOfState = stateAnalysis(rewriter, instr, state, stTypes, analyzedCore);
      succs[state].append(succOfState);
      for (auto succSt : succOfState) {
        preds[succSt].push_back(state);
      }
      nextLayers.append(succOfState);
    }
    layers = nextLayers;
  }

  auto lastState = StateStruct(0, layers.front().layers + 1, {});
  preds.try_emplace(lastState, layers);
  for (auto state : layers) {
    succs[state].push_back(lastState);
  }
  succs[lastState] = {};

  // Find the longest path in the PG
  DenseMap<wcet::state, int64_t> dists;
  wcet = longestPath(initState, lastState, succs, dists);
  dumpGraph(succs, instrs, dists);
}
} // namespace wcet
