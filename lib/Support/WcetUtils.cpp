//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Support/WcetUtils.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <fstream>

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
void dumpGraph(DenseMap<wcet::state, SmallVector<wcet::state>> outs, SmallVector<size_t> instrs,
               DenseMap<wcet::state, int64_t> &dists) {
  size_t instrsHash = 0;
  for (auto i : instrs) {
    instrsHash = instrsHash ^ i;
  }
  std::string outputFileName = "/tmp/graph_" + std::to_string(instrsHash) + ".dot";

  DenseMap<size_t, SmallVector<wcet::state>> statesByLayer;
  DenseMap<wcet::state, size_t> statesId;
  size_t id = 0;
  for (auto [st, outs] : outs) {
    statesByLayer[st.layers].push_back(st);
    statesId[st] = id++;
  }

  std::ofstream dot(outputFileName);
  dot << "digraph G {\n";
  dot << "  rankdir=LR;\n  node [shape=circle];\n  edge [constraint=true];\n  graph [rankstep=1.2, nodestep=0.6];\n";

  for (auto &[layer, states] : statesByLayer) {
    dot << "{ rank=same\n";
    for (auto st : states) {
      dot << "n" << statesId[st] << " [label=\""
          << "pen=" << st.pen << "\ndist=" << dists[st] << "\"];\n";
    }
    if (layer == 0 || layer >= instrs.size() + 1) {
      dot << "}\n";
      continue;
    }
    dot << "l" << layer << " [shape=plaintext, label=\"" << std::hex << instrs[layer - 1] << std::dec << "\"];\n";

    dot << "}\n\n";
  }

  for (auto &[src, out] : outs) {
    for (auto &dst : out) {
      dot << " n" << statesId[src] << " -> n" << statesId[dst] << ";\n";
    }
  }

  dot << "}\n";
  dot.close();
}

wcet::CoreOp createAnalyseCore(mlir::IRRewriter &rewriter, mlir::ModuleOp &top, wcet::CoreOp &analyzedCore,
                               mlir::SmallVector<std::optional<mlir::IntegerAttr>> &state,
                               mlir::SmallVector<mlir::Type> &types) {

  rewriter.setInsertionPointToEnd(top.getBody());
  wcet::CoreOp result = rewriter.create<wcet::CoreOp>(rewriter.getUnknownLoc(), rewriter.getFunctionType({}, {}),
                                                      rewriter.getStringAttr(CORE_ANALYSIS_NAME));
  result->setAttr("wcet.analysis", rewriter.getUnitAttr());
  rewriter.setInsertionPointToEnd(&result.getBody().front());

  mlir::SmallVector<mlir::Value> coreInputs;

  mlir::SmallVector<mlir::Value> dummyInputs;
  for (auto st : llvm::enumerate(state)) {
    if (st.value().has_value()) {
      dummyInputs.push_back(
          rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), types[st.index()], st.value().value()));
    } else {
      dummyInputs.push_back(rewriter.create<wcet::InitOp>(rewriter.getUnknownLoc(), types[st.index()], "Unknown"));
    }
  }

  auto firstDummy = rewriter.create<wcet::DummyOp>(rewriter.getUnknownLoc(), types, dummyInputs);
  firstDummy->setAttr("wcet.current", rewriter.getUnitAttr());
  for (auto out : firstDummy.getOutputs()) {
    coreInputs.push_back(out);
  }

  auto coreInstance = rewriter.create<wcet::CoreInstanceOp>(rewriter.getUnknownLoc(), analyzedCore, coreInputs);

  auto secondDummy = rewriter.create<wcet::DummyOp>(rewriter.getUnknownLoc(), coreInstance->getResultTypes(),
                                                    coreInstance->getResults());
  secondDummy->setAttr("wcet.next", rewriter.getUnitAttr());

  rewriter.create<wcet::CommitOp>(rewriter.getUnknownLoc(), result->getResultTypes(), mlir::ValueRange());
  // result->dumpPretty();
  return result;
}

SmallVector<int64_t> retrieveMultWcet(spechls::FSMOp &fsm) {
  int64_t gmv = 0;
  SmallVector<int64_t> result;
  auto packOp = dyn_cast_or_null<spechls::PackOp>(fsm.getMispec().getDefiningOp());
  auto penArr = fsm.getInputDelays();
  DenseSet<int64_t> tmpMap;
  SmallVector<ArrayAttr> nonConstant;
  if (!packOp)
    return result;
  for (auto in : llvm::enumerate(packOp.getInputs())) {
    auto inPen = cast<mlir::ArrayAttr>(penArr[in.index()]);
    auto inOp = in.value().getDefiningOp<circt::hw::ConstantOp>();
    int64_t max = 0;
    if (!inOp) {
      nonConstant.push_back(inPen);
      continue;
    }
    auto idx = inOp.getValueAttr().getInt();
    max = cast<IntegerAttr>(inPen[idx]).getInt();
    //    llvm::errs() << "pen: " << max << "\n";
    gmv += max;
  }

  tmpMap.insert(gmv);
  for (auto arr : nonConstant) {
    DenseSet<int64_t> newTmpMap;
    for (auto old : tmpMap) {
      for (auto uC : arr) {
        newTmpMap.insert(old + cast<IntegerAttr>(uC).getInt());
      }
    }
    tmpMap = newTmpMap;
  }
  return SmallVector<int64_t>(tmpMap.begin(), tmpMap.end());
}

SmallVector<std::optional<IntegerAttr>> generateNextState(IRRewriter &rewriter, wcet::CoreOp &analyzedCore,
                                                          SmallVector<Type> &stTypes,
                                                          SmallVector<std::optional<IntegerAttr>> &dumResult,
                                                          int64_t pen) {
  if (!dumResult[0].has_value())
    return {};
  SmallVector<std::optional<IntegerAttr>> state;
  assert(analyzedCore.getResultTypes().size() == dumResult.size());
  for (size_t i = 0; i < analyzedCore.getResultTypes().size(); i++) {
    auto nbPred = dyn_cast_or_null<IntegerAttr>(analyzedCore.getArgAttr(i, "wcet.nbPred"));
    IntegerType it = dyn_cast_or_null<IntegerType>(stTypes[i]);
    if (!nbPred || nbPred.getInt() == 0) {
      if (it && it.getWidth() == 1 && !dumResult[i]) {
        state.push_back({});
        // state.push_back(rewriter.getIntegerAttr(stTypes[i], 1));
      } else {
        state.push_back(dumResult[i]);
      }
    } else if (nbPred.getInt() > (int64_t)pen) {
      state.push_back(dumResult[i - pen]);
    } else {
      if (it && it.getWidth() == 1) {
        state.push_back(rewriter.getIntegerAttr(stTypes[i], 0));
      } else {
        state.push_back({});
      }
    }
  }
  return state;
}
