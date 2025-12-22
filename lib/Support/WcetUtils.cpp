//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Support/WcetUtils.h"
#include "circt/Dialect/HW/HWOps.h"

using namespace mlir;

wcet::CoreOp createAnalyseCore(mlir::IRRewriter &rewriter, mlir::ModuleOp &top, wcet::CoreOp &analyzedCore,
                               mlir::SmallVector<std::optional<mlir::IntegerAttr>> &state,
                               mlir::SmallVector<mlir::Type> &types, size_t instrs) {

  rewriter.setInsertionPointToEnd(top.getBody());
  wcet::CoreOp result = rewriter.create<wcet::CoreOp>(rewriter.getUnknownLoc(), rewriter.getFunctionType({}, {}),
                                                      rewriter.getStringAttr(CORE_ANALYSIS_NAME));
  result->setAttr("wcet.analysis", rewriter.getUnitAttr());
  rewriter.setInsertionPointToEnd(&result.getBody().front());

  mlir::SmallVector<mlir::Value> coreInputs;
  auto instr =
      rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), analyzedCore.getArgumentTypes().front(), instrs);
  coreInputs.push_back(instr);

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
  if (!packOp)
    return result;
  for (auto in : llvm::enumerate(packOp.getInputs())) {
    auto inPen = cast<mlir::ArrayAttr>(penArr[in.index()]);
    auto inOp = in.value().getDefiningOp<circt::hw::ConstantOp>();
    int64_t max = 0;
    if (!inOp) {
      continue;
    }
    auto idx = inOp.getValueAttr().getInt();
    max = cast<IntegerAttr>(inPen[idx]).getInt();
    if (max > gmv)
      gmv = max;
  }

  tmpMap.insert(gmv);
  for (auto in : llvm::enumerate(packOp.getInputs())) {
    auto inPen = cast<mlir::ArrayAttr>(penArr[in.index()]);
    auto inOp = in.value().getDefiningOp<circt::hw::ConstantOp>();
    if (inOp) {
      continue;
    }
    for (auto attr : inPen) {
      int64_t current = cast<IntegerAttr>(attr).getInt();
      if (current > gmv)
        tmpMap.insert(current);
    }
  }

  return SmallVector<int64_t>(tmpMap.begin(), tmpMap.end());
}
