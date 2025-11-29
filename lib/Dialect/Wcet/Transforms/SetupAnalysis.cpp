//===- SetupAnalysis.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the UnrollInstr pass
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"

#include "Dialect/Wcet/IR/WcetOps.h"
#include <string>

#define DEBUG_TYPE "SetupAnalysis"

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_SETUPANALYSISPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct SetupAnalysisPass : public impl::SetupAnalysisPassBase<SetupAnalysisPass> {

  using SetupAnalysisPassBase::SetupAnalysisPassBase;

public:
  void runOnOperation() override {
    auto mod = getOperation();
    wcet::CoreOp analysisedCore = nullptr;
    mod->walk([&](wcet::CoreOp core) {
      auto attr = core->getAttr("wcet.cpuCore");
      if (attr) {
        analysisedCore = core;
        return;
      }
    });

    if (!analysisedCore) {
      signalPassFailure();
      return;
    }

    size_t nbInstrs = 0;
    auto nbInstrsAttr = dyn_cast_or_null<IntegerAttr>(analysisedCore->getAttr("wcet.nbInstrs"));
    if (nbInstrsAttr) {
      nbInstrs = nbInstrsAttr.getInt();
    } else {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPointToEnd(getOperation().getBody());
    auto core = rewriter.create<wcet::CoreOp>(rewriter.getUnknownLoc(), rewriter.getFunctionType({}, {}),
                                              rewriter.getStringAttr("coreAnalysis"));
    core->setAttr("wcet.analysis", rewriter.getUnitAttr());
    rewriter.setInsertionPointToEnd(&core.getBody().front());
    SmallVector<Value> dummyInputs;
    SmallVector<Type> dummyTypes;
    for (; nbInstrs < analysisedCore.getArgAttrs()->size(); nbInstrs++) {
      std::string name = "init_" + std::to_string(nbInstrs);
      auto c =
          rewriter.create<wcet::InitOp>(rewriter.getUnknownLoc(), analysisedCore.getArgumentTypes()[nbInstrs], name);
      dummyInputs.push_back(c.getResult());
      dummyTypes.push_back(c.getType());
    }

    auto dummy = rewriter.create<wcet::DummyOp>(rewriter.getUnknownLoc(), dummyTypes, dummyInputs);
    dummy->setAttr("wcet.next", rewriter.getUnitAttr());
    dummy->setAttr("wcet.penalties", rewriter.getI32IntegerAttr(0));
    rewriter.create<wcet::CommitOp>(rewriter.getUnknownLoc(), core->getResultTypes(), ValueRange());
  }
};

} // namespace wcet
