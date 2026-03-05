//===- UnrollInstr.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
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

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "Support/WcetUtils.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <cstddef>
#include <cstdio>

using namespace mlir;
using namespace spechls;
using namespace circt;

namespace wcet {
#define GEN_PASS_DEF_UNROLLINSTRPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct UnrollInstrPass : public impl::UnrollInstrPassBase<UnrollInstrPass> {

  using UnrollInstrPassBase::UnrollInstrPassBase;

public:
  void runOnOperation() override {
    if (instrs.size() == 0) {
      signalPassFailure();
      return;
    }

    wcet::CoreOp analyzedCore = nullptr;
    ModuleOp mod = getOperation();
    mod->walk([&analyzedCore](wcet::CoreOp c) {
      if (c->hasAttr("wcet.cpuCore"))
        analyzedCore = c;
    });
    if (!analyzedCore) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPointToEnd(mod.getBody());

    SmallVector<std::optional<mlir::IntegerAttr>> state;
    SmallVector<mlir::Type> types;
    state.push_back(rewriter.getI32IntegerAttr(instrs[0]));
    types.push_back(analyzedCore.getArgumentTypes()[0]);
    for (size_t i = 1; i < analyzedCore.getArgumentTypes().size(); i++) {
      // mlir::IntegerType it = mlir::dyn_cast_or_null<mlir::IntegerType>(analyzedCore.getArgumentTypes()[i]);
      // if (it)
      //   state.push_back(rewriter.getIntegerAttr(it, it.getWidth() > 1 ? 0 : 1));
      // else
      state.push_back({});
      types.push_back(analyzedCore.getArgumentTypes()[i]);
    }

    wcet::CoreOp analyzeCore = createAnalyseCore(rewriter, mod, analyzedCore, state, types);
    wcet::DummyOp lastDum = nullptr;
    analyzeCore->walk([&lastDum](wcet::DummyOp d) {
      if (d->hasAttr("wcet.next"))
        lastDum = d;
    });
    lastDum->removeAttr("wcet.next");

    rewriter.setInsertionPointAfter(lastDum);
    for (size_t i = 1; i < instrs.size(); i++) {
      SmallVector<mlir::Value> nextInputs;
      hw::ConstantOp ins = rewriter.create<hw::ConstantOp>(rewriter.getUnknownLoc(), types[0],
                                                           rewriter.getIntegerAttr(types[0], instrs[i]));
      nextInputs.push_back(ins);
      for (auto res : lastDum->getResults()) {
        nextInputs.push_back(res);
      }
      auto coreInst = rewriter.create<wcet::CoreInstanceOp>(rewriter.getUnknownLoc(), analyzedCore, nextInputs);
      lastDum =
          rewriter.create<wcet::DummyOp>(rewriter.getUnknownLoc(), coreInst->getResultTypes(), coreInst->getResults());
    }
    lastDum->setAttr("wcet.next", rewriter.getUnitAttr());
  }
};

} // namespace wcet
