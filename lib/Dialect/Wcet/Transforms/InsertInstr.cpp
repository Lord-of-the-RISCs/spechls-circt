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

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_INSERTINSTRPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct InsertInstrPass : public impl::InsertInstrPassBase<InsertInstrPass> {
  using InsertInstrPassBase::InsertInstrPassBase;

public:
  void runOnOperation() override {
    if (instrs.empty())
      return;
    IRRewriter rewriter(&getContext());
    ModuleOp mod = getOperation();

    /*====----------------------------------------------------------------====*
     *             Get cores                                                  *
     *====----------------------------------------------------------------====*/
    wcet::CoreOp analysisCore = nullptr;
    wcet::CoreOp coreAnalysed = nullptr;
    uint32_t numInstr = 1;
    mod->walk([&](wcet::CoreOp core) {
      auto attr = core->getAttr("wcet.analysis");
      if (attr) {
        analysisCore = core;
        return;
      }
      attr = core->getAttr("wcet.cpuCore");
      if (attr) {
        auto numIn = circt::dyn_cast_or_null<IntegerAttr>(core->getAttr("wcet.numInstrs"));
        if (numIn)
          numInstr = numIn.getInt();
        coreAnalysed = core;
        return;
      }
    });

    if (!analysisCore || !coreAnalysed || numInstr != instrs.size()) {
      return;
    }

    /*====----------------------------------------------------------------====*
     *             Get the inputs of the core                                 *
     *====----------------------------------------------------------------====*/
    SmallVector<Value> inputs;
    //============ Get Current dummy node ====================================
    wcet::DummyOp currentDum = nullptr;
    int64_t pen = 0;
    analysisCore->walk([&](wcet::DummyOp dum) {
      auto attr = dum->getAttr("wcet.next");
      if (!attr)
        return;
      auto currentPen = circt::dyn_cast_or_null<IntegerAttr>(dum->getAttr("wcet.penalties"));
      if (!currentPen)
        return;
      currentDum = dum;
      pen = currentPen.getInt();
    });

    if (currentDum == nullptr) {
      return;
    }

    //============= Create constants for the instructions ====================
    rewriter.setInsertionPointAfter(currentDum);
    for (size_t i = 0; i < instrs.size(); i++) {
      auto c = rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), coreAnalysed.getArgumentTypes()[i],
                                                      instrs[i]);
      inputs.push_back(c.getResult());
    }

    //============= Setup the remaining inputs ================================
    for (size_t i = 0; i < coreAnalysed.getResultTypes().size(); i++) {
      auto lastResult = currentDum.getResult(i);
      auto nbPred = dyn_cast_or_null<IntegerAttr>(coreAnalysed.getArgAttr(i + instrs.size(), "wcet.nbPred"));
      if (!nbPred || nbPred.getInt() == 0) {
        inputs.push_back(lastResult);
      } else if (nbPred.getInt() > pen) {
        inputs.push_back(currentDum.getResult(i - pen));
      } else {
        IntegerType it = dyn_cast_or_null<IntegerType>(lastResult.getType());
        if (it && it.getWidth() == 1) {
          auto c0 = rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI1Type(), 0);
          inputs.push_back(c0.getResult());
        } else {
          auto dc = rewriter.create<wcet::DontCare>(rewriter.getUnknownLoc(), lastResult.getType());
          inputs.push_back(dc.getResult());
        }
      }
    }

    /*====----------------------------------------------------------------====*
     *             Build the next core                                        *
     *====----------------------------------------------------------------====*/
    auto nextCore = rewriter.create<wcet::CoreInstanceOp>(rewriter.getUnknownLoc(), coreAnalysed, inputs);

    /*====----------------------------------------------------------------====*
     *             Build the next dummy                                       *
     *====----------------------------------------------------------------====*/
    auto nextDummy =
        rewriter.create<wcet::DummyOp>(rewriter.getUnknownLoc(), nextCore->getResultTypes(), nextCore->getResults());

    nextDummy->setAttr("wcet.next", rewriter.getUnitAttr());
    nextDummy->setAttr("wcet.penalties", rewriter.getI32IntegerAttr(0));
    currentDum->setAttr("wcet.current", rewriter.getUnitAttr());
    currentDum->removeAttr("wcet.next");
  }
};
} // namespace wcet
