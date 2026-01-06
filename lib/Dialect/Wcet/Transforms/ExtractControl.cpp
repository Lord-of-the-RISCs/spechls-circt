//===- ExtractControl.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
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
#include "Dialect/Wcet/IR/Wcet.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <deque>

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_EXTRACTCONTROLPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct ExtractControlPass : public impl::ExtractControlPassBase<ExtractControlPass> {

  using ExtractControlPassBase::ExtractControlPassBase;

public:
  void runOnOperation() override {
    auto mod = getOperation();
    IRRewriter rewriter(&getContext());

    spechls::TaskOp top = nullptr;
    mod->walk([&](spechls::TaskOp t) {
      if (t->hasAttr("spechls.speculative"))
        top = t;
    });

    if (!top) {
      return;
    }

    size_t nbOp = 0;
    top->walk([&](Operation *op) { nbOp++; });

    // Mark all operation that are in the path of each speculated gammas
    SmallVector<spechls::GammaOp> speculatedGamma;
    top->walk([&](spechls::GammaOp g) {
      for (auto operand : g.getInputs()) {
        if (!operand.getDefiningOp())
          continue;
        if (operand.getDefiningOp()->getName().getStringRef() == wcet::PenaltyOp::getOperationName().str()) {
          speculatedGamma.push_back(g);
          break;
        }
      }
    });

    size_t counter = 0;
    std::deque<Operation *> stack;
    for (auto g : speculatedGamma) {
      if (!g.getSelect().getDefiningOp())
        continue;
      stack.push_back(g.getSelect().getDefiningOp());
      g.getSelect().getDefiningOp()->setAttr("wcet.inCtrlPath", rewriter.getUnitAttr());
      counter++;
    }

    while (!stack.empty()) {
      Operation *current = stack.at(0);
      stack.pop_front();
      for (auto operand : current->getOperands()) {
        if (!operand.getDefiningOp())
          continue;
        Operation *next = operand.getDefiningOp();
        if (next->hasAttr("wcet.inCtrlPath"))
          continue;
        // if (next->getName().getStringRef() == spechls::LoadOp::getOperationName().str())
        //   continue;
        counter++;
        next->setAttr("wcet.inCtrlPath", rewriter.getUnitAttr());
        stack.push_back(next);
      }
    }
    llvm::errs() << "total before: " << nbOp << "\n";
    llvm::errs() << "total after: " << counter << "\n";
  }
};

} // namespace wcet
