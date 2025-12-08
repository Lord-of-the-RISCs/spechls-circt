//===- EraseAnalyzedCore.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
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

#include "Dialect/Wcet/IR/Wcet.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <circt/Dialect/HW/HWOps.h>

#include <deque>

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_ERASEANALYZEDCOREPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct EraseAnalyzedCorePass : public impl::EraseAnalyzedCorePassBase<EraseAnalyzedCorePass> {

  using EraseAnalyzedCorePassBase::EraseAnalyzedCorePassBase;

public:
  void runOnOperation() override {
    auto top = getOperation();
    if (top.getSymName() != CORE_ANALYSIS_NAME) {
      return;
    }

    wcet::DummyOp dum = nullptr;
    top->walk([&](wcet::DummyOp d) {
      if (d->hasAttr("wcet.next")) {
        dum = d;
        return;
      }
    });
    if (!dum) {
      signalPassFailure();
      llvm::errs() << "dummyOp not found\n";
      return;
    }

    std::deque<Operation *> toProceed;
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(dum);
    for (auto input : llvm::enumerate(dum.getInputs())) {
      auto *op = input.value().getDefiningOp();
      if (!op)
        continue;
      if (op->hasTrait<OpTrait::ConstantLike>())
        continue;
      if (op->getName().getStringRef() == wcet::DummyOp::getOperationName().str())
        continue;

      toProceed.push_back(op);
      wcet::InitOp newOp = nullptr;
      for (auto output : op->getResults()) {
        if (output == input.value()) {
          newOp =
              rewriter.create<wcet::InitOp>(rewriter.getUnknownLoc(), output.getType(), rewriter.getStringAttr("none"));
          break;
        }
      }
      if (!newOp) {
        signalPassFailure();
        llvm::errs() << "Unknowns Operation\n";
        return;
      }
      dum->setOperand(input.index(), newOp);
    }

    mlir::DenseSet<Operation *> toRemove;
    while (!toProceed.empty()) {
      Operation *current = toProceed.at(0);
      toProceed.pop_front();
      toRemove.insert(current);
      for (auto inputs : current->getOperands()) {
        auto *op = inputs.getDefiningOp();
        if (!op)
          continue;
        if (op->hasTrait<OpTrait::ConstantLike>())
          continue;
        auto opName = op->getName().getStringRef();
        if (opName == wcet::DummyOp::getOperationName().str() || opName == wcet::InitOp::getOperationName().str())
          continue;
        toProceed.push_back(op);
      }
    }

    for (auto *op : toRemove) {
      op->dropAllUses();
      op->erase();
    }
  }
};

} // namespace wcet
