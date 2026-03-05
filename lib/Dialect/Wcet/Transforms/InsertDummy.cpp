//===- InsertDummy.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
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
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdio>

using namespace mlir;
using namespace spechls;
using namespace circt;

namespace wcet {
#define GEN_PASS_DEF_INSERTDUMMYPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct InsertDummyPass : public impl::InsertDummyPassBase<InsertDummyPass> {

  using InsertDummyPassBase::InsertDummyPassBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto top = getOperation();

    PatternRewriter rewriter(ctx);
    SmallVector<spechls::UnpackOp> unpacks = SmallVector<spechls::UnpackOp>();
    top->walk([&](spechls::UnpackOp unp) {
      if (unp.getInput().getDefiningOp()->getName().getStringRef() == spechls::PackOp::getOperationName())
        unpacks.push_back(unp);
    });

    if (unpacks.empty())
      return;

    for (auto un : unpacks) {
      rewriter.setInsertionPointAfter(un);
      spechls::PackOp pack = dyn_cast_or_null<spechls::PackOp>(un.getInput().getDefiningOp());
      rewriter.replaceOpWithNewOp<wcet::DummyOp>(un, pack.getInputs().getType(), pack.getInputs());
      rewriter.eraseOp(pack);
    }
  }
};

} // namespace wcet
