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

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>

using namespace mlir;
using namespace spechls;
using namespace circt;

namespace wcet {
#define GEN_PASS_DEF_INSERTDUMMYPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct InsertDummyPattern : OpRewritePattern<spechls::KernelOp> {
  using OpRewritePattern<spechls::KernelOp>::OpRewritePattern;

  // Constructor to save pass arguments
  InsertDummyPattern(MLIRContext *ctx, const llvm::ArrayRef<unsigned int> intrs)
      : OpRewritePattern<spechls::KernelOp>(ctx) {}

  LogicalResult matchAndRewrite(spechls::KernelOp top, PatternRewriter &rewriter) const override {
    SmallVector<spechls::UnpackOp> unpacks = SmallVector<spechls::UnpackOp>();
    top->walk([&](spechls::UnpackOp unp) {
      if (unp.getInput().getDefiningOp()->getName().getStringRef() == spechls::PackOp::getOperationName())
        unpacks.push_back(unp);
    });

    if (unpacks.empty())
      return failure();

    for (auto un : unpacks) {
      rewriter.setInsertionPointAfter(un);
      spechls::PackOp pack = dyn_cast_or_null<spechls::PackOp>(un.getInput().getDefiningOp());
      rewriter.replaceOpWithNewOp<wcet::DummyOp>(un, pack.getInputs().getType(), pack.getInputs());
      rewriter.eraseOp(pack);
    }

    return success();
  }
};

struct InsertDummyPass : public impl::InsertDummyPassBase<InsertDummyPass> {

  using InsertDummyPassBase::InsertDummyPassBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet dummyPatterns(ctx);
    dummyPatterns.add<InsertDummyPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation()->getParentOp(), std::move(dummyPatterns)))) {
      llvm::errs() << "failed\n";
      signalPassFailure();
    }
  }
};

} // namespace wcet
