//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "mlir/IR/PatternMatch.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <mlir/IR/Builders.h>

namespace spechls {
#define GEN_PASS_DEF_OPTFUNTOTASKPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

class OptFunToTaskPass : public spechls::impl::OptFunToTaskPassBase<OptFunToTaskPass> {
public:
  using OptFunToTaskPassBase::OptFunToTaskPassBase;

  void runOnOperation() override {
    auto kernel = getOperation();
    auto *ctx = &getContext();
    mlir::IRRewriter rewriter(ctx);
    unsigned idx = 0;
    kernel.walk([&](spechls::OptimizedFuncOp fun) {
      rewriter.setInsertionPoint(fun);
      auto structType =
          spechls::StructType::get(rewriter.getContext(), "task_inline_" + std::to_string(idx),
                                   {"enable", "commit_val_0"}, {rewriter.getI1Type(), fun.getResult().getType()});
      auto task = spechls::TaskOp::create(rewriter, fun->getLoc(), structType, "task_inlined_" + std::to_string(idx++),
                                          fun.getArgs());
      rewriter.setInsertionPointToStart(task.getBodyBlock());
      auto trueCst = circt::hw::ConstantOp::create(rewriter, rewriter.getUnknownLoc(), rewriter.getIntegerType(1), 1);
      llvm::SmallVector<mlir::Value> commitArgs;
      commitArgs.push_back(trueCst.getResult());
      commitArgs.push_back(llvm::cast<spechls::YieldOp>(fun.getOptBodyBlock()->getTerminator()).getValue());
      auto commit = spechls::CommitOp::create(rewriter, rewriter.getUnknownLoc(), commitArgs);
      rewriter.eraseOp(fun.getOptBodyBlock()->getTerminator());
      rewriter.inlineBlockBefore(fun.getOptBodyBlock(), commit, task.getBodyBlock()->getArguments());
      rewriter.setInsertionPointAfter(task);
      auto field = spechls::FieldOp::create(rewriter, rewriter.getUnknownLoc(), "commit_val_0", task.getResult());
      rewriter.replaceOp(fun, field);
    });
  }
};

} // namespace