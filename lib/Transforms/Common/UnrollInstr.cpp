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

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <cstdio>

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

// Setup the HWModuleOp to be unrolled
bool setupHWModule(hw::HWModuleOp op, PatternRewriter &rewriter,
                   SmallVector<MuOp> &mus) {
  // Check module validity
  if (!hasPragmaContaining(op, "UNROLL_NODE")) {
    return false;
  }

  // Remove all MuOp
  op.walk([&](MuOp mu) {
    mus.push_back(mu);
    // Replace the uses of the output of the mu by an op's argument
    std::pair<StringAttr, BlockArgument> arg =
        op.appendInput(mu.getName(), mu.getResult().getType());
    op.appendOutput(mu.getName(), mu.getNext());
    rewriter.replaceOp(mu, arg.second);
  });

  std::pair<StringAttr, BlockArgument> arg =
      op.appendInput("instrs", rewriter.getI32Type());

  op.walk([&](ArrayReadOp arr) {
    if (hasPragmaContaining(arr, "entry_point")) {
      rewriter.replaceAllUsesWith(arr, arg.second);
      return;
    }
  });
  op.walk([&](hw::InstanceOp inst) {
    if (hasPragmaContaining(inst, "entry_point")) {
      rewriter.replaceOp(inst, arg.second);
      return;
    }
    op.walk([&](DelayOp del) {
      auto input = del.getNext();
      input.getDefiningOp()->setAttr(
          "delay", rewriter.getI32IntegerAttr(del.getDepth()));
      rewriter.replaceOp(del, input);
    });
  });

  // Update pragma
  setPragmaAttr(op, rewriter.getStringAttr("INLINE"));
  return true;
}

namespace SpecHLS {

struct UnrollInstrPattern : OpRewritePattern<hw::HWModuleOp> {
  llvm::ArrayRef<unsigned int> instrs;
  using OpRewritePattern<hw::HWModuleOp>::OpRewritePattern;

  // Constructor to save pass arguments
  UnrollInstrPattern(MLIRContext *ctx, const llvm::ArrayRef<unsigned int> intrs)
      : OpRewritePattern<hw::HWModuleOp>(ctx) {
    this->instrs = intrs;
  }

  LogicalResult matchAndRewrite(hw::HWModuleOp top,
                                PatternRewriter &rewriter) const override {
    SmallVector<MuOp> mus = SmallVector<MuOp>();
    if (!setupHWModule(top, rewriter, mus)) {
      return failure();
    }

    // Create the new module with all instances
    SmallVector<hw::PortInfo> ports;

    StringAttr module_name =
        rewriter.getStringAttr(top.getName().str() + std::string("_unroll"));
    hw::HWModuleOp unrolled_module = rewriter.create<hw::HWModuleOp>(
        rewriter.getUnknownLoc(), module_name, ports);
    Block *body = unrolled_module.getBodyBlock();
    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockBegin(unrolled_module.getLoc(), body);

    SmallVector<InitOp> inits = SmallVector<InitOp>();
    size_t regs_num = 0;
    for (size_t i = 0; i < mus.size(); i++) {
      auto mu = mus[i];
      auto initial =
          builder.create<InitOp>(mu.getInit().getType(), mu.getName());
      if (mu.getName() == "regs") {
        regs_num = i;
        setPragmaAttr(initial, rewriter.getStringAttr("MU_INITIAL"));
      }
      inits.push_back(initial);
    }
    // Create a constOp for each instruction
    SmallVector<hw::ConstantOp> cons;
    hw::ConstantOp first_instrs =
        builder.create<hw::ConstantOp>(builder.getI32IntegerAttr(instrs[0]));
    cons.push_back(first_instrs);
    for (size_t i = 1; i < instrs.size(); i++) {
      cons.push_back(
          builder.create<hw::ConstantOp>(builder.getI32IntegerAttr(instrs[i])));
    }

    // Constant to drive the DelayOp between each instance
    hw::ConstantOp enable =
        builder.create<hw::ConstantOp>(builder.getBoolAttr(1));

    // Add the first instance with the initial value
    SmallVector<Value> inputs;
    for (size_t i = 0; i < inits.size(); i++)
      inputs.push_back(inits[i]);
    inputs.push_back(cons[0]);
    hw::InstanceOp inst =
        builder.create<hw::InstanceOp>(top, top.getName(), inputs);

    SmallVector<DelayOp> deltas = SmallVector<DelayOp>();
    for (size_t i = 0; i < mus.size(); i++) {
      DelayOp delta = builder.create<DelayOp>(
          inst.getType(i + 1), inst.getResult(i + 1), enable,
          inst.getResult(i + 1), builder.getI32IntegerAttr(1));
      deltas.push_back(delta);
    }
    // Add the other instances
    for (size_t i = 1; i < cons.size(); i++) {
      inputs.clear();
      for (size_t j = 0; j < deltas.size(); j++) {
        inputs.push_back(deltas[j].getResult());
      }
      inputs.push_back(cons[i]);
      inst = builder.create<hw::InstanceOp>(top, top.getName(), inputs);
      deltas.clear();
      for (size_t j = 0; j < mus.size(); j++) {
        DelayOp delta = builder.create<DelayOp>(
            inst.getType(j + 1), inst.getResult(j + 1), enable,
            inst.getResult(j + 1), builder.getI32IntegerAttr(1));
        deltas.push_back(delta);
      }
    }

    // Plug the last delay into the output of the module
    unrolled_module.appendOutput("out", deltas[regs_num].getResult());
    setPragmaAttr(unrolled_module, rewriter.getStringAttr("TOP"));

    return success();
  }
};

struct UnrollInstrPass : public impl::UnrollInstrPassBase<UnrollInstrPass> {

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto pm = PassManager::on<ModuleOp>(ctx);

    RewritePatternSet patterns(ctx);
    patterns.add<UnrollInstrPattern>(ctx, *instrs);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "failed\n";
      signalPassFailure();
    }

    OpPassManager dynamicPM("builtin.module");
    dynamicPM.addPass(createInlineModulesPass());
    dynamicPM.addPass(createCanonicalizerPass());
    if (failed(runPipeline(dynamicPM, getOperation()))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createUnrollInstrPass() {
  return std::make_unique<UnrollInstrPass>();
}

} // namespace SpecHLS
