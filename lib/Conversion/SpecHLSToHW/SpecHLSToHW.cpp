//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h" // IWYU pragma: keep
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HW/HWTypes.h>

#include <circt/Dialect/HW/PortImplementation.h>
#include <circt/Support/LLVM.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <mlir/Transforms/DialectConversion.h>
#include <Conversion/SpecHLS/SpecHLSToHW/common.h>
#include <string>

namespace spechls {
#define GEN_PASS_DEF_SPECHLSTOHWPASS
#include "Conversion/SpecHLS/Passes.h.inc"
}; // namespace spechls


#define debug llvm::nulls
//#define debug llvm::outs

using namespace mlir;

namespace {

static bool isStateFull(mlir::Operation *op) {
  bool found = false;
  debug() << "isStateFull for " << *op << "\n";
  op->walk([&](mlir::Operation *inner) -> mlir::WalkResult {
    debug() << "Visiting op: " << *inner << "\n";
    llvm::TypeSwitch<mlir::Operation *>(inner)
        .Case<spechls::MuOp, spechls::DelayOp, spechls::RollbackOp>([&](auto) { found = true; })
        .Default([&](mlir::Operation *) { /* no-op */ });

    if (found)
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  debug() << "isStateFull: " << found << "\n";
  return found;
}

struct KernelConversion : OpConversionPattern<spechls::KernelOp> {
  using OpConversionPattern<spechls::KernelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(spechls::KernelOp kernel, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<circt::hw::PortInfo> hwInputInfo;
    if (!kernel.getFunctionType().getResults().empty()) {
      // TODO: Deconstruct struct types.
      hwInputInfo.push_back({{rewriter.getStringAttr("result"), kernel.getFunctionType().getResults().front(),
                              circt::hw::ModulePort::Output}});
    }
    for (auto &&argType : kernel.getFunctionType().getInputs()) {
      debug() << "Arg type: " << argType << "\n";
      circt::hw::PortInfo elt = {{rewriter.getStringAttr("arg" + std::to_string(hwInputInfo.size() - 1)), argType,
                                  circt::hw::ModulePort::Input}};
      debug() << "Port type: " << elt << "\n";
      hwInputInfo.push_back(elt);
    }

    // hwInputInfo.push_back({{rewriter.getStringAttr("clk"),
    // circt::seq::ClockType::get(rewriter.getContext()),circt::hw::ModulePort::Input}});

    hwInputInfo.push_back({{rewriter.getStringAttr("clk"), circt::seq::ClockType::get(rewriter.getContext()),
                            circt::hw::ModulePort::Input}});
    hwInputInfo.push_back({{rewriter.getStringAttr("rst"), rewriter.getI1Type(), circt::hw::ModulePort::Input}});
    hwInputInfo.push_back({{rewriter.getStringAttr("ce"), rewriter.getI1Type(), circt::hw::ModulePort::Input}});

    circt::hw::ModulePortInfo hwPortInfo(hwInputInfo);

    auto hwModule = rewriter.create<circt::hw::HWModuleOp>(kernel.getLoc(), kernel.getSymNameAttr(), hwPortInfo);

    debug() << "HW Module type: " << hwModule << "\n";

    Block *block = &kernel.getBody().front();
    // add argument to kernel to enable inlining without errors
    block->addArgument(rewriter.getI1Type(), kernel.getLoc()); // clk
    block->addArgument(rewriter.getI1Type(), kernel.getLoc()); // rst
    block->addArgument(rewriter.getI1Type(), kernel.getLoc()); // ce
    //
    // TODO: Handle exit condition.
    auto *exitOp = block->getTerminator();
    auto *outputOp = hwModule.getBodyBlock()->getTerminator();
    rewriter.inlineBlockBefore(block, outputOp, hwModule.getBody().getArguments());
    outputOp->setOperands(cast<spechls::ExitOp>(exitOp).getValues());

    rewriter.eraseOp(exitOp);
    rewriter.eraseOp(kernel);
    return success();
  }
};

struct TaskConversion : OpConversionPattern<spechls::TaskOp> {
  using OpConversionPattern<spechls::TaskOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(spechls::TaskOp task, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto insertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(task->getParentOp());

    SmallVector<circt::hw::PortInfo> hwInputInfo;
    // TODO: Deconstruct struct types.
    hwInputInfo.push_back(
        {{rewriter.getStringAttr("result"), task.getResult().getType(), circt::hw::ModulePort::Output}});
    for (auto &&argType : task.getArgs().getTypes()) {
      hwInputInfo.push_back({{rewriter.getStringAttr("arg" + std::to_string(hwInputInfo.size() - 1)), argType,
                              circt::hw::ModulePort::Input}});
    }

    auto parentMod = task->getParentOfType<circt::hw::HWModuleOp>();
    if (!parentMod)
      return rewriter.notifyMatchFailure(task, "parent is not hw.module");
    // Fetch parent's CRE ports to forward into the instance.
    auto [pClk, pRst, pCe] = getCREArgs(parentMod);
    auto stateFull = isStateFull(task);
    if (stateFull) {
      if (pClk && pRst && pCe) {
        hwInputInfo.push_back({{rewriter.getStringAttr("clk"), pClk.getType(), circt::hw::ModulePort::Input}});
        hwInputInfo.push_back({{rewriter.getStringAttr("rst"), pRst.getType(), circt::hw::ModulePort::Input}});
        hwInputInfo.push_back({{rewriter.getStringAttr("ce"), pCe.getType(), circt::hw::ModulePort::Input}});
      } else {
        // failure
        return rewriter.notifyMatchFailure(task, "parent hw.module is missing CRE ports");
      }
    }

    circt::hw::ModulePortInfo hwPortInfo(hwInputInfo);

    auto hwModule = rewriter.create<circt::hw::HWModuleOp>(task.getLoc(), task.getSymNameAttr(), hwPortInfo);


    Block *block = &task.getBody().front();
    // TODO: Handle commit condition.
    auto *commitOp = block->getTerminator();
    auto *outputOp = hwModule.getBodyBlock()->getTerminator();
    if (stateFull) {
      if (pClk && pRst && pCe) {
        block->addArgument(pClk.getType(), task.getLoc()); // clk
        block->addArgument(pRst.getType(), task.getLoc()); // rst
        block->addArgument(pCe.getType(), task.getLoc());  // ce
      }
    }

    rewriter.inlineBlockBefore(block, outputOp, hwModule.getBody().getArguments());
    outputOp->setOperands({cast<spechls::CommitOp>(commitOp).getValue()});

    rewriter.restoreInsertionPoint(insertionPoint);
    SmallVector<Value> operands(task.getArgs());
    if (stateFull) {
      if (pClk && pRst && pCe) {
        operands.push_back(pClk);
        operands.push_back(pRst);
        operands.push_back(pCe);
      }
    }

    auto instanceOp =
         rewriter.create<circt::hw::InstanceOp>(task.getLoc(), hwModule, task.getSymNameAttr(), operands);

    for (auto rt : instanceOp.getResultTypes())
      rewriter.replaceOp(task, instanceOp.getResults());
    rewriter.eraseOp(commitOp);
    return success();
  }
};

struct GammaConversion : OpConversionPattern<spechls::GammaOp> {
  using OpConversionPattern<spechls::GammaOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(spechls::GammaOp gamma, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (gamma.getInputs().size() == 2) {
      rewriter.replaceOpWithNewOp<circt::comb::MuxOp>(gamma, gamma.getType(), gamma.getSelect(), gamma.getInputs()[1],
                                                      gamma.getInputs()[0]);
      return success();
    }

    // See https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/#no-multibit-mux-operations
    auto array = rewriter.create<circt::hw::ArrayCreateOp>(gamma.getLoc(), gamma.getInputs());
    rewriter.replaceOpWithNewOp<circt::hw::ArrayGetOp>(gamma, array, gamma.getSelect());
    return success();
  }
};


struct ConvertSpecHLSToHWPass : public spechls::impl::SpecHLSToHWPassBase<ConvertSpecHLSToHWPass> {
  FrozenRewritePatternSet hwPatterns;
  FrozenRewritePatternSet seqPatterns;

  using SpecHLSToHWPassBase<ConvertSpecHLSToHWPass>::SpecHLSToHWPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet hwPatternList{ctx};
    hwPatternList.add<KernelConversion>(ctx);
    hwPatternList.add<TaskConversion>(ctx);
    hwPatternList.add<GammaConversion>(ctx);
   // hwPatternList.add<MuConversion>(ctx);
    hwPatterns = std::move(hwPatternList);


    return success();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<circt::hw::HWDialect>();
    target.addLegalDialect<circt::comb::CombDialect>();
    target.addLegalDialect<circt::seq::SeqDialect>();
    target.addIllegalOp<spechls::KernelOp, spechls::TaskOp,spechls::GammaOp>();

    if (failed(mlir::applyPartialConversion(getOperation(), target, hwPatterns)))
      return signalPassFailure();
  }
};

} // namespace
