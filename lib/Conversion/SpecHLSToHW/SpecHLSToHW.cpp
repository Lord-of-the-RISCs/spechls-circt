//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h" // IWYU pragma: keep
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HW/HWTypes.h>
#include <circt/Dialect/HW/PortImplementation.h>
#include <circt/Support/LLVM.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>

namespace spechls {
#define GEN_PASS_DEF_SPECHLSTOHWPASS
#include "Conversion/SpecHLS/Passes.h.inc"
}; // namespace spechls

using namespace mlir;

namespace {

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
      hwInputInfo.push_back({{rewriter.getStringAttr("arg" + std::to_string(hwInputInfo.size() - 1)), argType,
                              circt::hw::ModulePort::Input}});
    }
    circt::hw::ModulePortInfo hwPortInfo(hwInputInfo);

    auto hwModule = rewriter.create<circt::hw::HWModuleOp>(kernel.getLoc(), kernel.getSymNameAttr(), hwPortInfo);

    // TODO: Handle exit condition.
    auto *exitOp = kernel.getBody().front().getTerminator();
    auto *outputOp = hwModule.getBodyBlock()->getTerminator();
    rewriter.inlineBlockBefore(&kernel.getBody().front(), outputOp, hwModule.getBody().getArguments());
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
    circt::hw::ModulePortInfo hwPortInfo(hwInputInfo);

    auto hwModule = rewriter.create<circt::hw::HWModuleOp>(task.getLoc(), task.getSymNameAttr(), hwPortInfo);

    // TODO: Handle commit condition.
    auto *commitOp = task.getBody().front().getTerminator();
    auto *outputOp = hwModule.getBodyBlock()->getTerminator();
    rewriter.inlineBlockBefore(&task.getBody().front(), outputOp, hwModule.getBody().getArguments());
    outputOp->setOperands({cast<spechls::CommitOp>(commitOp).getValue()});

    rewriter.restoreInsertionPoint(insertionPoint);
    SmallVector<Value> operands(task.getArgs());
    rewriter.replaceOpWithNewOp<circt::hw::InstanceOp>(task, hwModule, task.getSymNameAttr(), operands);

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
  FrozenRewritePatternSet patterns;

  using SpecHLSToHWPassBase<ConvertSpecHLSToHWPass>::SpecHLSToHWPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    patternList.add<KernelConversion>(ctx);
    patternList.add<TaskConversion>(ctx);
    patternList.add<GammaConversion>(ctx);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<circt::hw::HWDialect>();
    target.addLegalDialect<circt::comb::CombDialect>();
    target.addIllegalDialect<spechls::SpecHLSDialect>();

    if (failed(mlir::applyFullConversion(getOperation(), target, patterns)))
      return signalPassFailure();
  }
};

} // namespace
