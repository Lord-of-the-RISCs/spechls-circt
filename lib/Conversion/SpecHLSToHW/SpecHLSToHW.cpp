//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h" // IWYU pragma: keep
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombDialect.h"

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
#define GEN_PASS_DEF_SPECHLSTASKTOHWPASS
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
      auto ctrl = gamma.getSelect();
      auto ctrlType = llvm::dyn_cast<mlir::IntegerType>(ctrl.getType());
      auto zero = rewriter.create<circt::hw::ConstantOp>(ctrl.getLoc(), llvm::APInt(ctrlType.getWidth(), 0, false));
      auto ctrlTypeless =
          rewriter.create<circt::hw::BitcastOp>(ctrl.getLoc(), rewriter.getIntegerType(ctrlType.getWidth()), ctrl);
      auto ne = rewriter.create<circt::comb::ICmpOp>(ctrl.getLoc(), circt::comb::ICmpPredicate::ne, ctrlTypeless, zero);
      rewriter.replaceOpWithNewOp<circt::comb::MuxOp>(gamma, gamma.getType(), ne, gamma.getInputs()[1],
                                                      gamma.getInputs()[0]);
      return success();
    }
    llvm::SmallVector<mlir::Value> inputs;
    for (auto in : llvm::reverse(gamma.getInputs()))
      inputs.push_back(in);
    // See https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/#no-multibit-mux-operations
    auto array = rewriter.create<circt::hw::ArrayCreateOp>(gamma.getLoc(), inputs);
    auto select = gamma.getSelect();
    auto selectType = rewriter.getIntegerType(select.getType().getWidth());
    auto castedSelect = rewriter.create<circt::hw::BitcastOp>(select.getLoc(), selectType, select);
    rewriter.replaceOpWithNewOp<circt::hw::ArrayGetOp>(gamma, array, castedSelect);
    return success();
  }
};

struct SyncConvesion : OpConversionPattern<spechls::SyncOp> {
  using OpConversionPattern<spechls::SyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(spechls::SyncOp sync, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(sync, sync.getInputs().front());
    return mlir::success();
  }
};

struct LutConversion : OpConversionPattern<spechls::LUTOp> {
  using OpConversionPattern<spechls::LUTOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(spechls::LUTOp lut, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> inputs;
    auto outputType = lut.getResult().getType();
    auto signlessType = rewriter.getIntegerType(outputType.getWidth());
    for (auto it = lut.getContents().rbegin(); it != lut.getContents().rend(); ++it) {
      auto index = *it;
      auto cst = rewriter.create<circt::hw::ConstantOp>(lut.getLoc(), rewriter.getIntegerAttr(signlessType, index));
      inputs.push_back(cst.getResult());
    }
    auto array = rewriter.create<circt::hw::ArrayCreateOp>(lut.getLoc(), inputs);
    auto index = lut.getIndex();
    auto indexSignlessType = rewriter.getIntegerType(index.getType().getWidth());
    auto castedIndex = rewriter.create<circt::hw::BitcastOp>(index.getLoc(), indexSignlessType, index);
    auto get = rewriter.create<circt::hw::ArrayGetOp>(lut.getLoc(), array, castedIndex);
    rewriter.replaceOpWithNewOp<circt::hw::BitcastOp>(lut, outputType, get);
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
    patternList.add<LutConversion>(ctx);
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

struct ConvertSpecHLSTaskToHWPass : public spechls::impl::SpecHLSTaskToHWPassBase<ConvertSpecHLSTaskToHWPass> {
  FrozenRewritePatternSet patterns;

  using SpecHLSTaskToHWPassBase<ConvertSpecHLSTaskToHWPass>::SpecHLSTaskToHWPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    patternList.add<TaskConversion>(ctx);
    patternList.add<GammaConversion>(ctx);
    patternList.add<LutConversion>(ctx);
    patternList.add<SyncConvesion>(ctx);
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
