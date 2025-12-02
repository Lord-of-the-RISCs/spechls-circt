//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h" // IWYU pragma: keep
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
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
    auto taskType = task.getResult().getType();
    for (unsigned i = 0; i < taskType.getFieldTypes().size(); ++i) {
      hwInputInfo.push_back({{rewriter.getStringAttr(taskType.getFieldNames()[i]), taskType.getFieldTypes()[i],
                              circt::hw::ModulePort::Output}});
    }
    for (auto &&argType : task.getArgs().getTypes()) {
      hwInputInfo.push_back({{rewriter.getStringAttr("arg" + std::to_string(hwInputInfo.size() - 1)), argType,
                              circt::hw::ModulePort::Input}});
    }
    circt::hw::ModulePortInfo hwPortInfo(hwInputInfo);

    auto hwModule = rewriter.create<circt::hw::HWModuleOp>(task.getLoc(), task.getSymNameAttr(), hwPortInfo);

    // TODO: Handle commit condition.
    auto commitOp = cast<spechls::CommitOp>(task.getBody().front().getTerminator());
    auto *outputOp = hwModule.getBodyBlock()->getTerminator();
    rewriter.inlineBlockBefore(&task.getBody().front(), outputOp, hwModule.getBody().getArguments());

    outputOp->setOperands(commitOp.getOperands());

    rewriter.setInsertionPointAfter(task);
    auto instance = rewriter.create<circt::hw::InstanceOp>(task.getLoc(), hwModule, task.getSymNameAttr(),
                                                           llvm::SmallVector<Value>(task.getArgs()));
    rewriter.replaceOpWithNewOp<spechls::PackOp>(task, taskType, instance.getResults());

    rewriter.restoreInsertionPoint(insertionPoint);
    rewriter.eraseOp(commitOp);
    return success();
  }
};

struct GammaConversion : OpConversionPattern<spechls::GammaOp> {
  using OpConversionPattern<spechls::GammaOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(spechls::GammaOp gamma, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctrl = gamma.getSelect();
    if (gamma.getInputs().size() == 2) {
      mlir::Value ctrlTypeless;
      auto ctrlType = llvm::dyn_cast<mlir::IntegerType>(ctrl.getType());
      if (ctrlType.getWidth() != 1) {
        ctrlTypeless = rewriter.create<circt::comb::ExtractOp>(ctrl.getLoc(), ctrl, 0, 1);
      } else {
        ctrlTypeless =
            rewriter.create<circt::hw::BitcastOp>(ctrl.getLoc(), rewriter.getIntegerType(ctrlType.getWidth()), ctrl);
      }
      auto zero = rewriter.create<circt::hw::ConstantOp>(ctrl.getLoc(), llvm::APInt(1, 0, false));
      auto ne = rewriter.create<circt::comb::ICmpOp>(ctrl.getLoc(), circt::comb::ICmpPredicate::ne, ctrlTypeless, zero);
      rewriter.replaceOpWithNewOp<circt::comb::MuxOp>(gamma, gamma.getType(), ne, gamma.getInputs()[1],
                                                      gamma.getInputs()[0]);
      return success();
    }
    auto ctrlBw = ctrl.getType().getWidth();
    mlir::Value castedCtrl = rewriter.create<circt::hw::BitcastOp>(
        ctrl.getLoc(), mlir::IntegerType::get(rewriter.getContext(), ctrlBw), ctrl);
    unsigned numInput = 1u << ctrlBw;
    unsigned missingInputs = numInput - gamma.getInputs().size();
    llvm::SmallVector<mlir::Value> inputs;
    for (unsigned i = 0; i < missingInputs; ++i)
      inputs.push_back(gamma.getInputs().back());
    for (auto in : llvm::reverse(gamma.getInputs()))
      inputs.push_back(in);
    // See https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/#no-multibit-mux-operations
    auto array = rewriter.create<circt::hw::ArrayCreateOp>(gamma.getLoc(), inputs);
    rewriter.replaceOpWithNewOp<circt::hw::ArrayGetOp>(gamma, array, castedCtrl);
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

struct FieldConversion : OpConversionPattern<spechls::FieldOp> {
  using OpConversionPattern<spechls::FieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(spechls::FieldOp field, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (auto pack = llvm::dyn_cast_or_null<spechls::PackOp>(field.getOperand().getDefiningOp())) {
      auto name = field.getName();
      auto packType = pack.getType();
      for (unsigned i = 0; i < packType.getFieldNames().size(); ++i) {
        if (packType.getFieldNames()[i] == name) {
          auto value = pack.getOperand(i);
          rewriter.replaceAllUsesWith(field.getResult(), value);
          rewriter.eraseOp(field);
          return success();
        }
      }
    }
    return failure();
  }
};

struct ConvertSpecHLSToHWPass : public spechls::impl::SpecHLSToHWPassBase<ConvertSpecHLSToHWPass> {
  FrozenRewritePatternSet patterns1;
  FrozenRewritePatternSet patterns2;

  using SpecHLSToHWPassBase<ConvertSpecHLSToHWPass>::SpecHLSToHWPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList1{ctx};
    patternList1.add<TaskConversion>(ctx);
    patternList1.add<GammaConversion>(ctx);
    patternList1.add<LutConversion>(ctx);
    patterns1 = std::move(patternList1);
    RewritePatternSet patternList2{ctx};
    patternList2.add<KernelConversion>(ctx);
    patternList2.add<TaskConversion>(ctx);
    patternList2.add<GammaConversion>(ctx);
    patternList2.add<LutConversion>(ctx);
    patternList2.add<FieldConversion>(ctx);
    patterns2 = std::move(patternList2);
    return success();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<circt::hw::HWDialect>();
    target.addLegalDialect<circt::comb::CombDialect>();
    target.addIllegalDialect<spechls::SpecHLSDialect>();
    target.addLegalOp<spechls::PackOp>();
    target.addLegalOp<spechls::FieldOp>();
    target.addLegalOp<spechls::KernelOp>();
    target.addLegalOp<spechls::ExitOp>();

    if (failed(mlir::applyFullConversion(getOperation(), target, patterns1)))
      return signalPassFailure();

    ConversionTarget target2(getContext());
    target2.addLegalDialect<circt::hw::HWDialect>();
    target2.addLegalDialect<circt::comb::CombDialect>();
    target2.addIllegalDialect<spechls::SpecHLSDialect>();
    target2.addLegalOp<spechls::PackOp>();

    if (failed(mlir::applyFullConversion(getOperation(), target2, patterns2)))
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
    auto op = getOperation();
    auto taskSym = targetTask.getValue();
    spechls::TaskOp task;
    op.walk([&](spechls::TaskOp t) {
      if (t.getSymName() == taskSym) {
        task = t;
      }
    });
    target.addLegalDialect<circt::hw::HWDialect>();
    target.addLegalDialect<circt::comb::CombDialect>();
    target.addIllegalDialect<spechls::SpecHLSDialect>();
    target.addLegalOp<spechls::PackOp>();

    if (failed(mlir::applyFullConversion(task, target, patterns)))
      return signalPassFailure();
  }
};

} // namespace
