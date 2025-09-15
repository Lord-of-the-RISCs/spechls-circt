//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h" // IWYU pragma: keep
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/ErrorHandling.h"

#include <circt/Dialect/HW/HWOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>

namespace spechls {
#define GEN_PASS_DEF_SPECHLSTOHWPASS
#include "Conversion/SpecHLS/Passes.h.inc"
}; // namespace spechls

using namespace mlir;

namespace {

struct SpecHLSToHWOpConversion : OpConversionPattern<spechls::KernelOp> {
  using OpConversionPattern<spechls::KernelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(spechls::KernelOp kernel, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
#if 0
    SmallVector<circt::hw::PortInfo> hwInputInfo;
    if (!adaptor.getFunctionType().getResults().empty()) {
      // TODO: Deconstruct struct types.
      hwInputInfo.push_back({{rewriter.getStringAttr("result"), adaptor.getFunctionType().getResults().front(),
                              circt::hw::ModulePort::Output}});
    }
    for (auto &&argType : adaptor.getFunctionType().getInputs()) {
      hwInputInfo.push_back({{rewriter.getStringAttr("arg" + std::to_string(hwInputInfo.size() - 1)), argType,
                              circt::hw::ModulePort::Input}});
    }
    circt::hw::ModulePortInfo hwPortInfo(hwInputInfo);

    auto hwModule = rewriter.create<circt::hw::HWModuleOp>(kernel.getLoc(), adaptor.getSymNameAttr(), hwPortInfo);

    auto *exitOp = adaptor.getBody().front().getTerminator();
    auto *outputOp = hwModule.getBodyBlock()->getTerminator();
    rewriter.inlineBlockBefore(&adaptor.getBody().front(), outputOp, hwModule.getBody().getArguments());
    outputOp->setOperands({cast<spechls::ExitOp>(exitOp).getValues()});
    rewriter.eraseOp(exitOp);
    rewriter.eraseOp(kernel);
#endif
    rewriter.eraseOp(kernel);
    return success();
  }
};

struct ConvertSpecHLSToHWPass : public spechls::impl::SpecHLSToHWPassBase<ConvertSpecHLSToHWPass> {
  FrozenRewritePatternSet patterns;

  using SpecHLSToHWPassBase<ConvertSpecHLSToHWPass>::SpecHLSToHWPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    patternList.add<SpecHLSToHWOpConversion>(ctx);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<circt::hw::HWDialect>();
    target.addIllegalDialect<spechls::SpecHLSDialect>();
    // target.addLegalDialect<spechls::SpecHLSDialect>();
    // target.addIllegalOp<spechls::KernelOp>();
    // target.addIllegalOp<spechls::ExitOp>();

    if (failed(mlir::applyFullConversion(getOperation(), target, patterns)))
      return signalPassFailure();
  }
};

} // namespace
