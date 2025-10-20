#include "circt/Dialect/Comb/CombDialect.h"
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <Dialect/SpecHLS/IR/SpecHLSOps.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_REMOVEINTRARAWPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "RemoveIntraRAW.h.inc"

namespace {

struct RemoveIntraRAWPass : public spechls::impl::RemoveIntraRAWPassBase<RemoveIntraRAWPass> {
  FrozenRewritePatternSet patterns;

  using RemoveIntraRAWPassBase::RemoveIntraRAWPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("ThroughAlpha", throughAlphaImpl);
    patterns.getPDLPatterns().registerRewriteFunction("ThroughGamma", throughGammaImpl);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Operation *throughAlphaImpl(PatternRewriter &rewritter, Operation *op) {
    auto root = cast<spechls::LoadOp>(op);
    auto alpha = root.getArray().getDefiningOp<spechls::AlphaOp>();
    auto newLoad =
        rewritter.create<spechls::LoadOp>(rewritter.getUnknownLoc(), root.getType(), alpha.getArray(), root.getIndex());
    auto aliasCheck =
        rewritter.create<circt::comb::ICmpOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(),
                                              circt::comb::ICmpPredicate::eq, root.getIndex(), alpha.getIndex());
    SmallVector<Value> inputs;
    inputs.push_back(newLoad.getResult());
    inputs.push_back(alpha.getValue());
    auto gamma =
        rewritter.create<spechls::GammaOp>(rewritter.getUnknownLoc(), root.getType(),
                                           rewritter.getStringAttr("aliasDetection"), aliasCheck.getResult(), inputs);
    rewritter.replaceAllOpUsesWith(root, gamma);
    return newLoad;
  }

  static Operation *throughGammaImpl(PatternRewriter &rewritter, Operation *op) {
    auto root = cast<spechls::LoadOp>(op);
    auto gammaAlpha = root.getArray().getDefiningOp<spechls::GammaOp>();
    SmallVector<Value> newReads;
    for (auto in : gammaAlpha.getInputs()) {
      auto newRead = rewritter.create<spechls::LoadOp>(rewritter.getUnknownLoc(), root.getType(), in, root.getIndex());
      newReads.push_back(newRead.getResult());
    }
    auto newGamma =
        rewritter.create<spechls::GammaOp>(rewritter.getUnknownLoc(), root.getType(),
                                           rewritter.getStringAttr("gamma_x"), gammaAlpha.getSelect(), newReads);
    return newGamma;
  }
};
} // namespace
