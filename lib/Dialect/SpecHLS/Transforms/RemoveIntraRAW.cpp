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
    auto kernel = root->getParentOfType<spechls::KernelOp>();
    auto scn = dyn_cast_or_null<IntegerAttr>(kernel->getAttr("spechls.max_scc"));
    if (scn == nullptr) {
      scn = rewritter.getI32IntegerAttr(0);
    }
    auto alpha = root.getArray().getDefiningOp<spechls::AlphaOp>();
    alpha->setAttr("spechls.scn", scn);

    // auto *arrayProducer = alpha.getArray().getDefiningOp();
    // if (arrayProducer->getName().getStringRef() == spechls::MuOp::getOperationName()) {
    //   auto mu = cast<spechls::MuOp>(arrayProducer);
    //   auto muSkipReduce = mu->getAttr("spechls.skipReduce");
    //   if (muSkipReduce == nullptr) {
    //     auto muSpecDistance = dyn_cast_or_null<IntegerAttr>(mu->getAttr("spechls.memspec"));
    //     if (muSpecDistance != nullptr) {
    //       auto newDist = rewritter.getI32IntegerAttr(muSpecDistance.getInt() - 1);
    //       mu->setAttr("spechls.memspec", newDist);
    //       mu->setAttr("spechls.skipReduce", rewritter.getUnitAttr());
    //     }
    //   }
    // }

    Value rootIndex = root.getIndex();
    if (root.getIndex().getType().isSigned() || root.getIndex().getType().isUnsigned()) {
      auto rootI32Index = rewritter.create<circt::hw::BitcastOp>(
          rewritter.getUnknownLoc(), rewritter.getIntegerType(root.getIndex().getType().getWidth()), root.getIndex());
      rootI32Index->setAttr("spechls.scn", scn);
      rootIndex = rootI32Index.getResult();
    }

    Value alphaIndex = alpha.getIndex();
    if (alpha.getIndex().getType().isSigned() || root.getIndex().getType().isUnsigned()) {
      auto alphaI32Index = rewritter.create<circt::hw::BitcastOp>(
          rewritter.getUnknownLoc(), rewritter.getIntegerType(alpha.getIndex().getType().getWidth()), alpha.getIndex());
      alphaI32Index->setAttr("spechls.scn", scn);
      alphaIndex = alphaI32Index.getResult();
    }

    if (alphaIndex.getType() != rootIndex.getType()) {
      size_t loadSize = rootIndex.getType().getIntOrFloatBitWidth();
      size_t writeSize = alphaIndex.getType().getIntOrFloatBitWidth();
      if (writeSize > loadSize) {
        auto consSize = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(),
                                                                rewritter.getIntegerType(writeSize - loadSize), 0);
        auto concat = rewritter.create<circt::comb::ConcatOp>(rewritter.getUnknownLoc(), consSize, rootIndex);
        rootIndex = concat.getResult();
      } else {
        auto consSize = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(),
                                                                rewritter.getIntegerType(loadSize - writeSize), 0);
        auto concat = rewritter.create<circt::comb::ConcatOp>(rewritter.getUnknownLoc(), consSize, alphaIndex);
        alphaIndex = concat.getResult();
      }
    }

    // Create the new load before the alpha
    auto newLoad =
        rewritter.create<spechls::LoadOp>(rewritter.getUnknownLoc(), root.getType(), alpha.getArray(), root.getIndex());
    newLoad->setAttrs(root->getDiscardableAttrDictionary());
    newLoad->setAttr("spechls.scn", scn);

    // Create the alias detection logic
    auto addrAlias = rewritter.create<circt::comb::ICmpOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(),
                                                           circt::comb::ICmpPredicate::eq, rootIndex, alphaIndex);
    addrAlias->setAttr("spechls.scn", scn);
    auto aliasCheck = rewritter.create<circt::comb::AndOp>(rewritter.getUnknownLoc(), addrAlias, alpha.getWe());
    addrAlias->setAttr("spechls.scn", scn);

    // Set the gamma for the "forward"
    auto gammaCtrl =
        rewritter.create<circt::comb::AndOp>(rewritter.getUnknownLoc(), alpha.getWe(), aliasCheck.getResult());
    gammaCtrl->setAttr("spechls.scn", scn);
    SmallVector<Value> inputs;
    inputs.push_back(newLoad.getResult());
    inputs.push_back(alpha.getValue());
    // auto gamma =
    //     rewritter.create<spechls::GammaOp>(rewritter.getUnknownLoc(), root.getType(),
    //                                        rewritter.getStringAttr("aliasDetection"), gammaCtrl.getResult(), inputs);
    auto gamma = rewritter.create<circt::comb::MuxOp>(rewritter.getUnknownLoc(), gammaCtrl.getResult(),
                                                      newLoad.getResult(), alpha.getValue());
    gamma->setAttr("spechls.scn", scn);
    rewritter.replaceAllOpUsesWith(root, gamma);
    return newLoad;
  }

  static Operation *throughGammaImpl(PatternRewriter &rewritter, Operation *op) {
    auto root = cast<spechls::LoadOp>(op);
    auto kernel = root->getParentOfType<spechls::KernelOp>();
    auto scn = dyn_cast_or_null<IntegerAttr>(kernel->getAttr("spechls.max_scc"));
    if (scn == nullptr) {
      scn = rewritter.getI32IntegerAttr(0);
    }
    auto gammaAlpha = root.getArray().getDefiningOp<spechls::GammaOp>();
    gammaAlpha->setAttr("spechls.scn", scn);
    SmallVector<Value> newReads;
    for (auto in : gammaAlpha.getInputs()) {
      auto newRead = rewritter.create<spechls::LoadOp>(rewritter.getUnknownLoc(), root.getType(), in, root.getIndex());
      newRead->setAttrs(root->getAttrDictionary());
      newRead->setAttr("spechls.scn", scn);
      newReads.push_back(newRead.getResult());
    }
    auto newGamma =
        rewritter.create<spechls::GammaOp>(rewritter.getUnknownLoc(), root.getType(),
                                           rewritter.getStringAttr("gamma_x"), gammaAlpha.getSelect(), newReads);
    newGamma->setAttr("spechls.scn", scn);
    return newGamma;
  }
};
} // namespace
