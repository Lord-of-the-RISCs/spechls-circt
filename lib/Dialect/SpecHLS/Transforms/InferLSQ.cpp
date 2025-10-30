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
#define GEN_PASS_DEF_INFERLSQPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "InferLSQ.h.inc"

namespace {

struct InferLSQPass : public spechls::impl::InferLSQPassBase<InferLSQPass> {
  FrozenRewritePatternSet patterns;

  using InferLSQPassBase::InferLSQPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("BuildDelayRB", buildDelayRBImpl);
    patterns.getPDLPatterns().registerRewriteFunction("RetimingDelay", retimingDelayImpl);
    patterns.getPDLPatterns().registerRewriteFunction("RetimingRollback", retimingRollbackImpl);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Operation *buildDelayRBImpl(PatternRewriter &rewriter, Operation *op) {
    auto root = cast<spechls::DelayOp>(op);
    auto oldRb = root.getInput().getDefiningOp<spechls::RollbackOp>();
    auto oldMu = oldRb.getInput().getDefiningOp<spechls::MuOp>();
    auto newDelay = rewriter.create<spechls::DelayOp>(rewriter.getUnknownLoc(), oldMu.getLoopValue().getType(),
                                                      oldMu.getLoopValue(), rewriter.getUI32IntegerAttr(1),
                                                      root.getEnable(), oldMu.getInitValue());
    auto newRb =
        rewriter.create<spechls::RollbackOp>(rewriter.getUnknownLoc(), oldRb.getType(), oldRb.getDepths(),
                                             oldRb.getOffset(), newDelay, oldRb.getRollback(), oldRb.getWriteCommand());
    auto result = rewriter.create<spechls::DelayOp>(rewriter.getUnknownLoc(), newRb.getType(), newRb, root.getDepth(),
                                                    root.getEnable(), root.getInit());
    return result;
  }

  static Operation *retimingDelayImpl(PatternRewriter &rewriter, Operation *op) {
    auto root = cast<spechls::DelayOp>(op);
    auto alpha = root.getInput().getDefiningOp<spechls::AlphaOp>();

    auto weDelay = rewriter.create<spechls::DelayOp>(rewriter.getUnknownLoc(), alpha.getWe().getType(), alpha.getWe(),
                                                     rewriter.getUI32IntegerAttr(1), root.getEnable(), alpha.getWe());
    auto valDelay =
        rewriter.create<spechls::DelayOp>(rewriter.getUnknownLoc(), alpha.getValue().getType(), alpha.getValue(),
                                          rewriter.getUI32IntegerAttr(1), root.getEnable(), alpha.getValue());
    auto addrDelay =
        rewriter.create<spechls::DelayOp>(rewriter.getUnknownLoc(), alpha.getIndex().getType(), alpha.getIndex(),
                                          rewriter.getUI32IntegerAttr(1), root.getEnable(), alpha.getIndex());
    Value arrayInput = alpha.getArray();
    if (hasAlphaPred(arrayInput.getDefiningOp())) {
      auto arrDelay = rewriter.create<spechls::DelayOp>(rewriter.getUnknownLoc(), arrayInput.getType(), arrayInput,
                                                        rewriter.getUI32IntegerAttr(1), root.getEnable(), arrayInput);
      arrayInput = arrDelay.getResult();
    }

    auto result = rewriter.create<spechls::AlphaOp>(rewriter.getUnknownLoc(), arrayInput.getType(), arrayInput,
                                                    addrDelay, valDelay, weDelay);

    rewriter.replaceAllOpUsesWith(root, result);
    rewriter.eraseOp(root);
    return result;
  }

  static Operation *retimingRollbackImpl(PatternRewriter &rewriter, Operation *op) {
    auto root = cast<spechls::RollbackOp>(op);
    auto alpha = root.getInput().getDefiningOp<spechls::AlphaOp>();
    auto weRb = rewriter.create<spechls::RollbackOp>(rewriter.getUnknownLoc(), alpha.getWe().getType(),
                                                     root.getDepths(), root.getOffset(), alpha.getWe(),
                                                     root.getRollback(), root.getWriteCommand());
    auto valRb = rewriter.create<spechls::RollbackOp>(rewriter.getUnknownLoc(), alpha.getValue().getType(),
                                                      root.getDepths(), root.getOffset(), alpha.getValue(),
                                                      root.getRollback(), root.getWriteCommand());
    auto addrRb = rewriter.create<spechls::RollbackOp>(rewriter.getUnknownLoc(), alpha.getIndex().getType(),
                                                       root.getDepths(), root.getOffset(), alpha.getIndex(),
                                                       root.getRollback(), root.getWriteCommand());

    Value arrayInput = alpha.getArray();
    if (hasAlphaPred(arrayInput.getDefiningOp())) {
      auto arrRb = rewriter.create<spechls::RollbackOp>(rewriter.getUnknownLoc(), alpha.getArray().getType(),
                                                        root.getDepths(), root.getOffset(), alpha.getArray(),
                                                        root.getRollback(), root.getWriteCommand());
      arrayInput = arrRb.getResult();
    }

    auto result = rewriter.create<spechls::AlphaOp>(rewriter.getUnknownLoc(), arrayInput.getType(), arrayInput, addrRb,
                                                    valRb, weRb);
    rewriter.replaceAllOpUsesWith(root, result);
    rewriter.eraseOp(root);
    return result;
  }

  static bool hasAlphaPred(Operation *op) {
    if (op->getName().getStringRef() == spechls::AlphaOp::getOperationName().str()) {
      return true;
    }
    SmallVector<Operation *> stack;
    stack.push_back(op);
    while (!stack.empty()) {
      Operation *current = stack.back();
      stack.pop_back();
      for (auto pred : current->getOperands()) {
        Operation *cop = pred.getDefiningOp();
        if (!cop) {
          continue;
        }
        if (cop->getName().getStringRef() == spechls::AlphaOp::getOperationName().str()) {
          return true;
        }
        if (cop->getName().getStringRef() != spechls::MuOp::getOperationName().str()) {
          stack.push_back(cop);
        }
      }
    }
    return false;
  }
};
} // namespace
