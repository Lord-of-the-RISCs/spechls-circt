#include "circt/Dialect/Comb/CombDialect.h"
#include "mlir/IR/Operation.h"
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
#define GEN_PASS_DEF_EXPOSEMEMORYSPECULATIONPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "ExposeMemorySpeculation.h.inc"

namespace {

struct ExposeMemorySpeculationPass
    : public spechls::impl::ExposeMemorySpeculationPassBase<ExposeMemorySpeculationPass> {
  FrozenRewritePatternSet patterns;

  using ExposeMemorySpeculationPassBase::ExposeMemorySpeculationPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("ExposeMemorySpeculation", ExposeMemorySpeculationImpl);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Operation *ExposeMemorySpeculationImpl(PatternRewriter &rewritter, Operation *op) { return op; }
};
} // namespace
