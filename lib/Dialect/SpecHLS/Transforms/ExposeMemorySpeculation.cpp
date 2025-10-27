#include "circt/Dialect/Comb/CombDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
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
static SmallVector<SmallVector<Value>> *delayValues(PatternRewriter &rewritter, SmallVector<Value> values, int depth);

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
    patterns.getPDLPatterns().registerRewriteFunction("ExposeMemorySpeculation", exposeMemorySpeculationImpl);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Operation *exposeMemorySpeculationImpl(PatternRewriter &rewritter, Operation *op) {
    auto mu = cast<spechls::MuOp>(op);

    // Get the dependency distance
    auto dependencyDistance = cast<mlir::IntegerAttr>(mu->getDiscardableAttr("dependenciesDistances")).getInt();

    // Get all LoadOp attach to the mu
    SmallVector<spechls::LoadOp> loads;
    SmallVector<Value> loadsAddresses;
    for (auto *succ : mu.getResult().getUsers()) {
      if (succ->getName().getStringRef() == spechls::LoadOp::getOperationName()) {
        auto load = cast<spechls::LoadOp>(succ);
        loads.push_back(load);
        loadsAddresses.push_back(load.getIndex());
      }
    }

    // Get write addresse and WE of all AlphaOps dominated by the MuOp
    SmallVector<Value> writeAddresses;
    SmallVector<Value> writeEnables;
    SmallVector<Operation *> toCheck;
    toCheck.push_back(mu);
    while (!toCheck.empty()) {
      Operation *current = toCheck.back();
      toCheck.pop_back();
      for (auto *succ : current->getResults().getUsers()) {
        if (succ->getName().getStringRef() == spechls::AlphaOp::getOperationName()) {
          auto alpha = cast<spechls::AlphaOp>(succ);
          writeAddresses.push_back(alpha.getIndex());
          writeEnables.push_back(alpha.getWe());
          toCheck.push_back(succ);
        } else if (succ->getName().getStringRef() == spechls::GammaOp::getOperationName()) {
          toCheck.push_back(succ);
        }
      }
    }

    //
    // Build the alias detection logic
    //

    // Delayed Write addresses and WE
    auto *delayedWritesAddresses = delayValues(rewritter, writeAddresses, dependencyDistance);
    auto *delayedWE = delayValues(rewritter, writeEnables, dependencyDistance);

    // build the comparaison logic ((@Read = @Write) && WE) for:
    //    Each dependencies distances for:
    //     Each Read addresses for:
    //      Each Write addresse
    Value aliasGammaResult = nullptr;
    for (int i = dependencyDistance - 1; i >= 0; i--) {
      writeAddresses = delayedWritesAddresses->data()[i];
      writeEnables = delayedWE->data()[i];
      Value aliasCheck = nullptr;
      for (auto load : loadsAddresses) {
        Value lastOpResult = nullptr;
        for (size_t j = 0; j < writeAddresses.size(); j++) {
          auto write = writeAddresses.data()[j];
          auto we = writeEnables.data()[j];
          auto comp = rewritter.create<circt::comb::ICmpOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(),
                                                            circt::comb::ICmpPredicate::eq, write, load);
          auto andWe = rewritter.create<circt::comb::AndOp>(rewritter.getUnknownLoc(), comp, we);
          if (lastOpResult != nullptr) {
            lastOpResult =
                rewritter.create<circt::comb::OrOp>(rewritter.getUnknownLoc(), lastOpResult, andWe).getResult();
          } else {
            lastOpResult = andWe.getResult();
          }
        }
        if (aliasCheck != nullptr) {
          aliasCheck =
              rewritter.create<circt::comb::OrOp>(rewritter.getUnknownLoc(), lastOpResult, aliasCheck).getResult();
        } else {
          aliasCheck = lastOpResult;
        }
      }

      // Build the gamma for the distance i + 1
      SmallVector<Value> gammaInputs;
      // Case no alias -> aliasCheck = 0;
      if (aliasGammaResult != nullptr) {
        gammaInputs.push_back(aliasGammaResult);
      } else {
        auto firstConst = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(),
                                                                  rewritter.getI32IntegerAttr(dependencyDistance));
        gammaInputs.push_back(firstConst);
      }
      // Case alias -> aliasCheck = 1;
      auto cons = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(), rewritter.getI32IntegerAttr(i));
      gammaInputs.push_back(cons);

      aliasGammaResult = rewritter
                             .create<spechls::GammaOp>(rewritter.getUnknownLoc(), rewritter.getI32Type(),
                                                       rewritter.getStringAttr("aliasCheck"), aliasCheck, gammaInputs)
                             .getResult();
    }

    //
    // Build the memory speculation GammaOp
    //

    // Delay the output the mu
    SmallVector<Value> gammaInputs;
    Value lastInputs = mu.getResult();
    gammaInputs.push_back(lastInputs);
    auto delayEnable = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(), 1);
    for (int i = 0; i < dependencyDistance; i++) {
      auto input = rewritter.create<spechls::DelayOp>(rewritter.getUnknownLoc(), lastInputs.getType(), lastInputs, 1,
                                                      delayEnable, mu.getInitValue());
      gammaInputs.push_back(input.getResult());
      lastInputs = input.getResult();
    }

    // Build the gamma
    auto memSpecGamma =
        rewritter.create<spechls::GammaOp>(rewritter.getUnknownLoc(), lastInputs.getType(),
                                           rewritter.getStringAttr("memSpec"), aliasGammaResult, gammaInputs);

    //
    // Replace the array input of all LoadOp attach to the mu with the output of the gamma
    //
    for (auto load : loads) {
      load.setOperand(0, memSpecGamma.getResult());
    }

    auto result = rewritter.create<spechls::MuOp>(rewritter.getUnknownLoc(), mu.getSymName(), mu.getInitValue(),
                                                  mu.getLoopValue());
    rewritter.replaceAllUsesWith(mu, result);

    return result;
  }
};

SmallVector<SmallVector<Value>> *delayValues(PatternRewriter &rewritter, SmallVector<Value> values, int depth) {
  SmallVector<SmallVector<Value>> *result = new SmallVector<SmallVector<Value>>();
  auto enable = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(), 1);
  for (int i = 0; i < depth; i++) {
    SmallVector<Value> newVals;
    for (auto val : values) {
      newVals.push_back(
          rewritter.create<spechls::DelayOp>(rewritter.getUnknownLoc(), val.getType(), val, 1, enable, val));
    }
    result->push_back(newVals);
    values = newVals;
  }

  return std::move(result);
}
} // namespace
