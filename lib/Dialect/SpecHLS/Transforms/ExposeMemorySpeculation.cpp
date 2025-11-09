#include "circt/Dialect/Comb/CombDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <cstdio>
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
#include <string>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_EXPOSEMEMORYSPECULATIONPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "ExposeMemorySpeculation.h.inc"

namespace {
static SmallVector<SmallVector<Value>> delayValues(PatternRewriter &rewritter, SmallVector<Value> values, int depth,
                                                   std::string delayType, Value initDelay, Attribute scn);

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
    auto kernel = mu->getParentOfType<spechls::KernelOp>();
    auto maxSccAttr = dyn_cast_or_null<IntegerAttr>(kernel->getDiscardableAttr("spechls.max_scc"));
    int sccConstraint = 0;
    if (maxSccAttr != nullptr) {
      sccConstraint = maxSccAttr.getInt();
    }
    auto scn = rewritter.getIntegerAttr(rewritter.getI32Type(), sccConstraint);

    // Get the dependency distance
    auto dependencyDistance = cast<mlir::IntegerAttr>(mu->getDiscardableAttr("spechls.memspec")).getInt() - 1;

    // Get all LoadOp attach to the mu
    SmallVector<spechls::LoadOp> loads;
    SmallVector<Value> loadsAddresses;
    auto delayEnable = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(), 1);
    auto muDelay = rewritter.create<spechls::DelayOp>(rewritter.getUnknownLoc(), mu.getLoopValue(), 1, delayEnable,
                                                      mu.getInitValue());
    muDelay->setAttr("spechls.rollbackable", rewritter.getI32IntegerAttr(1));
    muDelay->setAttr("spechls.memspec", rewritter.getUnitAttr());
    muDelay->setAttr("spechls.scn", scn);
    for (auto *succ : mu.getResult().getUsers()) {
      if (succ->getName().getStringRef() == spechls::LoadOp::getOperationName()) {
        auto load = cast<spechls::LoadOp>(succ);
        auto loadAttr = load->getAttrDictionary();
        auto newLoad = rewritter.replaceOpWithNewOp<spechls::LoadOp>(load, muDelay, load.getIndex());
        newLoad->setAttrs(loadAttr);
        newLoad->setAttr("spechls.scn", scn);
        loads.push_back(newLoad);
        Value addr = newLoad.getIndex();
        if (newLoad.getIndex().getType().isSigned() || newLoad.getIndex().getType().isUnsigned()) {
          auto castOp = rewritter.create<circt::hw::BitcastOp>(
              rewritter.getUnknownLoc(), rewritter.getIntegerType(newLoad.getIndex().getType().getIntOrFloatBitWidth()),
              newLoad.getIndex());
          castOp->setAttr("spechls.scn", scn);
          addr = castOp.getResult();
        }
        loadsAddresses.push_back(addr);
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
          Value addr = alpha.getIndex();
          if (alpha.getIndex().getType().isSigned() || alpha.getIndex().getType().isUnsigned()) {
            auto castOp = rewritter.create<circt::hw::BitcastOp>(
                rewritter.getUnknownLoc(), rewritter.getIntegerType(alpha.getIndex().getType().getIntOrFloatBitWidth()),
                alpha.getIndex());
            castOp->setAttr("spechls.scn", scn);
            addr = castOp.getResult();
          }
          writeAddresses.push_back(addr);
          writeEnables.push_back(alpha.getWe());
          alpha->setAttr("spechls.memspec", rewritter.getUnitAttr());
          alpha->setAttr("spechls.scn", scn);
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
    auto initDelay = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(), 1);
    initDelay->setAttr("spechls.scn", scn);
    auto delayedWritesAddresses = delayValues(rewritter, writeAddresses, dependencyDistance, "", nullptr, scn);
    auto delayedWE = delayValues(rewritter, writeEnables, dependencyDistance, "spechls.cancellable", initDelay, scn);

    // build the comparaison logic ((@Read = @Write) && WE) for:
    //    Each dependencies distances for:
    //     Each Read addresses for:
    //      Each Write addresse
    Value aliasGammaResult = nullptr;
    for (int i = dependencyDistance - 1; i >= 0; i--) {
      writeAddresses = delayedWritesAddresses.data()[i];
      writeEnables = delayedWE.data()[i];
      Value aliasCheck = nullptr;
      for (auto load : loadsAddresses) {
        Value lastOpResult = nullptr;
        for (size_t j = 0; j < writeAddresses.size(); j++) {
          auto write = writeAddresses.data()[j];
          auto we = writeEnables.data()[j];
          if (load.getType() != write.getType()) {
            size_t loadSize = load.getType().getIntOrFloatBitWidth();
            size_t writeSize = write.getType().getIntOrFloatBitWidth();
            if (writeSize > loadSize) {
              auto consSize = rewritter.create<circt::hw::ConstantOp>(
                  rewritter.getUnknownLoc(), rewritter.getIntegerType(writeSize - loadSize), 0);
              consSize->setAttr("spechls.scn", scn);
              auto concat = rewritter.create<circt::comb::ConcatOp>(rewritter.getUnknownLoc(), consSize, load);
              concat->setAttr("spechls.scn", scn);
              load = concat.getResult();
            } else {
              auto consSize = rewritter.create<circt::hw::ConstantOp>(
                  rewritter.getUnknownLoc(), rewritter.getIntegerType(loadSize - writeSize), 0);
              consSize->setAttr("spechls.scn", scn);
              auto concat = rewritter.create<circt::comb::ConcatOp>(rewritter.getUnknownLoc(), consSize, write);
              concat->setAttr("spechls.scn", scn);
              write = concat.getResult();
            }
          }
          auto comp = rewritter.create<circt::comb::ICmpOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(),
                                                            circt::comb::ICmpPredicate::eq, write, load);
          comp->setAttr("spechls.scn", scn);
          auto andWe = rewritter.create<circt::comb::AndOp>(rewritter.getUnknownLoc(), comp, we);
          andWe->setAttr("spechls.scn", scn);
          if (lastOpResult != nullptr) {
            lastOpResult =
                rewritter.create<circt::comb::OrOp>(rewritter.getUnknownLoc(), lastOpResult, andWe).getResult();
            lastOpResult.getDefiningOp()->setAttr("spechls.scn", scn);
          } else {
            lastOpResult = andWe.getResult();
          }
        }
        if (aliasCheck != nullptr) {
          aliasCheck =
              rewritter.create<circt::comb::OrOp>(rewritter.getUnknownLoc(), lastOpResult, aliasCheck).getResult();
          aliasCheck.getDefiningOp()->setAttr("spechls.scn", scn);
        } else {
          aliasCheck = lastOpResult;
        }
      }
      llvm::errs() << aliasCheck.getType() << "\n";

      // Build the gamma for the distance i + 1
      std::string gammaName = "alias_check_" + mu.getSymName().str() + "_distance_" + std::to_string(i + 1);
      Value inputFalse;
      // Case no alias -> aliasCheck = 0;
      if (aliasGammaResult != nullptr) {
        inputFalse = aliasGammaResult;
      } else {
        auto firstConst = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(),
                                                                  rewritter.getI32IntegerAttr(dependencyDistance));
        firstConst->setAttr("spechls.scn", scn);
        inputFalse = firstConst.getResult();
      }
      // Case alias -> aliasCheck = 1;
      auto cons = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(), rewritter.getI32IntegerAttr(i));
      cons->setAttr("spechls.scn", scn);

      // aliasGammaResult = rewritter
      //                        .create<spechls::GammaOp>(rewritter.getUnknownLoc(), rewritter.getI32Type(),
      //                                                  rewritter.getStringAttr(gammaName), aliasCheck, gammaInputs)
      //                        .getResult();
      aliasGammaResult = rewritter.create<circt::comb::MuxOp>(rewritter.getUnknownLoc(), rewritter.getI32Type(),
                                                              aliasCheck, cons.getResult(), inputFalse);
      aliasGammaResult.getDefiningOp()->setAttr("spechls.scn", scn);
    }

    //
    // Build the memory speculation GammaOp
    //

    std::string gammaName = "memory_speculation_" + mu.getSymName().str();
    // Delay the output the mu
    SmallVector<Value> gammaInputs;
    Value lastInputs = muDelay.getResult();
    gammaInputs.push_back(lastInputs);
    for (int i = 0; i < dependencyDistance; i++) {
      auto input = rewritter.create<spechls::DelayOp>(rewritter.getUnknownLoc(), lastInputs.getType(), lastInputs, 1,
                                                      delayEnable, mu.getInitValue());
      input->setAttr("spechls.rollbackable", rewritter.getI32IntegerAttr(i + 2));
      input->setAttr("spechls.memspec", rewritter.getUnitAttr());
      input->setAttr("spechls.scn", scn);
      gammaInputs.push_back(input.getResult());
      lastInputs = input.getResult();
    }

    // Build the gamma
    auto memSpecGamma =
        rewritter.create<spechls::GammaOp>(rewritter.getUnknownLoc(), lastInputs.getType(),
                                           rewritter.getStringAttr(gammaName), aliasGammaResult, gammaInputs);
    memSpecGamma->setAttr("spechls.memspec", rewritter.getUnitAttr());
    memSpecGamma->setAttr("spechls.scn", scn);

    //
    // Replace the array input of all LoadOp attach to the mu with the output of the gamma
    //
    for (auto load : loads) {
      load.setOperand(0, memSpecGamma.getResult());
    }

    auto result = rewritter.create<spechls::MuOp>(rewritter.getUnknownLoc(), mu.getSymName(), mu.getInitValue(),
                                                  mu.getLoopValue());
    result->setAttr("spechls.scn", scn);
    rewritter.replaceAllUsesWith(mu, result);

    kernel->setAttr("spechls.max_scc", rewritter.getI32IntegerAttr(scn.getInt() + 1));
    return result;
  }
};

SmallVector<SmallVector<Value>> delayValues(PatternRewriter &rewritter, SmallVector<Value> values, int depth,
                                            std::string delayType, Value initDelay, Attribute scn) {
  SmallVector<SmallVector<Value>> result;
  if (values.empty())
    return result;
  auto enable = rewritter.create<circt::hw::ConstantOp>(rewritter.getUnknownLoc(), rewritter.getI1Type(), 1);
  enable->setAttr("spechls.scn", scn);
  for (int i = 1; i <= depth; i++) {
    SmallVector<Value> newVals;
    for (auto val : values) {
      auto newDel =
          rewritter.create<spechls::DelayOp>(rewritter.getUnknownLoc(), val.getType(), val, i, enable, initDelay);
      if (!delayType.empty())
        newDel->setAttr(delayType, rewritter.getI32IntegerAttr(depth - i));
      // newDel->setAttr("spechls.memspec", rewritter.getUnitAttr());
      newDel->setAttr("spechls.scn", scn);
      newVals.push_back(newDel);
    }
    result.push_back(newVals);
    // values = newVals;
  }

  return result;
}
} // namespace
