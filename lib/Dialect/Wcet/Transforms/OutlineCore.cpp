//===- OutlineCore.cpp  ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the OutlineCore pass
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

#include <Dialect/SpecHLS/IR/SpecHLSOps.h>
#include <Dialect/Wcet/IR/WcetOps.h>
#include <cstddef>
#include <string>

#define DEBUG_TYPE "OutlineCore"

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_OUTLINECOREPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct OutlineCorePass : public impl::OutlineCorePassBase<OutlineCorePass> {

  using OutlineCorePassBase::OutlineCorePassBase;

public:
  void runOnOperation() override {
    auto mod = getOperation();
    IRRewriter rewriter(&getContext());

    /**************************************************************************
     * Get the speculative Task                                               *
     *************************************************************************/
    spechls::TaskOp speculativeTask = nullptr;
    mod->walk([&](spechls::TaskOp t) {
      if (t->getAttr("spechls.speculative")) {
        speculativeTask = t;
        return;
      }
    });

    if (!speculativeTask) {
      LLVM_DEBUG({ LDBG() << "No speculative spechls.task found"; });
      signalPassFailure();
    }

    if (!speculativeTask->getOpOperands().empty()) {
      LLVM_DEBUG({ LDBG() << "speculative task must have no inputs"; });
    }

    /**************************************************************************
     * Build inputs/outputs of the wcet.core                                  *
     *************************************************************************/
    SmallVector<Operation *> firstsDelay;
    getNbDelayPred(rewriter, speculativeTask, firstsDelay);

    SmallVector<Type> coreInputsTypes;
    SmallVector<DictionaryAttr> coreInputsAttrs;
    SmallVector<Type> coreOutputsTypes;
    SmallVector<Value> coreInputs;
    SmallVector<Operation *> coreOutputs;
    SmallVector<Operation *> toRemove;

    //============= Retrieve instructions types ===============================
    size_t numInstrs = 0;
    speculativeTask->walk([&](Operation *op) {
      auto fetch = op->getAttr("wcet.fetch");
      if (!fetch)
        return;
      coreInputsTypes.push_back(op->getResultTypes().front());
      auto fetchNumber = dyn_cast_or_null<IntegerAttr>(fetch);
      if (!fetchNumber) {
        fetchNumber = rewriter.getI32IntegerAttr(0);
      }
      coreInputsAttrs.push_back(rewriter.getDictionaryAttr(rewriter.getNamedAttr("wcet.instrNb", fetchNumber)));
      coreInputs.push_back(op->getResult(0));
      toRemove.push_back(op);
      numInstrs++;
    });

    //============= Retrieve Mu ===============================================
    speculativeTask->walk([&](spechls::MuOp mu) {
      coreInputsTypes.push_back(mu.getType());
      coreOutputsTypes.push_back(mu.getType());
      coreInputsAttrs.push_back(rewriter.getDictionaryAttr({}));
      coreInputs.push_back(mu);
      coreOutputs.push_back(mu.getLoopValue().getDefiningOp());
      toRemove.push_back(mu);
    });

    //============= Retrieve Delays ===========================================
    for (auto *d : firstsDelay) {
      coreInputs.push_back(d->getResult(0));
      coreInputsTypes.push_back(d->getResult(0).getType());
      coreOutputsTypes.push_back(d->getResult(0).getType());
      coreInputsAttrs.push_back(
          rewriter.getDictionaryAttr(rewriter.getNamedAttr("wcet.nbPred", rewriter.getI32IntegerAttr(0))));
      coreOutputs.push_back(d->getOperand(0).getDefiningOp());
      toRemove.push_back(d);
      spechls::DelayOp nextDelay = nullptr;
      for (auto *op : d->getUsers()) {
        auto nd = dyn_cast_or_null<spechls::DelayOp>(op);
        if (nd) {
          nextDelay = nd;
          break;
        }
      }
      int nbPred = 0;
      while (nextDelay) {
        coreInputs.push_back(nextDelay);
        coreInputsTypes.push_back(nextDelay.getType());
        coreOutputsTypes.push_back(nextDelay.getType());
        coreInputsAttrs.push_back(
            rewriter.getDictionaryAttr(rewriter.getNamedAttr("wcet.nbPred", rewriter.getI32IntegerAttr(++nbPred))));
        coreOutputs.push_back(nextDelay.getInput().getDefiningOp());
        toRemove.push_back(nextDelay);
        auto nextUsers = nextDelay->getUsers();
        nextDelay = nullptr;
        for (auto *op : nextUsers) {
          auto nd = dyn_cast_or_null<spechls::DelayOp>(op);
          if (nd) {
            nextDelay = nd;
            break;
          }
        }
      }
    }

    /**************************************************************************
     * Create the wcet.core                                                   *
     *************************************************************************/
    rewriter.setInsertionPointToStart(mod.getBody());
    auto funType = rewriter.getFunctionType(coreInputsTypes, coreOutputsTypes);
    auto core =
        rewriter.create<wcet::CoreOp>(rewriter.getUnknownLoc(), funType,
                                      rewriter.getStringAttr(std::string("core_" + speculativeTask.getSymName().str())),
                                      ArrayRef<DictionaryAttr>(coreInputsAttrs));

    Block &coreBody = core.getBody().front();

    /**************************************************************************
     *  Populate core                                                         *
     **************************************************************************/
    DenseMap<Operation *, Operation *> cloneMap;

    speculativeTask->walk([&](Operation *op) {
      rewriter.setInsertionPointToEnd(&coreBody);
      auto opName = op->getName().getStringRef();
      if (opName == spechls::TaskOp::getOperationName().str() ||
          opName == spechls::CommitOp::getOperationName().str()) {
        return;
      }
      auto *newOp = rewriter.clone(*op);
      cloneMap.try_emplace(op, newOp);
    });

    /**************************************************************************
     *  Handle Inputs/Outputs                                                 *
     **************************************************************************/
    speculativeTask->walk([&](Operation *op) {
      auto opName = op->getName().getStringRef();
      if (opName == spechls::CommitOp::getOperationName().str() ||
          opName == spechls::TaskOp::getOperationName().str()) {
        return;
      }
      for (size_t i = 0; i < op->getNumOperands(); i++) {
        auto operand = op->getOperand(i);
        auto *opOperand = operand.getDefiningOp();
        for (size_t j = 0; j < opOperand->getNumResults(); j++) {
          if (opOperand->getResult(j) == operand) {
            cloneMap[op]->setOperand(i, cloneMap[opOperand]->getResult(j));
          }
        }
      }
    });

    SmallVector<Value> outs;
    for (size_t i = 0; i < coreOutputs.size(); i++) {
      auto *op = cloneMap[coreOutputs[i]];
      for (size_t j = 0; j < op->getNumResults(); j++) {
        outs.push_back(op->getResult(j));
      }
    }

    auto commitOp = rewriter.create<wcet::CommitOp>(rewriter.getUnknownLoc(), outs);

    for (size_t i = 0; i < coreInputs.size(); i++) {
      speculativeTask->walk([&](Operation *op) {
        for (size_t j = 0; j < op->getNumOperands(); j++) {
          if (op->getOperand(j) == coreInputs[i]) {
            cloneMap[op]->setOperand(j, core.getBody().getArgument(i));
          }
        }
      });
    }

    for (size_t i = 0; i < commitOp->getNumOperands(); i++) {
      auto operand = commitOp.getOperand(i);
      auto *opOperand = operand.getDefiningOp();
      size_t operandNum = 0;
      for (size_t j = 0; j < opOperand->getNumResults(); j++) {
        if (opOperand->getResult(j) == operand) {
          operandNum = j;
          break;
        }
      }
      speculativeTask->walk([&](Operation *op) {
        if (cloneMap[op] == opOperand) {
          auto taskOperand = op->getResult(operandNum);
          for (size_t j = 0; j < coreInputs.size(); j++) {
            if (coreInputs[j] == taskOperand) {
              commitOp.setOperand(i, core.getBody().getArgument(j));
            }
          }
        }
      });
    }

    for (auto *p : toRemove) {
      rewriter.eraseOp(cloneMap[p]);
    }

    core->setAttr("wcet.cpuCore", rewriter.getUnitAttr());
    core->setAttr("wcet.numInstrs", rewriter.getUI32IntegerAttr(numInstrs));
  }

private:
  void getNbDelayPred(OpBuilder &builder, spechls::TaskOp top, SmallVector<Operation *> &firstsDelay) {
    for (auto d : top.getOps<spechls::DelayOp>()) {
      bool isDelaySucc = llvm::TypeSwitch<Operation *, bool>(d.getInput().getDefiningOp())
                             .Case<spechls::DelayOp, spechls::RollbackableDelayOp, spechls::CancellableDelayOp>(
                                 [&](auto d) { return true; })
                             .Default(false);
      if (isDelaySucc) {
        continue;
      }

      firstsDelay.push_back(d);
    }
    for (auto d : top.getOps<spechls::CancellableDelayOp>()) {
      bool isDelaySucc = llvm::TypeSwitch<Operation *, bool>(d.getInput().getDefiningOp())
                             .Case<spechls::DelayOp, spechls::RollbackableDelayOp, spechls::CancellableDelayOp>(
                                 [&](auto d) { return true; })
                             .Default(false);
      if (isDelaySucc) {
        continue;
      }

      firstsDelay.push_back(d);
    }
    for (auto d : top.getOps<spechls::RollbackableDelayOp>()) {
      bool isDelaySucc = llvm::TypeSwitch<Operation *, bool>(d.getInput().getDefiningOp())
                             .Case<spechls::DelayOp, spechls::RollbackableDelayOp, spechls::CancellableDelayOp>(
                                 [&](auto d) { return true; })
                             .Default(false);
      if (isDelaySucc) {
        continue;
      }

      firstsDelay.push_back(d);
    }
  }
};

} // namespace wcet
