//===- ExtractControl.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the UnrollInstr pass
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <deque>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_EXTRACTCONTROLPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace wcet {

struct ExtractControlPass : public impl::ExtractControlPassBase<ExtractControlPass> {

  using ExtractControlPassBase::ExtractControlPassBase;

public:
  void runOnOperation() override {
    // auto mod = getOperation();
    // IRRewriter rewriter(&getContext());
    //
    // spechls::TaskOp top = nullptr;
    // mod->walk([&](spechls::TaskOp t) {
    //   if (t->hasAttr("spechls.speculative"))
    //     top = t;
    // });
    //
    // if (!top) {
    //   return;
    // }
    //
    // size_t nbOp = 0;
    // top->walk([&](Operation *op) { nbOp++; });
    //
    // // Mark all operation that are in the path of each speculated gammas
    // SmallVector<spechls::GammaOp> speculatedGamma;
    // top->walk([&](spechls::GammaOp g) {
    //   for (auto operand : g.getInputs()) {
    //     if (!operand.getDefiningOp())
    //       continue;
    //     if (operand.getDefiningOp()->getName().getStringRef() == wcet::PenaltyOp::getOperationName().str()) {
    //       speculatedGamma.push_back(g);
    //       break;
    //     }
    //   }
    // });
    //
    // size_t counter = 0;
    // std::deque<Operation *> stack;
    // for (auto g : speculatedGamma) {
    //   if (!g.getSelect().getDefiningOp())
    //     continue;
    //   stack.push_back(g.getSelect().getDefiningOp());
    //   g.getSelect().getDefiningOp()->setAttr("wcet.inCtrlPath", rewriter.getUnitAttr());
    //   counter++;
    // }
    //
    // while (!stack.empty()) {
    //   Operation *current = stack.at(0);
    //   stack.pop_front();
    //   for (auto operand : current->getOperands()) {
    //     if (!operand.getDefiningOp())
    //       continue;
    //     Operation *next = operand.getDefiningOp();
    //     if (next->hasAttr("wcet.inCtrlPath"))
    //       continue;
    //     // if (next->getName().getStringRef() == spechls::LoadOp::getOperationName().str())
    //     //   continue;
    //     counter++;
    //     next->setAttr("wcet.inCtrlPath", rewriter.getUnitAttr());
    //     stack.push_back(next);
    //   }
    // }
    // llvm::errs() << "total before: " << nbOp << "\n";
    // llvm::errs() << "total after: " << counter << "\n";

    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext *ctx = mod.getContext();
    mlir::IRRewriter rewriter(ctx);

    spechls::PackOp fsmPack;
    mod->walk([&](spechls::FSMOp fsm) {
      fsmPack = mlir::dyn_cast_or_null<spechls::PackOp>(fsm.getMispec().getDefiningOp());
      return;
    });
    if (!fsmPack) {
      llvm::errs() << "Error: can not retrieve spechls::packOp\n";
      return;
    }
    // fsmPack->dumpPretty();
    std::vector<mlir::Operation *> ctrls;
    for (mlir::Value v : fsmPack->getOperands()) {
      if (!v.getDefiningOp()) {
        llvm::errs() << "Error: extract-control pass doesn't handle control as core's inputs\n";
        return;
      }
      ctrls.push_back(v.getDefiningOp());
    }
    // llvm::errs() << "ctrls size: " << ctrls.size() << "\n";
    // for (mlir::Operation *op : ctrls) {
    //   op->dumpPretty();
    // }

    // retrieve inputs vector for each control signal
    std::map<mlir::Operation *, std::vector<mlir::Value>> ctrlsInputs;
    for (mlir::Operation *ctrl : ctrls) {
      std::vector<mlir::Value> inputs;
      std::deque<mlir::Value> stack;
      for (mlir::Value v : ctrl->getOperands()) {
        stack.push_back(v);
      }
      while (!stack.empty()) {
        mlir::Value current = stack.at(0);
        stack.pop_front();
        if (!current.getDefiningOp() || current.getDefiningOp<spechls::LoadOp>()) {
          inputs.push_back(current);
          continue;
        }
        mlir::Operation *cOp = current.getDefiningOp();
        for (mlir::Value v : cOp->getOperands()) {
          stack.push_back(v);
        }
      }
      ctrlsInputs[ctrl] = std::move(inputs);
    }
    // for (std::pair<mlir::Operation *, std::vector<mlir::Value>> p : ctrlsInputs) {
    //   llvm::errs() << "input vector size: " << p.second.size() << "\n";
    //   for (mlir::Value v : p.second) {
    //     v.dump();
    //   }
    // }

    // pack together all dependend ctrl signal
    std::map<mlir::Operation *, unsigned int> opeToKey;
    std::map<unsigned int, std::vector<mlir::Operation *>> keyToOps;
    std::map<unsigned int, std::vector<mlir::Value>> keyToInputs;
    unsigned int nextKey = 0;
    for (unsigned int i = 0; i < ctrls.size(); i++) {
      mlir::Operation *currentOp = ctrls[i];
      unsigned int currentKey;
      std::vector<mlir::Operation *> opVector;
      std::vector<mlir::Value> valueSet;
      if (opeToKey.find(currentOp) != opeToKey.end()) {
        continue;
      }
      currentKey = nextKey++;
      opeToKey[currentOp] = currentKey;
      valueSet = std::move(ctrlsInputs[currentOp]);
      opVector.push_back(currentOp);
      for (unsigned int j = i + 1; j < ctrls.size(); j++) {
        mlir::Operation *compOp = ctrls[j];
        // compOp is in same the group that currentOp
        bool compHasKey = opeToKey.find(compOp) != opeToKey.end();
        if (compHasKey && opeToKey[compOp] == currentKey) {
          continue;
        }

        // compOp has either no group or is in a different group than currentOp
        bool hasCommonInput = false;
        std::vector<mlir::Value> diffVector;
        for (mlir::Value v0 : ctrlsInputs[compOp]) {
          bool isEqual = false;
          for (mlir::Value v1 : valueSet) {
            if (isEqual)
              break;
            isEqual = v1 == v0;
          }
          hasCommonInput = hasCommonInput || isEqual;
          if (!isEqual) {
            diffVector.push_back(v0);
          }
        }

        if (hasCommonInput) {
          if (compHasKey) {
            unsigned int compKey = opeToKey[compOp];
            std::vector<mlir::Value> compIns = keyToInputs[compKey];
            std::vector<mlir::Operation *> compOpVector = keyToOps[compKey];
            for (mlir::Value v : compIns) {
              bool isEqual = false;
              for (mlir::Value v1 : valueSet) {
                if (isEqual)
                  break;
                isEqual = v == v1;
              }
              if (!isEqual)
                valueSet.push_back(v);
            }
            for (mlir::Operation *op : compOpVector) {
              opVector.push_back(op);
            }
            for (mlir::Operation *op : keyToOps[compKey]) {
              opeToKey[op] = currentKey;
            }
            keyToInputs.erase(compKey);
            keyToOps.erase(compKey);
          } else {
            opeToKey[compOp] = currentKey;
            opVector.push_back(compOp);
            for (mlir::Value v : diffVector) {
              valueSet.push_back(v);
            }
          }
        }
      }
      keyToInputs[currentKey] = std::move(valueSet);
      keyToOps[currentKey] = std::move(opVector);
    }
    // llvm::errs() << "keyToOps size: " << keyToOps.size() << "\n";
    // llvm::errs() << "keyToInputs size: " << keyToInputs.size() << "\n";

    // create a new function for each group of ctrl signal with the
    // corresponding logic
    for (std::pair<unsigned int, std::vector<mlir::Operation *>> p : keyToOps) {
      rewriter.setInsertionPointToStart(mod.getBody());
      unsigned int key = p.first;
      std::string structName = "res_" + std::to_string(key);
      std::vector<std::string> elsName;
      std::vector<mlir::Type> elsType;

      std::vector<mlir::Operation *> ops = p.second;
      for (unsigned int i = 0; i < ops.size(); i++) {
        elsName.push_back("arg" + std::to_string(i));
        elsType.push_back(ops[i]->getResultTypes().front());
      }
      mlir::ArrayRef arrNames(std::move(elsName));
      mlir::ArrayRef arrTypes(std::move(elsType));
      spechls::StructType resType = rewriter.getType<spechls::StructType>(structName, arrNames, arrTypes);
      // resType.dump();

      std::vector<mlir::Type> inTypes;
      for (mlir::Value v : keyToInputs[key]) {
        inTypes.push_back(v.getType());
      }
      mlir::FunctionType fType = rewriter.getFunctionType(inTypes, resType);
      wcet::CoreOp ctrlFunc = rewriter.create<wcet::CoreOp>(rewriter.getUnknownLoc(), structName, fType);
      mlir::Block &funcBody = ctrlFunc.getBody().front();
      rewriter.setInsertionPointToStart(&funcBody);

      std::map<mlir::Operation *, mlir::Operation *> cloneMap;
      std::vector<mlir::Value> resVec;
      for (mlir::Operation *op : ops) {
        std::deque<mlir::Operation *> stack;
        stack.push_back(op);
        while (!stack.empty()) {
          mlir::Operation *cOp = stack.front();
          stack.pop_front();
          if (cloneMap.find(cOp) != cloneMap.end()) {
            continue;
          }
          cloneMap[cOp] = rewriter.clone(*cOp);

          for (mlir::Value v : cOp->getOperands()) {
            mlir::Operation *nOp = v.getDefiningOp();
            if (!nOp || nOp->getName().getStringRef().str() == spechls::LoadOp::getOperationName().str()) {
              continue;
            }
            stack.push_back(nOp);
          }
        }
        resVec.push_back(cloneMap[op]->getResult(0));
      }

      for (std::pair<mlir::Operation *, mlir::Operation *> p : cloneMap) {
        mlir::Operation *originalOp = p.first;
        mlir::Operation *clonedOp = p.second;
        for (unsigned int i = 0; i < clonedOp->getOperands().size(); i++) {
          mlir::Value oriOperand = originalOp->getOperand(i);
          mlir::Operation *operandOp = oriOperand.getDefiningOp();
          if (operandOp && operandOp->getName().getStringRef().str() != spechls::LoadOp::getOperationName().str())
            clonedOp->setOperand(i, cloneMap[operandOp]->getResult(0));
          else {
            for (unsigned int j = 0; j < keyToInputs[key].size(); j++) {
              if (oriOperand == keyToInputs[key][j]) {
                clonedOp->setOperand(i, funcBody.getArgument(j));
              }
            }
          }
        }
      }

      spechls::PackOp resPack = rewriter.create<spechls::PackOp>(rewriter.getUnknownLoc(), resType, resVec);
      rewriter.setInsertionPointToEnd(&funcBody);
      rewriter.create<wcet::CommitOp>(rewriter.getUnknownLoc(), resPack.getResult());

      rewriter.setInsertionPointAfter(ops[0]);
      wcet::CoreInstanceOp call =
          rewriter.create<wcet::CoreInstanceOp>(rewriter.getUnknownLoc(), resType, keyToInputs[key]);
      call.setCallee(structName);
      spechls::UnpackOp unpack = rewriter.create<spechls::UnpackOp>(rewriter.getUnknownLoc(), call->getResults());
      for (unsigned int i = 0; i < ops.size(); i++) {
        rewriter.replaceAllOpUsesWith(ops[i], unpack->getResult(i));
      }
    }

    // replace the ctrl logic with a call to the function
  }
};

} // namespace wcet
