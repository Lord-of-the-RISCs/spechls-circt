//===- OutlineCore.cpp  ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the OutlineCore pass
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

#include <Dialect/SpecHLS/IR/SpecHLSOps.h>
#include <Dialect/Wcet/IR/WcetOps.h>
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
    OpBuilder builder = OpBuilder(&getContext());

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

    /**************************************************************************
     * Give an unique number to each delayOp in the speculativeTask-          *
     *************************************************************************/
    int delaysNumber = 0;
    speculativeTask->walk(
        [&](spechls::DelayOp delays) { delays->setAttr("wcet.num", builder.getI32IntegerAttr(delaysNumber++)); });

    /**************************************************************************
     * Retrieve core input's types, fetch and attributes                      *
     *************************************************************************/
    bool isResultsPacked = !speculativeTask.getOps<spechls::PackOp>().empty();

    SmallVector<Type> funInputsTypes = SmallVector<Type>(speculativeTask.getOperandTypes());
    SmallVector<Type> funOutputsTypes;
    SmallVector<std::string> funOutputsNames;
    SmallVector<DictionaryAttr> inputsAttrs;
    for (size_t i = 0; i < funInputsTypes.size(); ++i) {
      inputsAttrs.push_back(DictionaryAttr());
    }
    bool fetchFound = false;
    int delCount = 0;
    int muCount = 0;
    int packCount = 0;
    speculativeTask->walk([&](Operation *op) {
      if (op->getAttr("wcet.fetch")) {
        op->setAttr("wcet.toReplace", builder.getI64IntegerAttr(funInputsTypes.size()));
        funInputsTypes.push_back(op->getResultTypes().front());
        inputsAttrs.push_back(builder.getDictionaryAttr(builder.getNamedAttr("wcet.instrs", builder.getUnitAttr())));
        fetchFound = true;
        return;
      }
      auto opName = op->getName().getStringRef();
      if (opName == spechls::DelayOp::getOperationName().str()) {
        auto delay = cast<spechls::DelayOp>(op);
        auto *predOp = delay->getPrevNode();
        if (predOp->getName().getStringRef() == spechls::DelayOp::getOperationName().str()) {
          auto delNumber = llvm::dyn_cast_or_null<IntegerAttr>(predOp->getAttr("wcet.num"));
          inputsAttrs.push_back(builder.getDictionaryAttr(builder.getNamedAttr("wcet.pred", delNumber)));
        } else {
          inputsAttrs.push_back(DictionaryAttr());
        }
        op->setAttr("wcet.toReplace", builder.getI64IntegerAttr(funInputsTypes.size()));
        delay.getInput().getDefiningOp()->setAttr("wcet.output", builder.getUnitAttr());
        funOutputsTypes.push_back(delay.getInput().getType());
        funOutputsNames.push_back("delay_" + std::to_string(delCount++));
        funInputsTypes.push_back(op->getResultTypes().front());
      }
      if (opName == spechls::MuOp::getOperationName().str()) {
        auto mu = cast<spechls::MuOp>(op);
        inputsAttrs.push_back(DictionaryAttr());
        op->setAttr("wcet.toReplace", builder.getI64IntegerAttr(funInputsTypes.size()));
        mu.getLoopValue().getDefiningOp()->setAttr("wcet.output", builder.getUnitAttr());
        funOutputsTypes.push_back(mu.getType());
        funOutputsNames.push_back("mu_" + std::to_string(muCount++));
        funInputsTypes.push_back(op->getResultTypes().front());
      }
      if (opName == spechls::PackOp::getOperationName().str()) {
        auto pack = cast<spechls::PackOp>(op);
        for (auto v : pack.getInputs()) {
          v.getDefiningOp()->setAttr("wcet.output", builder.getUnitAttr());
          funOutputsNames.push_back("pack_" + std::to_string(packCount++));
          funOutputsTypes.push_back(v.getType());
        }
      }
      if (opName == spechls::CommitOp::getOperationName().str() && !isResultsPacked) {
        auto commit = cast<spechls::CommitOp>(op);
        funOutputsNames.push_back("result");
        funOutputsTypes.push_back(commit.getValue().front().getType());
        commit->getPrevNode()->setAttr("wcet.output", builder.getUnitAttr());
      }
    });

    if (!fetchFound) {
      LLVM_DEBUG({ LDBG() << "No fetch found"; });
      signalPassFailure();
    }

    /**************************************************************************
     * Create the core Operation                                              *
     *************************************************************************/
    builder.setInsertionPointToStart(mod.getBody());
    auto funType = builder.getFunctionType(
        funInputsTypes, builder.getType<spechls::StructType>("out", funOutputsNames, funOutputsTypes));
    auto core =
        builder.create<wcet::CoreOp>(builder.getUnknownLoc(), funType,
                                     builder.getStringAttr(std::string("core_" + speculativeTask.getSymName().str())),
                                     ArrayRef<DictionaryAttr>(inputsAttrs));

    builder.setInsertionPointToStart(&core.getBody().getBlocks().front());
    IRMapping mapper;
    for (const auto &it : llvm::enumerate(speculativeTask.getBody().getArguments())) {
      mapper.map(it.value(), core.getBody().getArgument(it.index()));
    }

    SmallVector<Value> outputsValue;
    speculativeTask->walk([&](Operation *op) {
      auto opName = op->getName().getStringRef();
      if (opName == spechls::TaskOp::getOperationName().str() ||
          opName == spechls::CommitOp::getOperationName().str() ||
          opName == spechls::PackOp::getOperationName().str()) {
        return;
      }
      auto *newOp = builder.clone(*op, mapper);
      if (newOp->getAttr("wcet.output")) {
        outputsValue.push_back(newOp->getResult(0));
      }
    });

    auto pack = builder.create<spechls::PackOp>(builder.getUnknownLoc(), funType.getResults(), outputsValue);
    builder.create<wcet::CommitOp>(builder.getUnknownLoc(), pack);

    core->walk([&](Operation *op) {
      auto attr = dyn_cast_or_null<IntegerAttr>(op->getAttr("wcet.toReplace"));
      if (attr) {
        int64_t idx = attr.getInt();
        op->getResult(0).replaceAllUsesWith(core.getBody().getArgument(idx));
        op->erase();
      }
    });
  }
};

} // namespace wcet
