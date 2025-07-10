//===- UnrollInstr.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
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

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>

using namespace mlir;
using namespace spechls;
using namespace circt;

namespace wcet {
#define GEN_PASS_DEF_UNROLLINSTRPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace {
bool hasPragmaContaining(Operation *op, llvm::StringRef keyword) {

  auto attr = op->getAttr(StringRef("#pragma"));
  if (attr != NULL) {
    if (auto strAttr = dyn_cast<mlir::StringAttr>(attr)) {
      // Compare the attribute value with an existing string
      if (strAttr.getValue().contains(keyword)) {
        return true;
      }
    }
  }
  return false;
}

// Setup the HWModuleOp to be unrolled
StructType setupTask(TaskOp top, PatternRewriter &rewriter) {
  auto savedIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&top.getBody().getBlocks().front());
  top->walk([&](Operation *op) {
    if (hasPragmaContaining(op, "WCET fetch")) {
      hw::ConstantOp inst = rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, rewriter.getI32IntegerAttr(0));
      inst->setAttr(rewriter.getStringAttr("wcet.pragma"), rewriter.getStringAttr("instruction"));
      return;
    }
  });

  CommitOp commit = *top.getBody().getOps<CommitOp>().begin();
  if (!commit)
    return nullptr;

  SmallVector<Value> outputs = SmallVector<Value>();
  SmallVector<Type> outputsType = SmallVector<Type>();
  SmallVector<std::string> outputsName = SmallVector<std::string>();

  outputs.push_back(commit.getValue());
  outputsType.push_back(commit.getValue().getType());
  outputsName.push_back("originalOutput");

  top.getBody().getArgument(0).getUsers().begin()->getName();

  for (size_t i = 0; i < top->getOperands().size(); i++) {
    auto in = top.getBody().getArgument(i);
    mlir::Value val = dyn_cast<mlir::Value>(in);
    for (auto *use : in.getUsers()) {
      if (use->getName().getStringRef() == spechls::MuOp::getOperationName()) {
        auto mu = dyn_cast_or_null<spechls::MuOp>(use);
        rewriter.replaceAllOpUsesWith(mu, mu.getInitValue());
        val = mu.getLoopValue();
        break;
      }
    }
    outputs.push_back(val);
    outputsType.push_back(val.getType());
    outputsName.push_back("input" + std::to_string(i));
  }

  int i = 0;
  top->walk([&](DelayOp delay) {
    outputs.push_back(delay.getInput());
    outputsName.push_back("delay_" + std::to_string(i));
    outputsType.push_back(delay.getType());
    delay->setAttr(rewriter.getStringAttr("wcet.pragma"), rewriter.getStringAttr("delay_" + std::to_string(i++)));
  });

  rewriter.setInsertionPointAfter(commit);
  auto resultStruct =
      rewriter.getType<spechls::StructType>(rewriter.getStringAttr("result_struct"), outputsName, outputsType);
  PackOp pack = rewriter.create<PackOp>(rewriter.getUnknownLoc(), resultStruct, outputs);
  rewriter.replaceOpWithNewOp<CommitOp>(commit, commit.getEnable(), pack.getResult());
  top.getResult().setType(resultStruct);
  rewriter.restoreInsertionPoint(savedIP);
  return resultStruct;
}
} // namespace

namespace wcet {

struct UnrollInstrPattern : OpRewritePattern<spechls::KernelOp> {
  llvm::ArrayRef<unsigned int> instrs;
  using OpRewritePattern<spechls::KernelOp>::OpRewritePattern;

  // Constructor to save pass arguments
  UnrollInstrPattern(MLIRContext *ctx, const llvm::ArrayRef<unsigned int> intrs)
      : OpRewritePattern<spechls::KernelOp>(ctx) {
    this->instrs = intrs;
  }

  LogicalResult matchAndRewrite(spechls::KernelOp top, PatternRewriter &rewriter) const override {
    if (instrs.size() == 0)
      return failure();
    TaskOp mainTask = nullptr;
    top->walk([&](TaskOp task) {
      task.getBody().walk([&](Operation *op) {
        if (hasPragmaContaining(op, "WCET fetch")) {
          mainTask = task;
          return;
        }
      });
    });

    if (!mainTask) {
      return failure();
    }

    StructType resultStruct = setupTask(mainTask, rewriter);
    if (!resultStruct)
      return failure();

    rewriter.setInsertionPointAfter(mainTask);
    spechls::UnpackOp lastUnpack = rewriter.create<spechls::UnpackOp>(rewriter.getUnknownLoc(), mainTask.getResult());
    rewriter.replaceAllUsesExcept(mainTask.getResult(), lastUnpack->getResult(0), lastUnpack);

    size_t numArgsOri = mainTask->getNumOperands();
    auto rTypes = resultStruct.getFieldTypes();
    auto rNames = resultStruct.getFieldNames();
    rewriter.setInsertionPointToStart(&top.getBody().getBlocks().front());
    for (size_t i = numArgsOri + 1; i < rTypes.size(); i++) {
      auto t = rTypes[i];
      wcet::InitOp ini = rewriter.create<wcet::InitOp>(rewriter.getUnknownLoc(), t, rNames[i]);
      mainTask.getBody().addArgument(t, mainTask.getBody().getLoc());
      mainTask.getArgsMutable().append(ini.getResult());
      auto arg = mainTask.getBody().getArgument(i - 1);
      mainTask->walk([&](DelayOp delay) {
        auto attr = delay->getAttr(rewriter.getStringAttr("wcet.pragma"));
        if (attr) {
          if (auto opName = dyn_cast_or_null<mlir::StringAttr>(attr)) {
            if (opName == rNames[i]) {
              rewriter.replaceAllOpUsesWith(delay, arg);
              rewriter.eraseOp(delay);
              return;
            }
          }
        }
      });
    }

    wcet::DummyOp dummyEntry = rewriter.create<wcet::DummyOp>(
        rewriter.getUnknownLoc(), mainTask->getOperands().getType(), mainTask->getOperands());
    dummyEntry->setAttr("wcet.entry", rewriter.getUnitAttr());
    for (size_t i = 0; i < dummyEntry.getInputs().size(); i++) {
      rewriter.replaceAllUsesWith(dummyEntry.getInputs()[i], dummyEntry->getResult(i));
    }

    unsigned int inst = instrs.front();
    mainTask->walk([&](hw::ConstantOp cons) {
      auto attr = cons->getAttr(rewriter.getStringAttr("wcet.pragma"));
      if (attr) {
        if (auto opName = dyn_cast_or_null<mlir::StringAttr>(attr)) {
          if (opName == "instruction") {
            cons.setValueAttr(rewriter.getI32IntegerAttr(inst));
            return;
          }
        }
      }
    });
    spechls::UnpackOp unpack = nullptr;
    for (size_t i = 1; i < instrs.size(); i++) {
      inst = instrs[i];
      rewriter.setInsertionPointAfter(mainTask);
      unpack = rewriter.create<UnpackOp>(rewriter.getUnknownLoc(), mainTask.getResult());
      auto newTask = dyn_cast_or_null<spechls::TaskOp>(rewriter.clone(*mainTask));
      llvm::SmallVector<Value> operands = llvm::SmallVector<Value>();
      for (size_t j = 1; j < unpack.getResults().size(); j++) {
        operands.push_back(unpack.getResult(j));
      }
      newTask->setOperands(operands);
      mainTask = newTask;
      mainTask->walk([&](hw::ConstantOp cons) {
        auto attr = cons->getAttr(rewriter.getStringAttr("wcet.pragma"));
        if (attr) {
          if (auto opName = dyn_cast_or_null<mlir::StringAttr>(attr)) {
            if (opName == "instruction") {
              cons.setValueAttr(rewriter.getI32IntegerAttr(inst));
              return;
            }
          }
        }
      });
    }
    lastUnpack.setOperand(mainTask.getResult());

    return success();
  }
};

struct InsertDummyPattern : OpRewritePattern<spechls::KernelOp> {
  using OpRewritePattern<spechls::KernelOp>::OpRewritePattern;

  // Constructor to save pass arguments
  InsertDummyPattern(MLIRContext *ctx, const llvm::ArrayRef<unsigned int> intrs)
      : OpRewritePattern<spechls::KernelOp>(ctx) {}

  LogicalResult matchAndRewrite(spechls::KernelOp top, PatternRewriter &rewriter) const override {
    SmallVector<spechls::UnpackOp> unpacks = SmallVector<spechls::UnpackOp>();
    top->walk([&](spechls::UnpackOp unp) {
      if (unp.getInput().getDefiningOp()->getName().getStringRef() == spechls::PackOp::getOperationName())
        unpacks.push_back(unp);
    });

    if (unpacks.empty())
      return failure();

    for (auto un : unpacks) {
      rewriter.setInsertionPointAfter(un);
      spechls::PackOp pack = dyn_cast_or_null<spechls::PackOp>(un.getInput().getDefiningOp());
      rewriter.replaceOpWithNewOp<wcet::DummyOp>(un, pack.getInputs().getType(), pack.getInputs());
      rewriter.eraseOp(pack);
    }

    return success();
  }
};

struct UnrollInstrPass : public impl::UnrollInstrPassBase<UnrollInstrPass> {

  using UnrollInstrPassBase::UnrollInstrPassBase;

public:
  void runOnOperation() override {
    auto top = getOperation();
    auto *ctx = &getContext();

    RewritePatternSet unrollPattern(ctx);
    unrollPattern.add<UnrollInstrPattern>(ctx, *instrs);
    if (failed(applyPatternsGreedily(getOperation()->getParentOp(), std::move(unrollPattern)))) {
      llvm::errs() << "failed\n";
      signalPassFailure();
    }

    OpPassManager dynamicPM(spechls::KernelOp::getOperationName());
    dynamicPM.addPass(wcet::createInlineTasksPass());
    if (failed(runPipeline(dynamicPM, top))) {
      return signalPassFailure();
    }

    RewritePatternSet dummyPatterns(ctx);
    dummyPatterns.add<InsertDummyPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation()->getParentOp(), std::move(dummyPatterns)))) {
      llvm::errs() << "failed\n";
      signalPassFailure();
    }

    //  dynamicPM.addPass(mlir::createCanonicalizerPass());
    //  if (failed(runPipeline(dynamicPM, getOperation()))) {
    //    return signalPassFailure();
    //  }
  }
};

} // namespace wcet
