//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "Dialect/SpecHLS/Transforms/TopologicalSort.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_LINEARIZETASKPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

struct TaskInlinePred : OpRewritePattern<spechls::TaskOp> {
  using OpRewritePattern<spechls::TaskOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spechls::TaskOp task, PatternRewriter &rewriter) const override {
    for (unsigned index = 0; index < task.getArgs().size(); ++index) {
      auto operand = task.getArgs()[index];
      if (auto *pred = operand.getDefiningOp()) {
        if (!llvm::isa<spechls::TaskOp>(pred)) {
          auto ip = rewriter.saveInsertionPoint();
          rewriter.setInsertionPointToStart(task.getBodyBlock());
          auto arg = task.getBodyBlock()->getArgument(index);
          IRMapping mapper;
          auto *newPred = rewriter.clone(*pred);
          rewriter.replaceAllUsesWith(arg, newPred->getResult(0));
          task.getArgsMutable().erase(index);
          task.getBodyBlock()->eraseArgument(index);
          llvm::SmallVector<mlir::Type> newArgsType;
          llvm::SmallVector<mlir::Location> newArgsLoc;
          llvm::SmallVector<mlir::Value> newArgs;
          unsigned numArgs = newPred->getNumOperands();
          for (auto arg : newPred->getOperands()) {
            newArgsType.push_back(arg.getType());
            newArgsLoc.push_back(mlir::UnknownLoc::get(rewriter.getContext()));
            newArgs.push_back(arg);
          }
          auto newBlockArgs = task.getBodyBlock()->addArguments(newArgsType, newArgsLoc);
          task.getArgsMutable().append(newArgs);
          auto *it = newBlockArgs.begin();
          for (unsigned i = 0; i < numArgs; ++i, ++it) {
            newPred->setOperand(i, *it);
          }

          rewriter.restoreInsertionPoint(ip);
          return llvm::success();
        }
      }
    }
    return llvm::failure();
  }
};

struct AddPredTaskPattern : OpRewritePattern<spechls::TaskOp> {

  llvm::DenseMap<spechls::TaskOp, unsigned> taskOrder;
  llvm::SmallVector<spechls::TaskOp> taskOrderReverse;

  AddPredTaskPattern(llvm::DenseMap<spechls::TaskOp, unsigned> taskOrder,
                     llvm::SmallVector<spechls::TaskOp> taskOrderReverse, mlir::MLIRContext *context)
      : OpRewritePattern(context), taskOrder(taskOrder), taskOrderReverse(taskOrderReverse) {}

  LogicalResult matchAndRewrite(spechls::TaskOp task, PatternRewriter &rewriter) const override {
    if (task.getResult().getType().getFieldNames().back() == "pred")
      return mlir::failure();
    unsigned index = taskOrder.at(task);
    if (index == 0)
      return mlir::failure();

    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(task.getBodyBlock());

    auto predTask = taskOrderReverse[index - 1];
    auto predType = predTask.getResult().getType();
    task.getArgsMutable().append(predTask.getResult());
    auto predVal = task.getBodyBlock()->addArgument(predType, rewriter.getUnknownLoc());

    llvm::dyn_cast<spechls::CommitOp>(task.getBodyBlock()->getTerminator()).getValueMutable().append(predVal);

    auto oldType = task.getResult().getType();

    llvm::SmallVector<std::string> newTypeFieldNames;
    newTypeFieldNames.append(oldType.getFieldNames().begin(), oldType.getFieldNames().end());
    newTypeFieldNames.push_back("pred");
    llvm::SmallVector<mlir::Type> newTypeFieldTypes;
    newTypeFieldTypes.append(oldType.getFieldTypes().begin(), oldType.getFieldTypes().end());
    newTypeFieldTypes.push_back(predType);
    task.getResult().setType(rewriter.getType<spechls::StructType>(std::string(oldType.getName()) + "_with_pred",
                                                                   newTypeFieldNames, newTypeFieldTypes));
    llvm::SmallVector<mlir::Value> fields;
    llvm::DenseSet<mlir::Operation *> newUses;

    rewriter.setInsertionPointAfter(task);

    for (auto fieldName : oldType.getFieldNames()) {
      auto newField = rewriter.create<spechls::FieldOp>(rewriter.getUnknownLoc(), fieldName, task.getResult());
      fields.push_back(newField.getResult());
      newUses.insert(newField);
    }

    auto pack = rewriter.create<spechls::PackOp>(rewriter.getUnknownLoc(), oldType, fields);

    task->replaceUsesWithIf(pack->getResults(),
                            [&](mlir::OpOperand &operand) { return !newUses.contains(operand.getOwner()); });

    rewriter.restoreInsertionPoint(ip);

    return mlir::success();
  }
};

struct UseOnlyDirectPredecessorPattern : OpRewritePattern<spechls::TaskOp> {

  llvm::DenseMap<spechls::TaskOp, unsigned> taskOrder;
  llvm::SmallVector<spechls::TaskOp> taskOrderReverse;

  UseOnlyDirectPredecessorPattern(llvm::DenseMap<spechls::TaskOp, unsigned> taskOrder,
                                  llvm::SmallVector<spechls::TaskOp> taskOrderReverse, mlir::MLIRContext *context)
      : OpRewritePattern(context), taskOrder(taskOrder), taskOrderReverse(taskOrderReverse) {}

  LogicalResult matchAndRewrite(spechls::TaskOp task, PatternRewriter &rewriter) const override {
    unsigned index = taskOrder.at(task);
    if (index == 0)
      return mlir::failure();
    auto predTask = taskOrderReverse[index - 1];
    for (unsigned idx = 0; idx < task.getArgs().size(); ++idx) {
      auto arg = task.getArgs()[idx];
      if (auto *pred = arg.getDefiningOp()) {
        if (pred != predTask) {
          if (task.getBodyBlock()->getArgument(idx).getNumUses() != 0) {
            unsigned predIndex = taskOrder.at(llvm::dyn_cast<spechls::TaskOp>(pred));
            unsigned numPred = index - predIndex - 1;
            task.getArgsMutable().append(predTask.getResult());
            auto ip = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointToStart(task.getBodyBlock());
            mlir::Value currentValue =
                task.getBodyBlock()->addArgument(predTask.getResult().getType(), rewriter.getUnknownLoc());
            for (unsigned i = 0; i < numPred; ++i) {
              auto field = rewriter.create<spechls::FieldOp>(rewriter.getUnknownLoc(), "pred", currentValue);
              currentValue = field.getResult();
            }
            rewriter.replaceAllUsesWith(task.getBodyBlock()->getArgument(idx), currentValue);
            rewriter.restoreInsertionPoint(ip);
            return mlir::success();
          }
        }
      }
    }
    return mlir::failure();
  }
};

struct LinearizeTaskPass : public spechls::impl::LinearizeTaskPassBase<LinearizeTaskPass> {
  using LinearizeTaskPassBase::LinearizeTaskPassBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto kernel = getOperation();

    mlir::FrozenRewritePatternSet patterns;
    RewritePatternSet patternList{ctx};
    patternList.add<TaskInlinePred>(ctx);
    patterns = std::move(patternList);
    if (failed(applyPatternsGreedily(kernel, patterns)))
      return signalPassFailure();

    mlir::sortTopologically(&kernel.getBody().front(), spechls::topologicalSortCriterion);
    llvm::DenseMap<spechls::TaskOp, unsigned> taskOrder;
    llvm::SmallVector<spechls::TaskOp> taskOrderReverse;
    unsigned currentTask = 0;
    kernel.walk([&](spechls::TaskOp task) {
      taskOrder.try_emplace(task, currentTask++);
      taskOrderReverse.push_back(task);
    });

    // outline everything that isn't in a task (except exit) in a last task
    llvm::DenseSet<mlir::Operation *> toOutline;
    for (auto &op : kernel.getBody().front().getOperations()) {
      if (!llvm::isa<spechls::TaskOp>(op) && !llvm::isa<spechls::ExitOp>(op)) {
        toOutline.insert(&op);
      }
    }

    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&kernel.getBody().front());
    llvm::SmallVector<mlir::Value> outputs;

    unsigned current = 0;
    llvm::SmallVector<mlir::Type> returnTypeFieldTypes{builder.getI1Type()};
    llvm::SmallVector<std::string> returnTypeFieldNames{"enable"};

    for (auto *op : toOutline) {
      if (op->getNumResults() != 0)
        for (auto &use : op->getResult(0).getUses()) {
          if (!toOutline.contains(use.getOwner())) {
            outputs.push_back(op->getResult(0));
            returnTypeFieldTypes.push_back(use.get().getType());
            returnTypeFieldNames.push_back("commit_val_" + std::to_string(current++));
            break;
          }
        }
    }

    auto lastTask = builder.create<spechls::TaskOp>(
        builder.getUnknownLoc(),
        builder.getType<spechls::StructType>("exit_task_type", returnTypeFieldNames, returnTypeFieldTypes), "exit_task",
        llvm::SmallVector<mlir::Value>());

    builder.setInsertionPointToStart(lastTask.getBodyBlock());
    mlir::IRMapping mapper;

    // Duplicate operations
    for (auto *op : toOutline) {
      builder.clone(*op, mapper);
    }

    // Rewire inputs
    for (auto *op : toOutline) {
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        auto operand = op->getOperand(i);
        if (auto *pred = operand.getDefiningOp()) {
          if (toOutline.contains(pred)) {
            mapper.lookup(op)->setOperand(i, mapper.lookup(pred)->getResult(0));
          } else {
            lastTask.getArgsMutable().append(operand);
            auto arg = lastTask.getBodyBlock()->addArgument(operand.getType(), builder.getUnknownLoc());
            mapper.lookup(op)->setOperand(i, arg);
          }
        } else {
          lastTask.getArgsMutable().append(operand);
          auto arg = lastTask.getBodyBlock()->addArgument(operand.getType(), builder.getUnknownLoc());
          mapper.lookup(op)->setOperand(i, arg);
        }
      }
    }

    llvm::SmallVector<mlir::Value> commitArgs{
        builder.create<circt::hw::ConstantOp>(builder.getUnknownLoc(), builder.getI1Type(), true)};
    builder.setInsertionPointAfter(lastTask);
    for (unsigned i = 0; i < outputs.size(); ++i) {
      auto out = outputs[i];
      commitArgs.push_back(mapper.lookup(out.getDefiningOp())->getResult(0));
      mlir::Value field = builder.create<spechls::FieldOp>(builder.getUnknownLoc(), "commit_val_" + std::to_string(i),
                                                           lastTask.getResult());
      out.replaceAllUsesWith(field);
    }
    builder.setInsertionPointToEnd(lastTask.getBodyBlock());
    builder.create<spechls::CommitOp>(builder.getUnknownLoc(), commitArgs);

    taskOrder.try_emplace(lastTask, currentTask++);
    taskOrderReverse.push_back(lastTask);

    mlir::FrozenRewritePatternSet patterns2;
    RewritePatternSet patternList2{ctx};
    patternList2.add<AddPredTaskPattern>(taskOrder, taskOrderReverse, ctx);
    patterns2 = std::move(patternList2);

    walkAndApplyPatterns(kernel, patterns2);

    if (failed(applyPatternsGreedily(kernel, patterns)))
      return signalPassFailure();

    mlir::FrozenRewritePatternSet patterns3;
    RewritePatternSet patternList3{ctx};
    patternList3.add<UseOnlyDirectPredecessorPattern>(taskOrder, taskOrderReverse, ctx);
    patterns3 = std::move(patternList3);

    if (failed(applyPatternsGreedily(kernel, patterns3)))
      return signalPassFailure();
  }
};

} // namespace