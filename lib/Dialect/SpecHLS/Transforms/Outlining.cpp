//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/Transforms/Outlining.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/SpecHLS/Transforms/TopologicalSort.h"
#include "circt/Support/LLVM.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/LogicalResult.h"

#include <circt/Dialect/HW/HWOps.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

spechls::TaskOp spechls::outlineControl(RewriterBase &rewriter, Location loc, std::string name,
                                        DenseSet<Operation *> &ops, Value output) {
  auto ip = rewriter.saveInsertionPoint();
  SmallVector<Value> inputs;
  DenseMap<Operation *, Operation *> cloneMap;

  SmallVector<Operation *> toAdd, interfaceCast;

  for (auto &&op : ops) {
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      auto &&operand = op->getOperand(i);
      auto *operation = operand.getDefiningOp();
      if (!ops.contains(operation)) {
        auto operandType = llvm::dyn_cast<IntegerType>(operand.getType());
        auto signlessType = rewriter.getIntegerType(operandType.getWidth());
        auto outCast = rewriter.create<circt::hw::BitcastOp>(operand.getLoc(), signlessType, operand);
        outCast->setAttr("out", rewriter.getUnitAttr());
        auto inCast = rewriter.create<circt::hw::BitcastOp>(operand.getLoc(), operandType, outCast.getResult());
        inCast->setAttr("out", rewriter.getUnitAttr());
        op->setOperand(i, inCast.getResult());
        toAdd.push_back(inCast.getOperation());
        inputs.push_back(inCast.getOperand());
      }
    }
  }
  ops.insert_range(toAdd);

  //  Type returnType = output.getType();
  StructType returnType = spechls::StructType::get(
      rewriter.getContext(), name + "_commit", llvm::SmallVector<std::string>{"enable", "commit_val_0"},
      llvm::SmallVector<mlir::Type>{mlir::IntegerType::get(rewriter.getContext(), 1), output.getType()});
  auto task = rewriter.create<spechls::TaskOp>(loc, returnType, name, inputs);
  Block &body = task.getBody().front();
  for (auto &&op : ops) {
    rewriter.setInsertionPointToEnd(&body);
    auto *cloned = rewriter.clone(*op);
    cloneMap.try_emplace(op, cloned);
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    for (auto &&op : ops) {
      for (size_t j = 0; j < op->getNumOperands(); ++j) {
        if (op->getOperand(j) == inputs[i]) {
          cloneMap[op]->setOperand(j, body.getArgument(i));
        }
      }
    }
  }

  for (auto &&op : ops) {
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      auto operand = op->getOperand(i);
      auto *operandOperation = operand.getDefiningOp();
      if (ops.contains(operandOperation)) {
        for (size_t j = 0; j < operandOperation->getNumResults(); ++j)
          if (operandOperation->getResult(j) == operand) {
            cloneMap[op]->setOperand(i, cloneMap[operandOperation]->getResult(j));
            break;
          }
      }
    }
  }
  rewriter.setInsertionPointAfter(task);
  auto field = rewriter.create<spechls::FieldOp>(output.getLoc(), "commit_val_0", task.getResult());
  output.replaceAllUsesWith(field.getResult());

  rewriter.setInsertionPointToEnd(&body);
  auto enable = rewriter.create<circt::hw::ConstantOp>(loc, rewriter.getI1Type(), 1);

  Operation *outputParent = output.getDefiningOp();
  for (size_t i = 0; i < outputParent->getNumResults(); ++i) {
    if (outputParent->getResult(i) == output) {
      rewriter.create<spechls::CommitOp>(loc,
                                         llvm::SmallVector<mlir::Value>{enable, cloneMap[outputParent]->getResult(i)});
    }
  }
  rewriter.restoreInsertionPoint(ip);
  return task;
}

spechls::OptimizedFuncOp spechls::outlineOptFunc(RewriterBase &rewriter, Location loc, std::string name,
                                                 llvm::DenseSet<mlir::Operation *> &ops, Value output) {
  auto ip = rewriter.saveInsertionPoint();
  SmallVector<Value> inputs;
  llvm::SmallVector<mlir::Operation *> casts;
  for (auto &&op : ops) {
    for (auto &&operand : op->getOperands()) {
      if (!ops.contains(operand.getDefiningOp())) {
        bool needInsert = true;
        for (auto &in : inputs) {
          if (in == operand) {
            needInsert = false;
            break;
          }
        }
        if (needInsert) {
          if (auto intType = llvm::dyn_cast<mlir::IntegerType>(operand.getType())) {
            if (!intType.isSignless()) {
              if (operand.getDefiningOp()) {
                rewriter.setInsertionPointAfter(operand.getDefiningOp());
              } else {
                rewriter.setInsertionPointToStart(operand.getParentBlock());
              }
              auto signlessType = rewriter.getIntegerType(intType.getWidth());
              auto cast1 = circt::hw::BitcastOp::create(rewriter, rewriter.getUnknownLoc(), signlessType, operand);
              auto cast2 = circt::hw::BitcastOp::create(rewriter, rewriter.getUnknownLoc(), intType, cast1.getResult());
              operand.replaceAllUsesExcept(cast2.getResult(), cast1);
              inputs.push_back(cast1.getResult());
              casts.push_back(cast2);
              continue;
            }
          }
          inputs.push_back(operand);
        }
      }
    }
  }
  for (auto *cast : casts) {
    ops.insert(cast);
  }

  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(output.getType())) {
    if (!intType.isSignless()) {
      rewriter.setInsertionPointAfter(output.getDefiningOp());
      auto signlessType = rewriter.getIntegerType(intType.getWidth());
      auto cast1 = circt::hw::BitcastOp::create(rewriter, rewriter.getUnknownLoc(), signlessType, output);
      auto cast2 = circt::hw::BitcastOp::create(rewriter, rewriter.getUnknownLoc(), intType, cast1.getResult());
      output.replaceAllUsesExcept(cast2.getResult(), cast1);
      ops.insert(cast1);
      output = cast2.getResult();
    }
  }

  auto optFun = spechls::OptimizedFuncOp::create(rewriter, loc, output.getType(), inputs);
  output.replaceAllUsesWith(optFun.getResult());
  auto fillBody = [&](mlir::Block *block) {
    rewriter.setInsertionPointToStart(block);
    IRMapping mapping;
    for (auto [from, to] : llvm::zip(inputs, block->getArguments())) {
      mapping.map(from, to);
    }
    llvm::SmallVector<mlir::Operation *> cloned;
    for (auto *op : ops) {
      cloned.push_back(rewriter.clone(*op, mapping));
    }
    for (auto *op : cloned) {
      for (size_t i = 0; i < op->getNumOperands(); ++i) {
        auto operand = op->getOperand(i);
        if (mapping.contains(operand)) {
          op->setOperand(i, mapping.lookup(operand));
        }
      }
    }
    rewriter.create<spechls::YieldOp>(loc, mapping.lookup(output));
  };
  fillBody(optFun.getBodyBlock());
  fillBody(optFun.getOptBodyBlock());

  rewriter.setInsertionPointAfter(optFun);

  rewriter.restoreInsertionPoint(ip);
  optFun->setAttr("sym_name", rewriter.getStringAttr(name));

  mlir::sortTopologically(optFun.getBodyBlock(), spechls::topologicalSortCriterion);
  mlir::sortTopologically(optFun.getOptBodyBlock(), spechls::topologicalSortCriterion);

  return optFun;
}

spechls::TaskOp spechls::outlineTask(RewriterBase &rewriter, Location loc, std::string name,
                                     const SmallPtrSetImpl<Operation *> &ops) {
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;

  for (auto &&op : ops) {
    for (auto &&operand : op->getOperands()) {
      if (!ops.contains(operand.getDefiningOp())) {
        bool needInsert = true;
        for (auto &in : inputs) {
          if (in == operand) {
            needInsert = false;
            break;
          }
        }
        if (needInsert)
          inputs.push_back(operand);
      }
    }
    for (auto &&result : op->getResults()) {
      if (llvm::any_of(result.getUsers(), [&](Operation *other) { return !ops.contains(other); })) {
        outputs.push_back(result);
      }
    }
  }

  // Compute the return type. If there are multiple exiting use-def edges, then we return a struct type.
  Type returnType{};
  SmallVector<std::string> fieldNames;
  SmallVector<Type> fieldTypes;
  fieldNames.reserve(outputs.size() + 1);
  fieldTypes.reserve(outputs.size() + 1);
  fieldNames.push_back("enable");
  fieldTypes.push_back(mlir::IntegerType::get(outputs.front().getContext(), 1));
  for (size_t i = 0; i < outputs.size(); ++i) {
    fieldNames.push_back("commit_val_" + std::to_string(i));
    fieldTypes.push_back(outputs[i].getType());
  }
  returnType = rewriter.getType<spechls::StructType>((name + std::string{"_result"}), fieldNames, fieldTypes);

  // Move operations into the task's body.
  auto task = rewriter.create<spechls::TaskOp>(loc, returnType, name, inputs);
  Block &body = task.getBody().front();
  for (auto &&op : ops) {
    rewriter.moveOpBefore(op, &body, body.end());
  }

  // Update inputs.
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (auto &&op : ops) {
      for (size_t j = 0; j < op->getNumOperands(); ++j) {
        if (op->getOperand(j) == inputs[i]) {
          op->setOperand(j, body.getArgument(i));
        }
      }
    }
  }

  rewriter.setInsertionPointAfter(task);
  Value result = task.getResult();
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto field = rewriter.create<spechls::FieldOp>(loc, fieldNames[i + 1], result);
    rewriter.replaceUsesWithIf(outputs[i], field, [&](auto &&opOperand) {
      if (ops.contains(opOperand.getOwner()))
        return false;
      return true;
    });
  }

  // Create the commit terminator.
  auto ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(&body);
  llvm::SmallVector<mlir::Value> returnValues;
  auto enable = rewriter.create<circt::hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
  returnValues.push_back(enable);
  for (auto &out : outputs) {
    returnValues.push_back(out);
  }
  rewriter.create<spechls::CommitOp>(loc, returnValues);
  rewriter.restoreInsertionPoint(ip);

  return task;
}
