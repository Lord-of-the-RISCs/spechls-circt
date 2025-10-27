//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/Transforms/Outlining.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"

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

  Type returnType = output.getType();
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

  output.replaceAllUsesWith(task.getResult());

  rewriter.setInsertionPointToEnd(&body);
  auto enable = rewriter.create<circt::hw::ConstantOp>(loc, rewriter.getI1Type(), 1);

  Operation *outputParent = output.getDefiningOp();
  for (size_t i = 0; i < outputParent->getNumResults(); ++i) {
    if (outputParent->getResult(i) == output) {
      rewriter.create<spechls::CommitOp>(loc, enable, cloneMap[outputParent]->getResult(i));
    }
  }
  rewriter.restoreInsertionPoint(ip);
  return task;
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
  if (outputs.size() == 1) {
    returnType = outputs[0].getType();
  } else if (outputs.size() > 1) {
    fieldNames.reserve(outputs.size());
    fieldTypes.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      fieldNames.push_back("commit_" + std::to_string(i));
      fieldTypes.push_back(outputs[i].getType());
    }
    returnType = rewriter.getType<spechls::StructType>((name + std::string{"_result"}), fieldNames, fieldTypes);
  }

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

  // Replace outgoing values with task results.
  if (outputs.size() == 1) {
    outputs[0].replaceAllUsesWith(task.getResult());
  } else {
    Value result = task.getResult();
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].replaceAllUsesWith(rewriter.create<spechls::FieldOp>(loc, fieldNames[i], result));
    }
  }

  // Create the commit terminator.
  auto ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(&body);
  Value returnValue{};
  if (outputs.size() == 1) {
    returnValue = outputs[0];
  } else {
    returnValue = rewriter.create<spechls::PackOp>(loc, returnType, outputs);
  }
  auto enable = rewriter.create<circt::hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
  rewriter.create<spechls::CommitOp>(loc, enable, returnValue);
  rewriter.restoreInsertionPoint(ip);

  return task;
}
