//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <llvm/ADT/TypeSwitch.h>
#include <queue>

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Support/TransitiveClosure.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"

namespace spechls {

void computeBackwardCone(mlir::Value value, llvm::SmallVector<mlir::Operation *> &cone,
                         llvm::SmallVector<mlir::Operation *> &delays) {
  llvm::SmallVector<mlir::Operation *> workingList;
  if (value.getDefiningOp() != nullptr)
    workingList.push_back(value.getDefiningOp());
  while (!workingList.empty()) {
    auto *current = workingList.pop_back_val();
    llvm::TypeSwitch<mlir::Operation *, void>(current)
        .Case<spechls::MuOp>([](mlir::Operation *) { /* nothing*/ })
        .Case<spechls::DelayOp, spechls::RollbackableDelayOp, spechls::CancellableDelayOp>(
            [&cone, &delays, &workingList, current](mlir::Operation *) {
              delays.push_back(current);
              cone.push_back(current);
              for (auto op : current->getOperands()) {
                if (auto *defOp = op.getDefiningOp())
                  workingList.push_back(defOp);
              }
            })
        .Default([&workingList, current](mlir::Operation *) {
          for (auto op : current->getOperands()) {
            if (auto *defOp = op.getDefiningOp())
              workingList.push_back(defOp);
          }
        });
  }
}

spechls::KernelOp outlineBackwardCone(mlir::Value value, mlir::RewriterBase &rewriter) {
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Type> inputTypes;
  llvm::DenseSet<mlir::Operation *> operations;
  mlir::IRMapping mapping;

  if (value.getDefiningOp() == nullptr)
    return nullptr;

  llvm::SmallVector<mlir::Operation *> workingList;
  workingList.push_back(value.getDefiningOp());

  while (!workingList.empty()) {
    auto *current = workingList.pop_back_val();
    if (auto mu = llvm::dyn_cast<spechls::MuOp>(current)) {
      if (llvm::find(inputs, mu.getInitValue()) == inputs.end()) {
        inputs.push_back(mu.getInitValue());
        inputTypes.push_back(mu.getType());
      }
      operations.insert(mu);
    } else {
      operations.insert(current);
      for (auto operand : current->getOperands())
        if (auto *defOp = operand.getDefiningOp()) {
          if (!operations.contains(defOp))
            workingList.push_back(defOp);
        } else {
          if (llvm::find(inputs, operand) == inputs.end()) {
            inputs.push_back(operand);
            inputTypes.push_back(operand.getType());
          }
        }
    }
  }
  rewriter.setInsertionPoint(value.getDefiningOp()->getParentOfType<mlir::ModuleOp>());
  auto module = mlir::ModuleOp::create(rewriter, mlir::UnknownLoc::get(rewriter.getContext()));
  rewriter.setInsertionPointToStart(module.getBody());
  auto funType = mlir::FunctionType::get(rewriter.getContext(), inputTypes, value.getType());

  auto kernel =
      rewriter.create<KernelOp>(mlir::UnknownLoc::get(rewriter.getContext()), rewriter.getStringAttr("my_kernel"),
                                mlir::TypeAttr::get(funType), nullptr, nullptr);

  auto *block = rewriter.createBlock(&kernel.getBody());
  for (auto &type : inputTypes) {
    block->addArgument(type, mlir::UnknownLoc::get(rewriter.getContext()));
  }
  rewriter.setInsertionPointToStart(&kernel.getBody().front());
  auto structType =
      spechls::StructType::get(rewriter.getContext(), "outline_fsm_control", {"commit_guard_0", "commit_val_0"},
                               {rewriter.getI1Type(), value.getType()});
  auto task = spechls::TaskOp::create(rewriter, value.getLoc(), structType, "outline_fsm_control",
                                      kernel.getBody().front().getArguments());

  rewriter.setInsertionPointToStart(task.getBodyBlock());
  for (auto [from, to] : llvm::zip(inputs, task.getBodyBlock()->getArguments())) {
    mapping.map(from, to);
  }
  for (auto *op : operations) {
    rewriter.clone(*op, mapping);
  }
  task.walk([&](mlir::Operation *op) {
    if (auto mu = llvm::dyn_cast<spechls::MuOp>(op)) {
      mu.getLoopValueMutable().assign(mu->getResult(0));
    }
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto operand = op->getOperand(i);
      if (mapping.contains(operand)) {
        op->setOperand(i, mapping.lookup(operand));
      }
    }
  });
  auto trueOp = circt::hw::ConstantOp::create(rewriter, mlir::UnknownLoc::get(rewriter.getContext()),
                                              mlir::IntegerAttr::get(rewriter.getI1Type(), 1));
  spechls::CommitOp::create(rewriter, mlir::UnknownLoc::get(rewriter.getContext()), {trueOp, mapping.lookup(value)});
  rewriter.setInsertionPointAfter(task);
  trueOp = circt::hw::ConstantOp::create(rewriter, mlir::UnknownLoc::get(rewriter.getContext()),
                                         mlir::IntegerAttr::get(rewriter.getI1Type(), 1));

  auto val = spechls::FieldOp::create(rewriter, mlir::UnknownLoc::get(rewriter.getContext()),
                                      mlir::StringAttr::get(rewriter.getContext(), "commit_val_0"), task.getResult());
  spechls::ExitOp::create(rewriter, mlir::UnknownLoc::get(rewriter.getContext()), trueOp, {val});

  if (failed(kernel.verify())) {
    llvm::errs() << "Kernel verification failed\n";
  }

  return kernel;
}

} // namespace spechls