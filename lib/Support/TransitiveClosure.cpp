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

namespace spechls {

void computeBackwardCone(mlir::Value &value, llvm::DenseSet<mlir::Operation *> &cone,
                         llvm::DenseSet<mlir::Operation *> &delays) {
  llvm::SmallVector<mlir::Operation *> workingList;
  if (value.getDefiningOp() != nullptr)
    workingList.push_back(value.getDefiningOp());
  while (!workingList.empty()) {
    auto *current = workingList.pop_back_val();
    llvm::TypeSwitch<mlir::Operation *, void>(current)
        .Case<spechls::MuOp>([](mlir::Operation *) { /* nothing*/ })
        .Case<spechls::DelayOp, spechls::RollbackableDelayOp, spechls::CancellableDelayOp>(
            [&cone, &delays, &workingList, current](mlir::Operation *) {
              delays.insert(current);
              cone.insert(current);
              for (auto op : current->getOperands()) {
                workingList.push_back(op.getDefiningOp());
              }
            })
        .Default([&workingList, current](mlir::Operation *) {
          for (auto op : current->getOperands()) {
            workingList.push_back(op.getDefiningOp());
          }
        });
  }
}

} // namespace spechls