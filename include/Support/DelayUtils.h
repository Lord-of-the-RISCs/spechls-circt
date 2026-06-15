//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_DELAY_UTILS_H
#define SPECHLS_DELAY_UTILS_H

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Dialect.h>

template <typename T, typename F>
void walkOnDelay(T &&block, F &&fun) {
  block.walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *, void>(op)
        .Case<spechls::DelayOp, spechls::CancellableDelayOp, spechls::RollbackableDelayOp>(fun);
  });
}

template <typename T, typename F>
void walkOnDelay(T *block, F &&fun) {
  block->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *, void>(op)
        .Case<spechls::DelayOp, spechls::CancellableDelayOp, spechls::RollbackableDelayOp>(fun);
  });
}

#endif
