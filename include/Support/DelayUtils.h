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
