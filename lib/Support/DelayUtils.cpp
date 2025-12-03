#include "Support/DelayUtils.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "llvm/ADT/TypeSwitch.h"

template <typename T, typename F>
void walkOnDelay(T &&block, F &&fun) {
  block.walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *, void>(op)
        .Case<spechls::DelayOp, spechls::CancellableDelayOp, spechls::RollbackableDelayOp>(fun);
  });
}
