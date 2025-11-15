#pragma once

#include <tuple>
#include "mlir/IR/Value.h"
#include "circt/Dialect/HW/HWOps.h"

using mlir::Value;

std::tuple<mlir::Value, mlir::Value, mlir::Value> getCREArgs(circt::hw::HWModuleOp mod);
