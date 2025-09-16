//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_TOPOLOGICAL_SORT_H
#define SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_TOPOLOGICAL_SORT_H

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace spechls {

inline bool topologicalSortCriterion(mlir::Value, mlir::Operation *op) {
  return mlir::isa<spechls::MuOp>(op) || mlir::isa<spechls::DelayOp>(op);
}

} // namespace spechls

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_TOPOLOGICAL_SORT_H
