//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#ifndef SPECHLS_TRANSITIVE_CLOSURE_H
#define SPECHLS_TRANSITIVE_CLOSURE_H

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinDialect.h>

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

namespace spechls {

void computeBackwardCone(mlir::Value value, llvm::SmallVector<mlir::Operation *> &cone,
                         llvm::SmallVector<mlir::Operation *> &delays);
KernelOp outlineBackwardCone(mlir::Value value, mlir::RewriterBase &rewriter);

} // namespace spechls

#endif // SPECHLS_TRANSITIVE_CLOSURE_H