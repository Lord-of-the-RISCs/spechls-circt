//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#ifndef SPECHLS_TRANSITIVE_CLOSURE_H
#define SPECHLS_TRANSITIVE_CLOSURE_H

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinDialect.h>

namespace spechls {

void computeBackwardCone(mlir::Value &value, llvm::DenseSet<mlir::Operation *> &cone,
                         llvm::DenseSet<mlir::Operation *> &delays);

} // namespace spechls

#endif // SPECHLS_TRANSITIVE_CLOSURE_H