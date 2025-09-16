//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_OUTLINING_H
#define SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_OUTLINING_H

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

namespace spechls {

TaskOp outlineTask(mlir::RewriterBase &rewriter, mlir::Location loc, mlir::StringRef name,
                   const mlir::SmallPtrSetImpl<mlir::Operation *> &ops);

} // namespace spechls

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_TRANSFORMS_OUTLINING_H
