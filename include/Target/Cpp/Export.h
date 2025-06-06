//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

namespace spechls {

llvm::LogicalResult translateToCpp(mlir::Operation *op, llvm::raw_ostream &os, bool declareStructTypes,
                                   bool declareFunctions);

} // namespace spechls
