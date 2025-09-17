//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Operation.h>

namespace spechls {

struct TranslationToCppOptions {
  bool declareStructTypes;
  bool declareFunctions;
  bool lowerArraysAsValues;
  bool generateCpi;
  bool vitisHlsCompatibility;
};

llvm::LogicalResult translateToCpp(mlir::Operation *op, llvm::raw_ostream &os, TranslationToCppOptions options);

} // namespace spechls
