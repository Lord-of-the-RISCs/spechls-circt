//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "Dialect/SpecHLS/IR/SpecHLS.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<spechls::SpecHLSDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "SpecHLS optimizer driver", registry));
}
