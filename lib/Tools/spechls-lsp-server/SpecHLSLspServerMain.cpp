//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/ScheduleDialect.h"
#include "Dialect/Schedule/ScheduleOps.cpp.inc"
#include "Dialect/Schedule/ScheduleOps.h"
#include "Dialect/Schedule/ScheduleOpsDialect.cpp.inc"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.cpp.inc"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSOpsDialect.cpp.inc"
#include "InitAllDialects.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  circt::registerAllDialects(registry);
  SpecHLS::registerAllDialects(registry);

  return failed(mlir::MlirLspServerMain(argc, argv, registry));
}
