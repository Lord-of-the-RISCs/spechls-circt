//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/IR/Schedule.h"
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/Wcet/IR/Wcet.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  circt::registerAllDialects(registry);
  registry.insert<spechls::SpecHLSDialect, wcet::WcetDialect, schedule::ScheduleDialect>();
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
