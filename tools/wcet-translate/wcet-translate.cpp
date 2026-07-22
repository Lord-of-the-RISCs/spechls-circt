//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Operation.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include <mlir/Transforms/Passes.h>

#include "Target/Cpp/Export.h"
#include "Target/ExportWcetCpp/ExportWcetCpp.h"

int main(int argc, char **argv) {

  spechls::registerExportWcetCpp();

  return failed(mlir::mlirTranslateMain(argc, argv, "WCET translation driver"));
}
