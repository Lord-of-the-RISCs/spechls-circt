//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include <mlir/Transforms/Passes.h>

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Target/Cpp/Export.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  mlir::TranslateFromMLIRRegistration registration(
      "spechls-to-cpp", "Translate SpecHLS to C++",
      [](mlir::Operation *op, llvm::raw_ostream &os) { return spechls::translateToCpp(op, os); },
      [](mlir::DialectRegistry &registry) {
        registry.insert<spechls::SpecHLSDialect, circt::comb::CombDialect, circt::hw::HWDialect>();
      });

  return failed(mlir::mlirTranslateMain(argc, argv, "SpecHLS translation driver"));
}
