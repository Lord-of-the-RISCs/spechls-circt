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

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Target/Cpp/Export.h"

int main(int argc, char **argv) {
  static llvm::cl::opt<bool> declareStructTypes(
      "declare-struct-types", llvm::cl::desc("Declare structure types at the top of the emitted C++ file"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> declareFunctions(
      "declare-functions", llvm::cl::desc("Declare called functions at the top of the emitted C++ file"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> lowerArraysAsValues(
      "arrays-as-values", llvm::cl::desc("Lower arrays with value semantics"), llvm::cl::init(false));
  static llvm::cl::opt<bool> generateCpi("generate-cpi", llvm::cl::desc("Generate CPI extraction code"),
                                         llvm::cl::init(false));
  static llvm::cl::opt<bool> vitisHlsCompatibility(
      "vitis-hls-compat", llvm::cl::desc("Emit code that circumvents bugs and limitations in Vitis HLS"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> catapultCompatibility(
      "catapult-compat", llvm::cl::desc("Emit code that is compatible with catapult"), llvm::cl::init(false));

  mlir::TranslateFromMLIRRegistration registration(
      "spechls-to-cpp", "Translate SpecHLS to C++",
      [](mlir::Operation *op, llvm::raw_ostream &os) {
        return spechls::translateToCpp(
            op, os, {declareStructTypes, declareFunctions, lowerArraysAsValues, generateCpi, vitisHlsCompatibility});
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<spechls::SpecHLSDialect, circt::comb::CombDialect, circt::hw::HWDialect,
                        mlir::arith::ArithDialect>();
      });

  return failed(mlir::mlirTranslateMain(argc, argv, "SpecHLS translation driver"));
}
