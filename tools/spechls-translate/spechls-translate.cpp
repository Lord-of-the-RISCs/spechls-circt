//===- spechls-translate.cpp ----------------------------------*- C++ -*-===//
//
// This file is part of the SpecHLS project.
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/raw_ostream.h>

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Operation.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include <mlir/Transforms/Passes.h>

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Target/Cpp/Export.h"
#include "Target/Uclid/Export.h"   // <-- new: Uclid emitter registration

using namespace mlir;

int main(int argc, char **argv) {
  // ---- C++ emitter flags (unchanged) -------------------------------------
  static llvm::cl::opt<bool> declareStructTypes(
      "declare-struct-types",
      llvm::cl::desc("Declare structure types at the top of the emitted C++ file"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> declareFunctions(
      "declare-functions",
      llvm::cl::desc("Declare called functions at the top of the emitted C++ file"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> lowerArraysAsValues(
      "arrays-as-values", llvm::cl::desc("Lower arrays with value semantics"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> generateCpi(
      "generate-cpi", llvm::cl::desc("Generate CPI extraction code"),
      llvm::cl::init(false));
  static llvm::cl::opt<bool> vitisHlsCompatibility(
      "vitis-hls-compat",
      llvm::cl::desc(
          "Emit code that circumvents bugs and limitations in Vitis HLS"),
      llvm::cl::init(false));

  // Registration: SpecHLS → C++ (existing)
  mlir::TranslateFromMLIRRegistration regCpp(
      "spechls-to-cpp", "Translate SpecHLS to C++",
      [](mlir::Operation *op, llvm::raw_ostream &os) {
        return spechls::translateToCpp(op, os,
                                       {declareStructTypes, declareFunctions,
                                        lowerArraysAsValues, generateCpi,
                                        vitisHlsCompatibility});
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<spechls::SpecHLSDialect, circt::comb::CombDialect,
                        circt::hw::HWDialect, mlir::arith::ArithDialect>();
      });

  // ---- UCLID5 emitter registration (new) ---------------------------------
  // Minimal, option-less registration. If your Export.h exposes options,
  // wire CLI flags here similarly to the C++ emitter.
  mlir::TranslateFromMLIRRegistration regUclid(
      "spechls-to-uclid", "Translate SpecHLS to Uclid5",
      [](mlir::Operation *op, llvm::raw_ostream &os) {
        // Minimal call; adjust if your exporter exposes options.
        return spechls::translateToUclid(op, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<spechls::SpecHLSDialect, circt::comb::CombDialect,
                        circt::hw::HWDialect, mlir::arith::ArithDialect>();
      });

  return failed(mlir::mlirTranslateMain(argc, argv,
                                        "SpecHLS translation driver"));
}
