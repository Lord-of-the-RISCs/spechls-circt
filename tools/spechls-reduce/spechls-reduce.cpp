//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/SSP/SSPDialect.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Tools/mlir-reduce/MlirReduceMain.h>
#include <mlir/Transforms/Passes.h>

#include "Dialect/Schedule/IR/Schedule.h"
#include "Dialect/SpecHLS/IR/SpecHLS.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<spechls::SpecHLSDialect, schedule::ScheduleDialect, circt::ssp::SSPDialect, circt::comb::CombDialect,
                  circt::hw::HWDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);

  mlir::registerTransformsPasses();

  return mlir::failed(mlir::mlirReduceMain(argc, argv, context));
}
