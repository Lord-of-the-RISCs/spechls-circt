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
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>
#include <circt/Transforms/Passes.h>

#include "Conversion/Schedule/Passes.h"
#include "Conversion/SpecHLS/Passes.h"
#include "Dialect/Schedule/IR/Schedule.h"
#include "Dialect/Schedule/Transforms/Passes.h"
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"

#include <circt/Dialect/Seq/SeqOps.h>
#include <circt/Dialect/Seq/SeqTypes.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<spechls::SpecHLSDialect, schedule::ScheduleDialect, circt::comb::CombDialect, circt::seq::SeqDialect, circt::hw::HWDialect,
                  circt::ssp::SSPDialect, mlir::arith::ArithDialect>();


  //mlir::registerTransformsPasses();
  circt::registerCIRCTTransformsPasses();
  spechls::registerSpecHLSPasses();
  spechls::registerSpecHLSToHWPass();
  spechls::registerSpecHLSToSeqPass();

  schedule::registerSchedulePasses();
  schedule::registerScheduleToSSPPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "SpecHLS optimizer driver", registry));
}
