//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/SSP/SSPDialect.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include "Dialect/Schedule/IR/Schedule.h"
#include "Dialect/Schedule/Transforms/Passes.h"
#include "Conversion/Passes.h"
#include "Dialect/SpecHLS/IR/SpecHLS.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<spechls::SpecHLSDialect, schedule::ScheduleDialect, circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::ssp::SSPDialect>();

  mlir::registerCanonicalizerPass();
  schedule::registerSchedulePasses();
  spechls::registerscheduleToSSPPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "SpecHLS optimizer driver", registry));
}
