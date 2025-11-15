//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h" // IWYU pragma: keep
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HW/HWTypes.h>

#include <circt/Dialect/HW/PortImplementation.h>
#include <circt/Support/LLVM.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <mlir/Transforms/DialectConversion.h>
#include <string>

#include <Conversion/SpecHLS/SpecHLSToHW/common.h>

using namespace mlir;

/// Return (%clk,%rst,%ce) block arguments of an hw.module.
std::tuple<Value, Value, Value> getCREArgs(circt::hw::HWModuleOp mod) {
  Value clk, rst, ce;
  for (auto &pi : mod.getPortList()) {
    if (pi.isInput()) {
      Value arg = mod.getArgumentForInput(pi.argNum);
      if (!clk && isa<circt::seq::ClockType>(pi.type)) {
        clk = arg;
        continue;
      }
      if (!rst)
        if (auto it = dyn_cast<IntegerType>(pi.type);
            it && it.getWidth() == 1 && pi.name && pi.name.getValue() == "rst") {
          rst = arg;
          continue;
            }
      if (!ce)
        if (auto it = dyn_cast<IntegerType>(pi.type);
            it && it.getWidth() == 1 && pi.name && pi.name.getValue() == "ce") {
          ce = arg;
          continue;
            }
    }
  }
  return {clk, rst, ce};
}
