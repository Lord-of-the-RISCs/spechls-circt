//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/Transforms/Passes.h" // IWYU pragma: keep

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Utils.h"

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_EXPOSECONTROLFLOWSPECULATION
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "Target/Uclid/Export.h"

namespace {

struct ExposeControlFlowSpeculationPass : public spechls::impl::ExposeControlFlowSpeculationBase<ExposeControlFlowSpeculationPass> {
  FrozenRewritePatternSet patterns;

  using ExposeControlFlowSpeculationBase::ExposeControlFlowSpeculationBase;


  void runOnOperation() override {
    spechls::dumpToUclid(getOperation(), "output.uclid");
  }




};

} // namespace
