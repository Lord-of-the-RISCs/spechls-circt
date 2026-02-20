//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/SpecHLS/OperationDelayAnalysis.h"
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/WalkResult.h>

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <cmath>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_AFFECTPROFILINGIDPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

class AffectProfilingID : public spechls::impl::AffectProfilingIDPassBase<AffectProfilingID> {
public:
  using AffectProfilingIDPassBase::AffectProfilingIDPassBase;

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();
      int id = 0;
      getOperation().walk([&](spechls::GammaOp t) {
      t->setAttr("spechls.profilingId", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), id));
      id++;
      });
  }
};

} // namespace