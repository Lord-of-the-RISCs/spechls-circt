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
#define GEN_PASS_DEF_ANNOTATETIMINGPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

class AnnotateTimingPass : public spechls::impl::AnnotateTimingPassBase<AnnotateTimingPass> {
public:
  using AnnotateTimingPassBase::AnnotateTimingPassBase;

  bool hasFailed = false;

  mlir::WalkResult annotateOperation(mlir::Operation *op, spechls::TimingAnalyser &analyser, double clockPeriod,
                                     mlir::MLIRContext *ctx) {
    auto timing = analyser.computeOperationTiming(op, clockPeriod, ctx);
    if (timing) {
      auto [in, lat, out] = *timing;
      if (lat == 0) {
        op->setAttr("spechls.combDelay", mlir::FloatAttr::get(mlir::Float64Type::get(ctx), in));
      } else {
        op->setAttr("spechls.combDelay",
                    mlir::FloatAttr::get(mlir::Float64Type::get(ctx), in + clockPeriod * lat + out));
      }
      return mlir::WalkResult::advance();
    }
    return mlir::WalkResult::interrupt();
  }

  void annotateTask(spechls::TaskOp &task, spechls::TimingAnalyser &analyser, double clockPeriod,
                    mlir::MLIRContext *ctx) {
    task.walk([&](mlir::Operation *op) { return annotateOperation(op, analyser, clockPeriod, ctx); });
  }

  void runOnOperation() override {
    if (failed(spechls::timingAnalyserFactory.registerAnalyers(targetsFile)))
      return signalPassFailure();

    auto analyser = spechls::timingAnalyserFactory.get(target);
    mlir::MLIRContext *ctx = &getContext();

    if (task != "") {
      getOperation().walk([&](spechls::TaskOp t) {
        if (t.getSymName() == task) {
          annotateTask(t, analyser, clockPeriod, ctx);
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
    } else {
      getOperation().getBody().getBlocks().front().walk([&](mlir::Operation *op) {
        if (auto t = llvm::dyn_cast<spechls::TaskOp>(op)) {
          annotateTask(t, analyser, clockPeriod, ctx);
        } else {
          annotateOperation(op, analyser, clockPeriod, ctx);
        }
        if (hasFailed)
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      });
    }
    if (hasFailed)
      return signalPassFailure();
  }
};

} // namespace