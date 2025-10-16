#include "Dialect/SpecHLS/Transforms/Passes.h"

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

#include <Dialect/SpecHLS/IR/SpecHLSOps.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_REMOVEINTRARAWPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "RemoveIntraRAW.h.inc"

namespace {
struct RemoveIntraRAWPass : public spechls::impl::RemoveIntraRAWPassBase<RemoveIntraRAWPass> {
  FrozenRewritePatternSet patterns;

  using RemoveIntraRAWPassBase::RemoveIntraRAWPassBase;

  void runOnOperation() override { return; }
};
} // namespace
