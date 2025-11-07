//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===- SpecHLSToSeq.cpp ----------------------------------------*- C++ -*-===//
// Lower SpecHLS state-holding ops to CIRCT seq registers (with CE and reset).
//
// This pass rewrites:
//   - spechls.mu     -> seq.compreg.ce (clock-enabled register)
//   - spechls.delay  -> a chain of seq.compreg.ce of depth = delay
// It handles reset by gating the D input with a MUX (rst ? resetValue : D)
// for portability across CIRCT versions (instead of relying on *Reset* forms).
//
// Clock/reset/CE ports are retrieved from the surrounding hw.module inputs
// when present. If CE or RST are missing, we synthesize constants (ce=1, rst=0).
// For CLK, we require it to be present (seq regs need a clock).
//
// If you want to *add* missing ports to hw.module, see the TODO notes at the
// end of this file for a safe module-rebuild recipe.
//
//===----------------------------------------------------------------------===//

#include "Conversion/SpecHLS/Passes.h"
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace spechls {
#define GEN_PASS_DEF_SPECHLSTOSEQPASS
#include "Conversion/SpecHLS/Passes.h.inc"
}; // namespace spechls

using namespace mlir;
using namespace circt;

namespace {

struct ClockResetEnableInfo {
  Value clk; // required
  Value rst; // optional (may be null)
  Value ce;  // optional (may be null)
};

static FailureOr<ClockResetEnableInfo> findCRE(hw::HWModuleOp module) {

  ClockResetEnableInfo info{};
  auto ports = module.getPortList();
  for (auto &p : ports) {
    if (!p.isInput())
      continue;
    Value arg = module.getArgumentForInput(p.argNum);

    // Clock is a seq.clock
    if (!info.clk)
      if (isa<seq::ClockType>(p.type)) {
        info.clk = arg;
        continue;
      }

    // rst: i1, commonly named "rst"
    if (!info.rst)
      if (auto it = dyn_cast<IntegerType>(p.type); it && it.getWidth() == 1)
        if (p.name == "rst") {
          info.rst = arg;
          continue;
        }

    // ce: i1, commonly named "ce"
    if (!info.ce)
      if (auto it = dyn_cast<IntegerType>(p.type); it && it.getWidth() == 1)
        if (p.name == "ce") {
          info.ce = arg;
          continue;
        }
  }

  if (!info.clk)
    return failure();
  return info;
}

/// Create an all-zero constant for integer-typed values. If unsupported, return
/// null and the caller should skip reset gating for that type.
static Value createZeroLike(Location loc, Type ty, OpBuilder &b) {
  if (auto it = dyn_cast<IntegerType>(ty)) {
    auto zero = b.getIntegerAttr(it, 0);
    return b.create<hw::ConstantOp>(loc, zero);
  }
  // Add more cases here if you need arrays/structs (build with hw.constant*).
  return Value();
}

/// Apply synchronous reset by muxing D with a zero value: rst ? 0 : D.
static Value applyResetIfAvailable(Location loc, Value d, Value rst,
                                   OpBuilder &b) {
  if (!rst)
    return d;
  auto zero = createZeroLike(loc, d.getType(), b);
  if (!zero)
    return d; // unsupported type: skip reset gating
  // twoState=false (use 4-state semantics if available in comb)
  return b.create<comb::MuxOp>(loc, rst, zero, d, /*twoState=*/false);
}

struct IdentityTypeConverter : TypeConverter {
  IdentityTypeConverter() { addConversion([](Type t) { return t; }); }
};

/// spechls.mu -> seq.compreg.ce (with optional reset gating)
struct MuConversion : OpConversionPattern<spechls::MuOp> {
  MuConversion(TypeConverter &tc, MLIRContext *ctx, const ClockResetEnableInfo &cre)
      : OpConversionPattern<spechls::MuOp>(tc, ctx), cre(cre) {}

  using Adaptor = typename OpConversionPattern<spechls::MuOp>::OpAdaptor;

  LogicalResult matchAndRewrite(spechls::MuOp op, Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (!cre.clk)
      return rewriter.notifyMatchFailure(op, "no clk in enclosing hw.module");

    // Choose the data to register (adjust to your Mu semantics if needed).
    Value d = op.getLoopValue();
    if (!d)
      return rewriter.notifyMatchFailure(op, "missing mu input");

    // CE: constant true if not provided.
    Value ce = cre.ce;
    if (!ce) {
      ce = rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(),1);
    }

    // Synchronous reset via mux on D.
    Value gatedD = applyResetIfAvailable(loc, d, cre.rst, rewriter);

     // for potential naming use
    auto reg = rewriter.create<seq::CompRegClockEnabledOp>(loc, gatedD, cre.clk, ce, op.getSymName().str());

    rewriter.replaceOp(op, reg.getResult());
    return success();
  }

  ClockResetEnableInfo cre;
};

/// spechls.delay -> N chained seq.compreg.ce registers
struct DelayConversion : OpConversionPattern<spechls::DelayOp> {
  DelayConversion(TypeConverter &tc, MLIRContext *ctx, const ClockResetEnableInfo &cre)
      : OpConversionPattern<spechls::DelayOp>(tc, ctx), cre(cre) {}

  using Adaptor = typename OpConversionPattern<spechls::DelayOp>::OpAdaptor;

  LogicalResult matchAndRewrite(spechls::DelayOp op, Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (!cre.clk)
      return rewriter.notifyMatchFailure(op, "no clk in enclosing hw.module");

    Value v = op.getInput();
    if (!v)
      return rewriter.notifyMatchFailure(op, "missing delay input");

    Value ce = cre.ce;
    if (!ce)
      ce = rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(),1);

    unsigned depth = op.getDepth();
    for (unsigned i = 0; i < depth; ++i) {
      Value d = applyResetIfAvailable(loc, v, cre.rst, rewriter);
      v = rewriter.create<seq::CompRegClockEnabledOp>(loc,  d, cre.clk, ce, "d_"+std::to_string(i))
              .getResult();
    }

    rewriter.replaceOp(op, v);
    return success();
  }

  ClockResetEnableInfo cre;
};

struct ConvertSpecHLSToSeqPass
    : public spechls::impl::SpecHLSToSeqPassBase<ConvertSpecHLSToSeqPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSpecHLSToSeqPass)
  StringRef getArgument() const final { return "convert-spechls-to-seq"; }

  StringRef getDescription() const final {
    return "Lower SpecHLS Mu/Delay to CIRCT seq compreg.ce with CE/reset";
  }

  void runOnOperation() override {
    auto root = getOperation();
    auto creOr = findCRE(root);
    if (failed(creOr)) {
      root.emitError() << "SpecHLSToSeq: enclosing hw.module must provide a seq.clock input";
      signalPassFailure();
      return;
    }
    ClockResetEnableInfo cre = *creOr;

    IdentityTypeConverter typeConv;
    RewritePatternSet patterns(&getContext());
    patterns.add<MuConversion>(typeConv, &getContext(), cre);
    patterns.add<DelayConversion>(typeConv, &getContext(), cre);

    ConversionTarget target(getContext());
    target.addLegalDialect<hw::HWDialect, seq::SeqDialect, comb::CombDialect>();
    target.addIllegalOp<spechls::MuOp, spechls::DelayOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyFullConversion(root, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSpecHLSToSeqPass() {
  return std::make_unique<ConvertSpecHLSToSeqPass>();
}

//===----------------------------------------------------------------------===//
// TODO: Adding missing RST/CE/CLK ports to hw.module
//
// Rebuilding an hw.module with extra input ports requires creating a *new*
// hw.module with the updated ModulePortInfo, cloning the body while remapping
// block arguments, fixing symbol uses, and erasing the old module. CIRCT has
// utilities for port handling, but the safest pattern is:
//   1) collect old ports (module.getPorts()),
//   2) append PortInfo for the new inputs (e.g., clk: seq.clock, rst:i1, ce:i1),
//   3) create a new hw.module with the same symbol name and new ports,
//   4) clone ops from old body to new body with IRMapping that maps
//      old block args → new block args for the original inputs,
//   5) replace all symbol uses (SymbolTable::replaceAllSymbolUses) if needed,
//      then erase the old module.
// Doing this correctly depends on your project’s symbol/instance conventions,
// so it’s left as a follow-up utility in your codebase.
//===----------------------------------------------------------------------===//
