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
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"

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


#include <Conversion/SpecHLS/SpecHLSToHW/common.h>

namespace spechls {
#define GEN_PASS_DEF_SPECHLSTOSEQPASS
#include "Conversion/SpecHLS/Passes.h.inc"
}; // namespace spechls

using namespace mlir;
using namespace circt;

#define debug llvm::outs
//#define debug llvm::nulls

namespace {




/// Normalize index to the required address width.
Value normalizeIndex(Value idx, uint64_t size,
                     PatternRewriter &rewriter, Location loc)  {
  unsigned addrWidth = llvm::Log2_64_Ceil(size ? size : 1);
  // If your index type is already `iN` with N >= addrWidth, truncate.
  // If it's smaller, zext. This is deliberately simplistic; adapt as needed.
  auto intTy = dyn_cast<IntegerType>(idx.getType());
  if (!intTy || intTy.getWidth() == addrWidth)
    return idx;


  unsigned curWidth = intTy.getWidth();
  if (curWidth == addrWidth)
    return idx;

  // Truncate: keep low addrWidth bits.
  if (curWidth > addrWidth) {
    auto targetTy = rewriter.getIntegerType(addrWidth);
    // comb.extract(input, lowBit) → resultType
    return rewriter.createOrFold<comb::ExtractOp>(
        loc, targetTy, idx,
        /*lowBit=*/0);
  }

  // Zero-extend: prepend padWidth zero bits and concat.
  unsigned padWidth = addrWidth - curWidth;
  auto padVal = rewriter.create<hw::ConstantOp>(
      loc,  APInt::getZero(padWidth));

  // Result type is inferred as i[padWidth + curWidth].
  Value extended = rewriter.createOrFold<comb::ConcatOp>(loc, padVal, idx);

  // In principle, extended already has width == addrWidth.
  return extended;
}

inline void printOpUsers(llvm::raw_ostream &os, mlir::Operation *op) {
  if (!op) {
    os << "null op\n";
    return;
  }

  os << "Operation: ";
  op->print(os);
  os << " by [";

  bool first = true;
  for (auto result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (!first)
        os << ", ";
      user->print(os);
      first = false;
    }
  }

  if (first)
    os << "none";

  os << "]\n";
}






struct ConvertSpecHLSToSeqPass
    : public spechls::impl::SpecHLSToSeqPassBase<ConvertSpecHLSToSeqPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSpecHLSToSeqPass)


  struct DelayConversion : OpConversionPattern<spechls::DelayOp> {
    using OpConversionPattern<spechls::DelayOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(spechls::DelayOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      auto [pClk, pRst, pCe] =
          getCREArgs(op->getParentOfType<circt::hw::HWModuleOp>());
      if (!pClk || !pRst || !pCe)
        return rewriter.notifyMatchFailure(op, "parent hw.module is missing CRE ports");


      Value v = op.getInput();
      if (!v)
        return rewriter.notifyMatchFailure(op, "missing delay input");
      auto loc = op.getLoc();

      unsigned depth = op.getDepth();
      for (unsigned i = 0; i < depth; ++i) {
        v = rewriter.create<circt::seq::CompRegClockEnabledOp>(loc,  v, pClk, pCe, "d_"+std::to_string(i))
                .getResult();
      }

      rewriter.replaceOp(op, v);
      return success();
    }

  };

  struct ScalarMuConversion : OpConversionPattern<spechls::MuOp> {
    using OpConversionPattern<spechls::MuOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(spechls::MuOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      auto [pClk, pRst, pCe] =
          getCREArgs(op->getParentOfType<circt::hw::HWModuleOp>());
      if (!pClk || !pRst || !pCe) {
        debug() << "MuOp parent hw.module is missing CRE ports: " << *op << "\n";
        return rewriter.notifyMatchFailure(op, "parent hw.module is missing CRE ports");
      }
      //   return failure(); // Only handle array loads here.
        auto intType = dyn_cast<IntegerType>(op.getType());
        if (intType== nullptr ) {
          return failure();
          //return rewriter.notifyMatchFailure(op, "only integer MuOp are supported");
        } else {

          Value loop = op.getLoopValue();
          Value init = op.getInitValue();
          Value d = rewriter.create<circt::comb::MuxOp>(op.getLoc(), loop.getType(), pRst, loop, init);
          auto loc = op.getLoc();
          auto mu_reg = rewriter.create<circt::seq::CompRegClockEnabledOp>(loc,  d, pClk, pCe, "mu_"+op.getSymName().str())
                   .getResult();
          debug() << "Replacing MuOp: " << *op << " by reg: " << mu_reg << "\n";
          rewriter.replaceOp(op, mu_reg);
          return success();
        }
      }


  };

  struct ArrayMuConversion : OpConversionPattern<spechls::MuOp> {
    using OpConversionPattern<spechls::MuOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(spechls::MuOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      // // check that op input type is intgere type
      auto arrayTy = dyn_cast<spechls::ArrayType>(op.getType());
      if (arrayTy) {
        if (!op.getResult().use_empty()) {
          // get number of users
          if (op.getResult().getNumUses()==1) {;
            // check if the only user is the mu itself
            auto user = *op.getResult().getUsers().begin();
            if (isa<spechls::MuOp>(user)) {
              debug() << "Array MuOp is only used by itslef: " << *op << " by \n";

              rewriter.eraseOp(op);
              return success();
            }
          } else {
            debug() << "Array MuOp has multiple users: " << *op << " by \n";
            printOpUsers(debug(), op);
            return failure();
          }
        } else {
          debug()  << "Erasing dead MuOp: " << *op << "\n";
          rewriter.eraseOp(op);
          return success();
        }
      }
      return failure();
    }

  };
  /// Helper to walk back array SSA chain to the defining MuOp.
  static spechls::MuOp findRootMu(Value arrayVal) {
    while (arrayVal!=nullptr) {
      debug() << "Traversing " << arrayVal << "\n";
      if (auto mu = arrayVal.getDefiningOp<spechls::MuOp>()) {
      debug() << "Found MuOp: " << mu << "\n";
      return mu;
      }
      if (auto alpha = arrayVal.getDefiningOp<spechls::AlphaOp>()) {

        arrayVal = alpha.getArray();
        continue;
      }
      if (auto mux = arrayVal.getDefiningOp<comb::MuxOp>()) {

        arrayVal = mux.getOperand(0);
        continue;
      }
      if (auto sync = arrayVal.getDefiningOp<spechls::SyncOp>()) {

        arrayVal = sync.getOperand(0);
        continue;
      }

      // TODO: optionally support other producers here (phi-like, selects, etc.)
      return nullptr;
    }
  }

 /// Pattern for lowering LoadOp.
  struct LoadPattern : OpRewritePattern<spechls::LoadOp> {
    ConvertSpecHLSToSeqPass *parent;
    using OpRewritePattern::OpRewritePattern;

    LoadPattern(MLIRContext *ctx, ConvertSpecHLSToSeqPass *p)
        : OpRewritePattern(ctx), parent(p) {}

    LogicalResult matchAndRewrite(spechls::LoadOp op,
                                  PatternRewriter &rewriter) const override {
      auto arrayTy = dyn_cast<spechls::ArrayType>(op.getArray().getType());
      if (!arrayTy)
        return failure(); // Only handle array loads here.

      debug() << "Lowering LoadOp: " << op << "\n";
      auto mu = findRootMu(op.getArray());
      if (!mu) {
        op.emitError("LoadOp array is not ultimately defined by a MuOp");
        return failure();
      }
      debug() << "Found source mu: " << mu << "\n";
      auto mem = parent->getOrCreateMemForMu(mu, rewriter);
      if (!mem)
        return failure();

      debug() << "Found target memory : " << mem << "\n";

      Location loc = op.getLoc();
      uint64_t size = arrayTy.getSize();
      Value addr = normalizeIndex(op.getIndex(), size, rewriter, loc);

      // Enable = 1 (if you have predication, plug it here).
      Value en =
          rewriter.create<hw::ConstantOp>(loc,APInt(1, 1));

      auto elemTy = arrayTy.getElementType();
      auto read   = rewriter.create<seq::ReadPortOp>(
          loc, elemTy, mem.getHandle(), ValueRange{addr}, en,
          /*latency=*/0);

      debug() << "replace " << op << " by "<< read << "\n";
      rewriter.replaceOp(op, read.getResult());
      return success();
    }
  };

  /// Pattern for lowering AlphaOp.
  struct AlphaPattern : OpRewritePattern<spechls::AlphaOp> {
    ConvertSpecHLSToSeqPass *parent;
    using OpRewritePattern::OpRewritePattern;

    AlphaPattern(MLIRContext *ctx, ConvertSpecHLSToSeqPass *p)
        : OpRewritePattern(ctx), parent(p) {}


    LogicalResult matchAndRewrite(spechls::AlphaOp op,
                                  PatternRewriter &rewriter) const override {
      auto arrayTy = dyn_cast<spechls::ArrayType>(op.getArray().getType());
      if (!arrayTy)
        return failure();

      auto mu = findRootMu(op.getArray());
      if (!mu) {
        op.emitError("AlphaOp array is not ultimately defined by a MuOp");
        return failure();
      }

      auto mem = parent->getOrCreateMemForMu(mu, rewriter);
      debug() << "Found target memory : " << mem << "\n";


      if (!mem)
        return failure();

      Location loc = op.getLoc();
      uint64_t size = arrayTy.getSize();
      Value addr = normalizeIndex(op.getIndex(), size, rewriter, loc);

      Value data = op.getValue();
      Value en   = op.getWe();

      unsigned latency = 1;
      if (auto latAttr = op->getAttrOfType<IntegerAttr>("latency"))
        latency = latAttr.getInt();

      auto write  =rewriter.create<seq::WritePortOp>(
          loc, mem.getHandle(), ValueRange{addr}, data, en, latency);

      debug() << "Add " << write << " to "<< mem << "\n";
      // The Alpha result is now redundant: forward the input array value.
      rewriter.replaceOp(op, op.getArray());
      return success();
    }
  };

  /// Pattern to erase dead MuOps after lowering.
  struct EraseDeadMuPattern : OpRewritePattern<spechls::MuOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(spechls::MuOp mu,
                                  PatternRewriter &rewriter) const override {
      if (!mu.getResult().use_empty())
        return failure();
      llvm::outs() << "Erasing dead MuOp: " << *mu << "\n";
      rewriter.eraseOp(mu);
      return success();
    }
  };

  DenseMap<Operation *, seq::HLMemOp> muToMem;

  /// Get or create the HLMemOp associated with a given MuOp.
  seq::HLMemOp getOrCreateMemForMu(spechls::MuOp mu, PatternRewriter &rewriter) {
    if (!mu)
      return nullptr;
    debug() << "Getting/creating HLMem for MuOp: " << *mu << "\n";
    auto it = muToMem.find(mu.getOperation());
    if (it != muToMem.end()) {
      debug() << "Found existing HLMem for MuOp: " << *mu << "\n";

      return it->second;

    }

    auto arrayTy = dyn_cast<spechls::ArrayType>(mu.getResult().getType());
    if (!arrayTy) {
      mu.emitError("expected ArrayType result on MuOp to create hlmem");
      return nullptr;
    }

    Location loc = mu.getLoc();
    auto elemTy  = arrayTy.getElementType();
    uint64_t size = arrayTy.getSize();

    // Build HLMem type: <size x elemTy>
    auto memTy = seq::HLMemType::get(rewriter.getContext(),
                                SmallVector<int64_t>{(int64_t)size},
                                elemTy);

    // Insert the memory in a stable place (e.g., top of the kernel).
    auto topmodule = mu->getParentOfType<hw::HWModuleOp>();
    if (!topmodule) {
      mu.emitError("MuOp not inside a HWModule");
      return nullptr;
    }

    OpBuilder::InsertionGuard guard(rewriter);
    Block &entry = topmodule.getBody().front();
    rewriter.setInsertionPointToStart(&entry);

    // Find or invent a clock.
    auto [ clk, rst, ce] = getCREArgs(mu->getParentOfType<circt::hw::HWModuleOp>());
    if (!clk) {
      mu.emitError("no clock found to create seq.hlmem");
      return nullptr;
    }

    // Give the memory a name derived from mu.sym_name or a fallback.
    StringRef baseName = "spechls_mem";
    if (auto sym = mu.getSymNameAttr())
      baseName = sym.strref();

    auto mem =
        rewriter.create<seq::HLMemOp>(loc, memTy, clk, rst,
                                 rewriter.getStringAttr(baseName));
    debug() << "Create  HLMem : " << *mem << "\n";
    muToMem[mu] = mem;
    return mem;
  }
  void runOnOperation() override {
    auto root = getOperation();
    auto [ clk, rst,ce ] = getCREArgs(root);
    if (!clk) {
      root.emitError() << "SpecHLSToSeq: enclosing hw.module must provide a seq.clock input";
      signalPassFailure();
      return;
    }


    MLIRContext * ctxt = &getContext();
    RewritePatternSet patterns(ctxt);
    patterns.add<DelayConversion>(ctxt);
    patterns.add<LoadPattern, AlphaPattern>(ctxt, this);
    patterns.add<ScalarMuConversion>(ctxt);

    ConversionTarget target(getContext());
    target.addLegalDialect<hw::HWDialect, seq::SeqDialect, comb::CombDialect>();
    target.addIllegalOp<spechls::MuOp,spechls::DelayOp,spechls::AlphaOp,spechls::LoadOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // Make MuOps that produce an ArrayType dynamically legal so array-based Mus
    // are left in place for the mem/alpha/load patterns to handle.
    target.addDynamicallyLegalOp<spechls::MuOp>([](spechls::MuOp mu) {
      return llvm::isa<spechls::ArrayType>(mu.getResult().getType());
    });
    if (failed(applyFullConversion(root, target, std::move(patterns))))
      signalPassFailure();


     ConversionTarget targetmu(getContext());
    targetmu.addLegalDialect<hw::HWDialect, seq::SeqDialect, comb::CombDialect>();
    targetmu.addIllegalOp<spechls::MuOp, spechls::DelayOp,spechls::AlphaOp,spechls::LoadOp>();
    RewritePatternSet mupatterns(ctxt);

     //mupatterns.add<ArrayMuConversion,EraseDeadMuPattern>(ctxt);
     mupatterns.add<ArrayMuConversion>(ctxt);
    //
    if (failed(applyFullConversion(root, targetmu, std::move(mupatterns))))
      signalPassFailure();

     }
};


} // namespace

std::unique_ptr<mlir::Pass> createSpecHLSToSeqPass() {
  return std::make_unique<ConvertSpecHLSToSeqPass>();
}
