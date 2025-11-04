//===- MergeGammas.cpp - Factor Gamma inputs (mux-pushing) -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass factors identical producers that appear on multiple arms of the
// same spechls::GammaOp (a multi-input mux), by pushing the muxing to the
// *operands* of that producer and sharing a single instance of the operation.
//
// Intuition (mux-pushing / operator-sharing):
//
//   Before:
//     %p0 = comb.add %a0, %b0
//     %p1 = comb.add %a1, %b1
//     %y  = sp.Gamma %sel [%p0, %p1, %z]
//
//   After:
//     // Map outer select {0,1,...} to compact inner index over the matched set
//     %lut01 = sp.LookUpTable %sel table=[0,1,0,0,...] : i1
//
//     %x0 = sp.Gamma %lut01 [%a0, %a1]
//     %x1 = sp.Gamma %lut01 [%b0, %b1]
//     %r  = comb.add %x0, %x1
//     %y  = sp.Gamma %sel [%r, %r, %z]
//
// Correctness: when %sel picks any of the matched arms, the LUT emits the
// compact index (0..|matches|-1), each inner Gamma selects the corresponding
// original operand, and the shared op computes exactly what that arm computed.
// For other arms, the outer Gamma forwards the unmatched value unchanged.
//
// This reduces hardware resources by sharing FUs and often enables further
// canonicalization/folding.
//
// Key safety conditions enforced here:
//   * Only pure, one-result ops are factored (comb/hw.constant/selected spechls).
//   * All matched producers must be the same op *kind*, with the same operand
//     and result types, and (where relevant) equal semantics-carrying attributes.
//   * The chosen root producer's result must be single-use to avoid accidental
//     duplication of work.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/Transforms/Outlining.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"

#include "Dialect/SpecHLS/Transforms/Passes.h" // IWYU pragma: keep

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;

namespace spechls {
#define GEN_PASS_DEF_FACTORGAMMAINPUTSPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

namespace spechls {

/// Return true iff `a` and `b` are the *same* op kind with equivalent types
/// and semantics-carrying attributes. We keep this explicit and conservative
/// (rather than generic OperationEquivalence) to avoid accidentally matching
/// effectful or attribute-sensitive ops.
///
/// Pre-checked by the caller:
///   - same name
///   - same number of operands
///   - same operand types
/// Here we also check result types and op-specific attributes.
static bool areOpsSemanticallyEquivalent(Operation *a, Operation *b) {
  if (!a || !b)
    return false;

  if (a->getName() != b->getName())
    return false;

  if (a->getNumOperands() != b->getNumOperands())
    return false;

  // Operand type equality is ensured by the caller; ensure result arity & type.
  if (a->getNumResults() != b->getNumResults())
    return false;
  if (a->getNumResults() == 1 &&
      a->getResult(0).getType() != b->getResult(0).getType())
    return false;

  // Only consider one-result combinational producers for factoring.
  if (!(a->getNumResults() == 1 && b->getNumResults() == 1))
    return false;

  return TypeSwitch<Operation *, bool>(a)
      // Plain comb ops without attrs.
      .Case<comb::AddOp, comb::SubOp, comb::AndOp, comb::OrOp, comb::XorOp,
            comb::ConcatOp, comb::MulOp, comb::ShlOp, comb::ShrUOp, comb::ShrSOp,comb::DivSOp, comb::ModSOp, comb::DivUOp, comb::ModUOp, comb::MuxOp>([&](auto) { return true; })
      // Attribute-sensitive comb ops.
      .Case<comb::ICmpOp>([&](Operation *opA) {
        auto aa = cast<comb::ICmpOp>(opA);
        auto bb = dyn_cast<comb::ICmpOp>(b);
        return bb && aa.getPredicate() == bb.getPredicate();
      })
      .Case<comb::ExtractOp>([&](Operation *opA) {
        auto aa = cast<comb::ExtractOp>(opA);
        auto bb = dyn_cast<comb::ExtractOp>(b);

        return bb && aa.getLowBit() == bb.getLowBit() &&
               aa.getResult().getType().getIntOrFloatBitWidth() == bb.getResult().getType().getIntOrFloatBitWidth();
      })
      .Case<comb::TruthTableOp>([&](Operation *opA) {
        auto aa = cast<comb::TruthTableOp>(opA);
        auto bb = dyn_cast<comb::TruthTableOp>(b);
        return bb && aa.getLookupTable() == bb.getLookupTable();
      })
      // HW constants: match literal value.
      .Case<hw::ConstantOp>([&](Operation *opA) {
        auto aa = cast<hw::ConstantOp>(opA);
        auto bb = dyn_cast<hw::ConstantOp>(b);
        return bb && aa.getValue() == bb.getValue();
      })
      // spechls custom ops: be conservative and require full attribute equality.
      .Case<spechls::AlphaOp>([&](Operation *opA) {
        auto bb = dyn_cast<spechls::AlphaOp>(b);
        return bb && opA->getAttrs() == bb->getAttrs();
      })
      // Do NOT try to factor LookUpTableOps by default unless proven safe.
      .Default([&](Operation *) { return false; });
}

/// Pattern that performs one greedy factoring step:
///  1) Find a Gamma with at least two arms produced by semantically equivalent
///     one-result, pure ops.
///  2) Pick the first such arm as "root" (must be single-use).
///  3) Build a reindexing LUT that maps outer select to [0..M-1] over matches.
///  4) For each operand position j of the producers, create an inner Gamma
///     that muxes the j-th operands across the matched set (driven by the LUT).
///  5) Retarget the root producer's operands to those inner Gammas.
///  6) Rewire all matched Gamma arms to the root producer's result.
struct FactorGammaInputsPattern : OpRewritePattern<spechls::GammaOp> {
  using OpRewritePattern<spechls::GammaOp>::OpRewritePattern;

  // Set to true locally when debugging (kept off in normal builds).
  bool verbose = false;

  /// Try to collect a set of input indices {i, k, ...} in `op` that are
  /// produced by semantically equivalent ops. We stop at the first non-trivial
  /// (size > 1) set for greedy application.
  LogicalResult extractMatches(spechls::GammaOp op,
                               llvm::SmallVector<int32_t> &matches) const {
    matches.clear();
    const unsigned nbInputs = op.getInputs().size();

    for (unsigned i = 0; i < nbInputs; ++i) {
      Value rootVal = op.getInputs()[i];
      Operation *root = rootVal.getDefiningOp();
      if (!root)
        continue;

      // Only factor one-result producers that are single-use (safe sharing).
      if (root->getNumResults() != 1 || !root->getResult(0).hasOneUse())
        continue;

      // Start a new equivalence class with arm i.
      llvm::SmallVector<int32_t> cur{static_cast<int32_t>(i)};

      for (unsigned k = i + 1; k < nbInputs; ++k) {
        Value vk = op.getInputs()[k];
        Operation *ok = vk.getDefiningOp();
        if (!ok)
          continue;

        // Fast pre-checks: same name, arity, operand types.
        if (root->getName() != ok->getName())
          continue;
        if (root->getNumOperands() != ok->getNumOperands())
          continue;

        bool operandTypesMatch = true;
        for (unsigned t = 0, e = root->getNumOperands(); t < e; ++t) {
          if (root->getOperand(t).getType() != ok->getOperand(t).getType()) {
            operandTypesMatch = false;
            break;
          }
        }
        if (!operandTypesMatch)
          continue;

        if (areOpsSemanticallyEquivalent(root, ok))
          cur.push_back(static_cast<int32_t>(k));
      }

      if (cur.size() > 1) {
        matches = std::move(cur);
        return success();
      }
    }

    return failure();
  }

  /// Sanity-check that every matched producer has the same operand arity and
  /// return that arity. Returns -1 on mismatch.
  int32_t analyzeMatchedOps(spechls::GammaOp op,
                            llvm::ArrayRef<int32_t> matches) const {
    int32_t nbMatchInputs = -1;
    for (int32_t idx : matches) {
      Value v = op.getInputs()[static_cast<unsigned>(idx)];
      Operation *m = v.getDefiningOp();
      if (!m)
        continue;

      const int32_t nb = static_cast<int32_t>(m->getNumOperands());
      if (nbMatchInputs < 0) {
        nbMatchInputs = nb;
      } else if (nb != nbMatchInputs) {
        // Variadic ops with different operand counts must not be factored.
        return -1;
      }
    }
    return nbMatchInputs;
  }

  /// Build the inner reindexing LUT:
  ///   - For k in [0..nbInputs), if k ∈ matches => emit an increasing index
  ///     0,1,2,... ; otherwise emit 0.
  ///   - Pad the table to 2^(bw(select)) with zeros.
  /// The LUT result width is ceil_log2(|matches|) (>= 1 because |matches|>1).
  spechls::LUTOp createInnerReindexingLUT(spechls::GammaOp op,
                                          llvm::ArrayRef<int32_t> matches,
                                          PatternRewriter &rewriter) const {
    const unsigned nbInputs = op.getInputs().size();

    // Quick membership test via bitvector.
    llvm::SmallBitVector inSet(nbInputs, false);
    for (int32_t i : matches)
      if (i >= 0 && static_cast<unsigned>(i) < nbInputs)
        inSet.set(static_cast<unsigned>(i));

    // Compute compact indices.
    llvm::SmallVector<int32_t> lutContent;
    lutContent.reserve(nbInputs);
    int32_t nextIdx = 0;
    for (unsigned k = 0; k < nbInputs; ++k) {
      if (inSet.test(k)) {
        lutContent.push_back(nextIdx++);
      } else {
        lutContent.push_back(0);
      }
    }

    // Bitwidth checks and padding out to 2^(bw(select)).
    auto selTy = dyn_cast<IntegerType>(op.getSelect().getType());
    assert(selTy && "Gamma select must be an integer type");
    const unsigned inBW = selTy.getWidth();

    const uint64_t fullSize = 1ULL << inBW;
    (void)fullSize;
    assert(fullSize >= nbInputs &&
           "select bitwidth too small for number of Gamma inputs");

    for (uint64_t k = lutContent.size(); k < fullSize; ++k)
      lutContent.push_back(0);

    const unsigned m = static_cast<unsigned>(matches.size());
    assert(m > 1 && "reindexing LUT only built for |matches| > 1");
    const unsigned outBW = std::max(1u, llvm::Log2_64_Ceil(m));

    // Convert to 64-bit contents to satisfy LUTOp builder.
llvm::SmallVector<int64_t> lut64;
lut64.reserve(lutContent.size());
for (int32_t v : lutContent)
  lut64.push_back(static_cast<int64_t>(v));
auto lut = rewriter.create<spechls::LUTOp>(
    op.getLoc(), rewriter.getIntegerType(outBW), op.getSelect(),
    rewriter.getDenseI64ArrayAttr(lut64));

    // Verify the newly created op eagerly in debug builds.
    (void)mlir::verify(lut);
    return lut;
  }

  /// Create a Gamma for the j-th operand across all matched producers.
  /// Returns the new Gamma value (one result).
  FailureOr<Value>
  createGammaForOperand(unsigned j, spechls::GammaOp op,
                        spechls::LUTOp innerLUT,
                        llvm::ArrayRef<int32_t> matches,
                        PatternRewriter &rewriter) const {
    llvm::SmallVector<Value> args;
    args.reserve(matches.size());

    for (int32_t mid : matches) {
      const unsigned uMid = static_cast<unsigned>(mid);
      if (uMid >= op.getInputs().size())
        return failure();

      Value armVal = op->getOperand(/*0 is select*/ uMid + 1);
      Operation *producer = armVal.getDefiningOp();
      if (!producer)
        return failure();

      if (j >= producer->getNumOperands())
        return failure();

      Value arg = producer->getOperand(j);
      if (!arg)
        return failure();

      args.push_back(arg);
    }

    assert(!args.empty() && "expected at least two args (matches > 1)");
    auto gamma = rewriter.create<spechls::GammaOp>(
        op.getLoc(), args.front().getType(), op.getSymName(), innerLUT.getResult(),
        ValueRange(args));

    return gamma.getResult();
  }

  LogicalResult matchAndRewrite(spechls::GammaOp op,
                                PatternRewriter &rewriter) const override {
    // 1) Find one equivalence class of matching arms.
    llvm::SmallVector<int32_t> matches;
    if (failed(extractMatches(op, matches)))
      return failure();

    // 2) Choose the first matched arm as root and ensure single-use.
    const unsigned firstIdx = static_cast<unsigned>(matches.front());
    Value rootVal = op.getInputs()[firstIdx];
    Operation *root = rootVal.getDefiningOp();
    if (!root || root->getNumResults() != 1 ||
        !root->getResult(0).hasOneUse())
      return failure(); // We only factor when we can safely share the root.

    // 3) Check operand arity consistency across matched producers.
    int32_t nbMatchInputs = analyzeMatchedOps(op, matches);
    if (nbMatchInputs < 0)
      return failure();

    // If the producers have no operands (e.g., constants), we can still share
    // the single root by simply rewiring the outer Gamma arms.
    if (nbMatchInputs == 0) {
      rewriter.modifyOpInPlace(op, [&]() {
        for (int32_t mid : matches)
          op->setOperand(static_cast<unsigned>(mid) + 1, rootVal);
      });
      return success();
    }

    // 4) Build the inner LUT that reindexes the outer select to [0..M-1].
    auto innerLUT = createInnerReindexingLUT(op, matches, rewriter);

    // 5) Create inner Gammas, one per operand of the producer.
    llvm::SmallVector<Value> newGammas;
    newGammas.reserve(static_cast<size_t>(nbMatchInputs));

    for (unsigned j = 0; j < static_cast<unsigned>(nbMatchInputs); ++j) {
      auto gv = createGammaForOperand(j, op, innerLUT, matches, rewriter);
      if (failed(gv))
        return failure();
      newGammas.push_back(*gv);
    }

    assert(newGammas.size() == static_cast<unsigned>(nbMatchInputs) &&
           "must produce one Gamma per operand");

    // 6) Retarget the root producer to the new inner Gammas.
    rewriter.modifyOpInPlace(root, [&]() {
      for (unsigned j = 0; j < static_cast<unsigned>(nbMatchInputs); ++j)
        root->setOperand(j, newGammas[j]);
    });

    // 7) Rewire all matched Gamma arms to the root value.
    rewriter.modifyOpInPlace(op, [&]() {
      for (int32_t mid : matches)
        op->setOperand(static_cast<unsigned>(mid) + 1, rootVal);
    });

    return success();
  }
};

struct FactorGammaInputsPass
    : public impl::FactorGammaInputsPassBase<FactorGammaInputsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FactorGammaInputsPattern>(ctx);

    // New API: applyPatternsGreedily (applyPatternsAndFoldGreedily is deprecated).
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen))) {
      llvm::errs() << "FactorGammaInputs: greedy apply failed.\n";
      signalPassFailure();
      return;
    }

    if (failed(mlir::verify(getOperation()))) {
      llvm::errs() << "FactorGammaInputs: IR verification failed after pass.\n";
      signalPassFailure();
      return;
    }
  }
};

} // namespace spechls
