//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;

namespace SpecHLS {

struct FactorGammaInputsPattern : OpRewritePattern<GammaOp> {

  using OpRewritePattern<GammaOp>::OpRewritePattern;
  bool verbose = false;

private:
  bool isMatch(mlir::Operation *a, mlir::Operation *b) const {
    if (a->getName() != b->getName()) {
      return false;
    }
    if (verbose) {
      llvm::outs() << "\t- comparing opname " << a->getName() << " and "
                   << b->getName() << "\n";
      llvm::outs() << "\t- comparing dialect "
                   << a->getDialect()->getNamespace() << " and "
                   << b->getDialect()->getNamespace() << "\n";
    }

    if (a->getNumOperands() != b->getNumOperands()) {
      return false;
    }

    unsigned numOp = a->getNumOperands();
    for (unsigned i = 0; i < numOp; i++) {
      if (a->getOperand(i).getType() != b->getOperand(i).getType()) {
        return false;
      }
    }

    if (verbose)
      llvm::outs() << "\t- comparing #operands " << a->getNumOperands()
                   << " and " << b->getNumOperands() << "\n";

    if (b->getNumResults() == 1 && a->getNumResults() == 1) {
      if (verbose)
        llvm::outs() << "\t- match !\n";
      return true;
    }
    return false;
  }

  bool checkMatch(mlir::Operation::operand_range inputs, int i, int k,
                  SmallVector<int32_t> &matches) const {
    if (i >= inputs.size() || k >= inputs.size() || i < 0 || k < 0) {
      if (verbose)
        llvm::outs() << "\t- out of bounds  " << i << "," << k << " in "
                     << inputs.size() << "\n";
      return false;
    }

    Value va = inputs[i];
    Value vb = inputs[k];
    if (i == k || (va.getType() != vb.getType())) {
      return false;
    }

    auto a = va.getDefiningOp();
    auto b = vb.getDefiningOp();
    if (a == NULL || b == NULL) {
      return false;
    }

    if (verbose)
      llvm::outs() << "\t- comparing  (" << i << "," << k << ") ->" << *a
                   << " and " << *b << "\n";
    return isMatch(a, b);
  }

  // Cette fonction construit une liste avec la position des arguments produit
  // par des op d'une même classe d'equivalence (definie par checkmatch)
  LogicalResult extractMatches(GammaOp op,
                               SmallVector<int32_t> &matches) const {
    // llvm::errs() << "analyzing  " << op << " \n";

    u_int32_t nbInputs = op.getInputs().size();
    for (int i = 0; i < nbInputs; i++) {
      auto rootValue = op.getInputs()[i];
      auto root = rootValue.getDefiningOp();
      if (root == NULL || root->getNumResults() != 1 ||
          !root->getResult(0).hasOneUse()) {
        continue;
      }

      /* build the set of nodes (at pos K>i) that match the current target
       * node */
      matches.clear();
      matches.push_back(i);
      for (int k = i + 1; k < nbInputs; k++) {
        if (checkMatch(op.getInputs(), i, k, matches)) {
          matches.push_back(k);
        }
      }

      if (matches.size() > 1) {
        if (verbose) {
          llvm::outs() << "match {\n";
          for (auto m : matches) {
            auto defOp = op.getInputs()[m].getDefiningOp();
            llvm::outs() << "\tin[" << m << "] -> " << *defOp << "\n";
          }
          llvm::outs() << "}\n";
        }

        return success();
      }
    }
    return failure();
  }

  SpecHLS::GammaOp createGammaForOperand(u_int32_t j, GammaOp op,
                                         LookUpTableOp innerLUT,
                                         SmallVector<int32_t> &matches,
                                         PatternRewriter &rewriter) const {
    SmallVector<Value> args;
    if (verbose)
      llvm::outs() << "-Extracting all " << j << "th args in matched ops \n";

    for (auto mid : matches) {
      auto value = op.getInputs()[mid];
      if (!value) {
        llvm::errs() << "Invalid value at " << mid << "\n";
        return NULL;
      }

      auto matchedOp = value.getDefiningOp();
      if (!matchedOp) {
        llvm::errs() << "No defining op for value " << value << " at offset "
                     << mid << "\n";
        return NULL;
      }

      if (j >= matchedOp->getNumOperands()) {
        continue;
      }

      if (verbose)
        llvm::outs() << "\t-analyzing match " << *matchedOp << " at offset  "
                     << j << "\n";

      auto matchedArgValue = matchedOp->getOperand(j);
      if (!matchedArgValue) {
        llvm::errs() << "No valid arrgValue at offset " << j << "\n";
        return NULL;
      }

      if (verbose)
        llvm::outs() << "\t-extracting value " << matchedArgValue << " \n";
      args.push_back(matchedArgValue);
    }
    auto gamma = rewriter.create<SpecHLS::GammaOp>(
        op->getLoc(), args[0].getType(), op.getName(), innerLUT->getResult(0),
        args);
    if (verbose)
      llvm::outs() << "- Creating inner gamma " << gamma << " at offset  " << j
                   << "\n";
    return gamma;
  }

  int32_t analyzeMatchedOps(GammaOp op, SmallVector<int32_t> &matches) const {
    int32_t nbMatchInputs = -1;
    Operation *rootMatchedOp;
    u_int32_t nbInputs = op.getInputs().size();

    /* computes the number of inputs on matched ops */
    for (auto mid : matches) {
      auto matchedValue = op.getInputs()[mid];
      if (!matchedValue) {
        continue;
      }

      auto matchedOp = matchedValue.getDefiningOp();
      if (!matchedOp) {
        continue;
      }

      auto nbOperands = matchedOp->getNumOperands();
      if (nbMatchInputs < 0) {
        nbMatchInputs = nbOperands;
        // We keep track of one of the matched op
        rootMatchedOp = matchedOp;
        if (verbose)
          llvm::outs() << "Reference matched op " << *matchedOp << "\n";
      }
      if (nbOperands != nbMatchInputs) {
        llvm::errs() << "Inconsistent arity for " << *matchedOp << ", expected "
                     << nbOperands << "\n";
        return -1;
      }
    }
    return nbMatchInputs;
  }

  SpecHLS::LookUpTableOp
  createInnerReindexingLUT(GammaOp op, SmallVector<int32_t> &matches,
                           PatternRewriter &rewriter) const {
    SmallVector<int> lutContent;
    auto nbInputs = op.getInputs().size();
    int pos = 0;
    if (verbose)
      llvm::outs() << "Reindexing inner Gamma op inputs \n";
    for (int k = 0; k < nbInputs; k++) {
      auto newIndex = 0;
      if (std::count_if(matches.begin(), matches.end(),
                        [&](const auto &e) { return (k == e); })) {
        newIndex = pos++;
      }
      lutContent.push_back(newIndex);
      if (verbose)
        llvm::outs() << "- " << k << " -> " << newIndex << "\n";
    }

    auto innerBW = APInt(32, matches.size()).ceilLogBase2();
    return rewriter.create<SpecHLS::LookUpTableOp>(
        op->getLoc(), rewriter.getIntegerType(innerBW), op.getSelect(),
        rewriter.getI32ArrayAttr(lutContent));
  }

public:
  LogicalResult matchAndRewrite(GammaOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<int32_t> matches;
    u_int32_t nbInputs = op.getInputs().size();
    if (!extractMatches(op, matches).succeeded()) {
      return failure();
    }

    auto firstMatchIndex = matches[0];
    auto rootValue = op.getInputs()[firstMatchIndex];
    auto root = rootValue.getDefiningOp();
    if (root->getNumResults() != 1 || !root->getResult(0).hasOneUse()) {
      return failure();
    }

    auto innerLUT = createInnerReindexingLUT(op, matches, rewriter);
    auto nbMatchInputs = analyzeMatchedOps(op, matches);

    SmallVector<Value> newGammas;
    for (u_int32_t j = 0; j < nbMatchInputs; j++) {
      auto gamma = createGammaForOperand(j, op, innerLUT, matches, rewriter);
      newGammas.push_back(gamma);
    }

    assert(newGammas.size() == nbMatchInputs);

    if (verbose)
      llvm::outs() << "Rewiring " << nbMatchInputs
                   << " arguments in the root op " << *root << "\n";

    for (u_int32_t j = 0; j < nbMatchInputs; j++) {
      if (verbose)
        llvm::outs() << "\t-replace arg[" << j << "]=" << root->getOperand(j)
                     << " by  " << newGammas[j] << "\n";

      root->setOperand(j, newGammas[j]);
    }

    if (verbose) {
      llvm::outs() << "Root op is now " << *root << "\n";
      llvm::outs() << "Before rewiring outer gamma " << op << "\n";
    }

    for (auto mid : matches) {
      if (verbose)
        llvm::outs() << "\t- Update operand from " << op->getOperand(mid + 1)
                     << " to " << rootValue << "\n";
      op->setOperand(mid + 1, rootValue);
      if (verbose)
        llvm::outs() << "\t - " << op << "\n";
    }

    if (verbose) {
      llvm::outs() << "After rewiring outer gamma " << op << "\n";
      llvm::outs() << "##################################################"
                      "############\n\n\n";
      llvm::outs() << "##################################################"
                      "############\n";
      llvm::outs() << *op->getParentOp() << "\n";
    }

    return success();
  }
};

struct FactorGammaInputsPass
    : public impl::FactorGammaInputsPassBase<FactorGammaInputsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<FactorGammaInputsPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<>> createFactorGammaInputsPass() {
  return std::make_unique<FactorGammaInputsPass>();
}

} // namespace SpecHLS
