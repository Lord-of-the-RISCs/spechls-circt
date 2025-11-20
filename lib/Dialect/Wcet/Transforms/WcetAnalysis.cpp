//===- WcetAnalysis.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the UnrollInstr pass
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>

using namespace mlir;
using namespace spechls;
using namespace circt;

namespace wcet {
#define GEN_PASS_DEF_WCETANALYSISPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace {} // namespace

namespace wcet {

struct WcetAnalysisPattern : OpRewritePattern<spechls::KernelOp> {
  llvm::ArrayRef<unsigned int> instrs;
  using OpRewritePattern<spechls::KernelOp>::OpRewritePattern;

  // Constructor to save pass arguments
  WcetAnalysisPattern(MLIRContext *ctx, const llvm::ArrayRef<unsigned int> intrs)
      : OpRewritePattern<spechls::KernelOp>(ctx) {
    this->instrs = intrs;
  }

  LogicalResult matchAndRewrite(spechls::KernelOp top, PatternRewriter &rewriter) const override { return failure(); }
};

struct WcetAnalysisPass : public impl::WcetAnalysisPassBase<WcetAnalysisPass> {

  using WcetAnalysisPassBase::WcetAnalysisPassBase;

public:
  void runOnOperation() override {
    auto top = getOperation();
    OpPassManager dynamicPM(spechls::KernelOp::getOperationName());
    dynamicPM.addPass(wcet::createSplitDelaysPass());

    std::string unrollInstr = "";
    for (auto i : instrs) {
      unrollInstr = unrollInstr + "," + std::to_string(i);
    }
    unrollInstr[0] = '=';

    auto unrollPass = wcet::createUnrollInstrPass();
    if (failed(unrollPass->initializeOptions("instrs" + unrollInstr, [](const Twine &msg) {
          llvm::errs() << msg << '\n';
          return failure();
        })))
      return signalPassFailure();

    dynamicPM.addPass(std::move(unrollPass));
    dynamicPM.addPass(wcet::createInlineCorePass());
    dynamicPM.addPass(wcet::createInsertDummyPass());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addPass(wcet::createLongestPathPass());
    if (failed(runPipeline(dynamicPM, top))) {
      return signalPassFailure();
    }
  }
};

} // namespace wcet
