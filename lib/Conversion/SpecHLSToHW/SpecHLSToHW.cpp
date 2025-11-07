//===- SpecHLSToHW.cpp ----------------------------------------*- C++ -*-===//
//
// Convert SpecHLS ops to CIRCT HW/Comb, ensuring every created hw.module has
// clk: !seq.clock, rst: i1, ce: i1 inputs; consistently wire parent->child.
//
// This pass is declared in ODS as:
//   def SpecHLSToHWPass : Pass<"spechls-to-hw", "spechls::KernelOp"> { ... }
//
// Structure (close to an “original” staged lowering):
//   1) KernelOp -> HWModuleOp   (append clk/rst/ce; inline body; hook outputs)
//   2) Inside that module:
//        TaskOp   -> child HWModuleOp (+clk/rst/ce) + hw.instance (forward CRE)
//        GammaOp  -> comb.mux (2-way) | hw.array_get(array_create(...)) (N-way)
//
// API: Latest MLIR/CIRCT (2025). Uses applyPatternsGreedily only for local
//      Task/Gamma rewrites inside the newly emitted hw.module.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"    // OpBuilder::mergeBlockBefore / PatternRewriter::mergeBlockBefore
#include "mlir/IR/Operation.h"   // Operation::getParentOfType
#include "mlir/IR/PatternMatch.h" // OpRewritePattern
#include <tuple>

#define DEBUG_TYPE "spechls-to-hw"


#ifndef LEGACY_PASS
using namespace mlir;
using namespace circt;

namespace spechls {
#define GEN_PASS_DEF_SPECHLSTOHWPASS
#include "Conversion/SpecHLS/Passes.h.inc"
}; // namespace spechls


namespace {

/// Small helpers for port assembly and CRE discovery.
static hw::PortInfo makeInputPort(MLIRContext *ctx, StringRef name, Type ty) {
  hw::PortInfo p({  StringAttr::get(ctx, name), ty, hw::ModulePort::Input });
  return p;
}
static hw::PortInfo makeOutputPort(MLIRContext *ctx, StringRef name, Type ty) {
  hw::PortInfo p({  StringAttr::get(ctx, name), ty, hw::ModulePort::Output });
  return p;
}

/// Return (%clk,%rst,%ce) block arguments of an hw.module.
static std::tuple<Value, Value, Value> getCREArgs(hw::HWModuleOp mod) {
  Value clk, rst, ce;
  for (auto &pi : mod.getPortList()) {
    if (pi.isInput()) {
      Value arg = mod.getArgumentForInput(pi.argNum);
      if (!clk && isa<seq::ClockType>(pi.type)) { clk = arg; continue; }
      if (!rst) if (auto it = dyn_cast<IntegerType>(pi.type);
                    it && it.getWidth() == 1 && pi.name && pi.name.getValue() == "rst") { rst = arg; continue; }
      if (!ce)  if (auto it = dyn_cast<IntegerType>(pi.type);
                    it && it.getWidth() == 1 && pi.name && pi.name.getValue() == "ce")  { ce  = arg; continue; }

    }
  }
  return {clk, rst, ce};
}

/// Lower `spechls.gamma` locally (no dialect conversion framework needed).
struct GammaRewriter : OpRewritePattern<spechls::GammaOp> {
  using OpRewritePattern<spechls::GammaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spechls::GammaOp gamma,
                                PatternRewriter &rewriter) const override {
    auto loc = gamma.getLoc();
    auto inVals = gamma.getInputs();
    if (inVals.size() == 2) {
      // 2-way: comb.mux(select, in1, in0) -- note order (true, false).
      rewriter.replaceOpWithNewOp<comb::MuxOp>(
          gamma, gamma.getType(), gamma.getSelect(), inVals[1], inVals[0],
          /*twoState=*/false);
      return success();
    }
    // N-way: array[index]
    Type elemTy = gamma.getType();
    auto arrTy = hw::ArrayType::get(elemTy, inVals.size());
    auto arr = rewriter.create<hw::ArrayCreateOp>(loc, arrTy, inVals);
    rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(gamma, elemTy, arr, gamma.getSelect());
    return success();
  }
};

/// Lower `spechls.task` to a child hw.module (+CRE) and replace with hw.instance.
struct TaskRewriter : OpRewritePattern<spechls::TaskOp> {
  using OpRewritePattern<spechls::TaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spechls::TaskOp task,
                                PatternRewriter &rewriter) const override {
    auto parentMod = task->getParentOfType<hw::HWModuleOp>();
    if (!parentMod)
      return rewriter.notifyMatchFailure(task, "parent is not hw.module");

    // Fetch parent's CRE ports to forward into the instance.
    auto [pClk, pRst, pCe] = getCREArgs(parentMod);
    if (!pClk || !pRst || !pCe)
      return rewriter.notifyMatchFailure(task, "parent missing clk/rst/ce");

    auto *ctx = rewriter.getContext();

    // Build child signature: functional inputs, then clk/rst/ce; one output.
    SmallVector<hw::PortInfo> inputs, outputs;
    for (auto it : llvm::enumerate(task.getArgs().getTypes()))
      inputs.push_back(makeInputPort(ctx, ("arg" + Twine(it.index())).str(), it.value()));

    inputs.push_back(makeInputPort(ctx, "clk", seq::ClockType::get(ctx)));
    inputs.push_back(makeInputPort(ctx, "rst", IntegerType::get(ctx, 1)));
    inputs.push_back(makeInputPort(ctx, "ce",  IntegerType::get(ctx, 1)));

    outputs.push_back(makeOutputPort(ctx, "result", task.getResult().getType()));


    hw::ModulePortInfo portInfo(inputs,outputs);

    // Create child module next to the parent (keeps symbol visibility simple).
    auto guard = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(parentMod);

    auto child = rewriter.create<hw::HWModuleOp>(task.getLoc(), task.getSymNameAttr(), portInfo);

    // Inline task body into child body; map only functional args.
    Block &src = task.getBody().front();
    Block *dst = child.getBodyBlock();
    Operation *dstTerm = dst->getTerminator();

    unsigned nFuncArgs = task.getArgs().size();
    SmallVector<Value> argMap;
    argMap.reserve(nFuncArgs);
    for (unsigned i = 0; i < nFuncArgs; ++i)
      argMap.push_back(dst->getArgument(i));

    rewriter.inlineBlockBefore(&src, dstTerm, argMap);

    // Wire commit -> output.
    Operation *maybeCommit = dstTerm->getPrevNode();
    auto commit = dyn_cast_or_null<spechls::CommitOp>(maybeCommit);
    if (!commit)
      return task.emitError("expected spechls.commit at end of task body"), failure();

    dstTerm->setOperands({commit.getValue()});
    // We keep the `hw.output` as module terminator, so just erase commit op.
    rewriter.eraseOp(maybeCommit);



    // Restore insertion point, build instance.
    rewriter.restoreInsertionPoint(guard);

    SmallVector<Value> instOperands(task.getArgs().begin(), task.getArgs().end());
    instOperands.push_back(pClk);
    instOperands.push_back(pRst);
    instOperands.push_back(pCe);

    SmallVector<Type> resTys;
    hw::ModulePortInfo childPorts(child.getPortList());
    for (auto &po : childPorts)
      resTys.push_back(po.type);

    // auto inst = rewriter.create<hw::InstanceOp>(
    //     task.getLoc(), resTys, FlatSymbolRefAttr::get(ctx, child.getName()),
    //     instOperands, /*parameters=*/ArrayAttr(), /*innerSym=*/StringAttr(),
    //     /*name=*/task.getSymNameAttr());

    auto inst = rewriter.create<hw::InstanceOp>(
    task.getLoc(),
    child,
    task.getSymName(),
    instOperands);

    SmallVector<mlir::Attribute> argNames, resNames;
    for (const hw::PortInfo &p : childPorts.getInputs())
      argNames.push_back(p.name ? p.name : rewriter.getStringAttr(""));
    for (const hw::PortInfo &p : childPorts.getOutputs())
      resNames.push_back(p.name ? p.name : rewriter.getStringAttr(""));
    inst->setAttr("argNames", mlir::ArrayAttr::get(rewriter.getContext(), argNames));
    inst->setAttr("resultNames", mlir::ArrayAttr::get(rewriter.getContext(), resNames));


    rewriter.replaceOp(task, inst.getResults());
    return success();
  }
};

/// The pass: anchored on a single `spechls::KernelOp`.
struct SpecHLSToHWPass
    : public spechls::impl::SpecHLSToHWPassBase<SpecHLSToHWPass> {
  void runOnOperation() override {
    auto kernel = getOperation(); // spechls::KernelOp
    auto *ctx = &getContext();

    LLVM_DEBUG(llvm::dbgs() << "[spechls-to-hw] Lowering kernel '"
                            << kernel.getSymName() << "' to hw.module...\n");


    // ---- Step 1: Create the parent hw.module with clk/rst/ce ----

    FunctionType fnTy = kernel.getFunctionType();

    SmallVector<hw::PortInfo> inputs, outputs;
    // Functional inputs
    for (auto it : llvm::enumerate(fnTy.getInputs()))
      inputs.push_back(makeInputPort(ctx, ("arg" + Twine(it.index())).str(), it.value()));

    // CRE inputs appended at the end (fixed order)
    inputs.push_back(makeInputPort(ctx, "clk", seq::ClockType::get(ctx)));
    inputs.push_back(makeInputPort(ctx, "rst", IntegerType::get(ctx, 1)));
    inputs.push_back(makeInputPort(ctx, "ce",  IntegerType::get(ctx, 1)));


    // Outputs (single or multiple)
    if (fnTy.getNumResults() == 1) {
      outputs.push_back(makeOutputPort(ctx, "result", fnTy.getResult(0)));
    } else {
      for (unsigned i = 0; i < fnTy.getNumResults(); ++i)
        outputs.push_back(makeOutputPort(ctx, ("result" + Twine(i)).str(), fnTy.getResult(i)));
    }

    hw::ModulePortInfo portInfo (inputs,outputs);

    OpBuilder b(kernel);
    auto hwModule = b.create<hw::HWModuleOp>(kernel.getLoc(), kernel.getSymNameAttr(), portInfo);

    // ---- Step 2: Inline kernel body into hw.module and hook outputs ----
    Block &src = kernel.getBody().front();
    Block *dst = hwModule.getBodyBlock();
    Operation *dstTerm = dst->getTerminator();

    unsigned nFuncArgs = fnTy.getNumInputs();
    SmallVector<Value> argMap;
    argMap.reserve(nFuncArgs);
    for (unsigned i = 0; i < nFuncArgs; ++i)
      argMap.push_back(dst->getArgument(i));

    // After inlining, the `spechls.exit` should be just before `hw.output`.
    mlir::IRRewriter rewriter(b);
    rewriter.setInsertionPoint(dstTerm);
    rewriter.inlineBlockBefore(&src, dstTerm, argMap);

    Operation *maybeExit = dstTerm->getPrevNode();
    auto exit = dyn_cast_or_null<spechls::ExitOp>(maybeExit);
    if (!exit) {
      kernel.emitError("expected spechls.exit at end of kernel body");
      signalPassFailure();
      return;
    }
    dstTerm->setOperands(exit.getValues());
    maybeExit->erase();

    // Remove the original kernel op.
    b.setInsertionPoint(hwModule);
    kernel->erase();

    // ---- Step 3: Lower all Task/Gamma inside this module ----
    // Run local greedy patterns on the module body.
    RewritePatternSet patterns(ctx);
    patterns.add<TaskRewriter, GammaRewriter>(ctx);

    GreedyRewriteConfig config;
    config.enableConstantCSE(true);
    config.enableFolding(true);

    if (failed(applyPatternsGreedily(hwModule, std::move(patterns), config))) {
      hwModule.emitError("greedy rewrite failed");
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "[spechls-to-hw] Done for module '"
                            << hwModule.getName() << "'\n");
  }
};

} // namespace

// // Pass f actory (declared in your Passes.h from the ODS td).
// std::unique_ptr<mlir::Pass> spechls::createSpecHLSToHWPass() {
//   return std::make_unique<SpecHLSToHWPass>();
// }
#endif // LEGACY_PASS