//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Outlining.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "Dialect/SpecHLS/Transforms/TopologicalSort.h"
#include "Dialect/SpecHLS/Transforms/YosysSetup.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

#include <circt/Conversion/CombToSynth.h>
#include <circt/Conversion/ExportVerilog.h>
#include <circt/Conversion/ImportVerilog.h>
#include <circt/Conversion/MooreToCore.h>
#include <circt/Conversion/SynthToComb.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Debug/DebugDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HW/HWPasses.h>
#include <circt/Dialect/HW/HWTypes.h>
#include <circt/Dialect/LLHD/IR/LLHDDialect.h>
#include <circt/Dialect/LLHD/Transforms/LLHDPasses.h>
#include <circt/Dialect/LTL/LTLDialect.h>
#include <circt/Dialect/Moore/MooreDialect.h>
#include <circt/Dialect/Moore/MooreOps.h>
#include <circt/Dialect/Moore/MoorePasses.h>
#include <circt/Dialect/SV/SVDialect.h>
#include <circt/Dialect/Sim/SimDialect.h>
#include <circt/Dialect/Synth/SynthDialect.h>
#include <circt/Dialect/Synth/SynthOps.h>
#include <circt/Dialect/Synth/Transforms/SynthPasses.h>
#include <circt/Dialect/Verif/VerifDialect.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/UniqueVector.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Inliner.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/Transforms/Passes.h>
#include <string>

#define _YOSYS_
// push because yosys generate warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <kernel/log.h>
#include <kernel/register.h>
#include <kernel/yosys.h>
#pragma GCC diagnostic pop
#undef _YOSYS_

#include <memory>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_OPTIMIZECONTROLLOGICPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

void inlineControlTasks(spechls::KernelOp kernel, mlir::IRRewriter &rewriter) {
  kernel.walk([&](spechls::TaskOp task) {
    if (task.getSymName().starts_with("ctrl_")) {
      auto *block = task->getBlock();
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointAfter(task);
      auto pack = spechls::PackOp::create(rewriter, rewriter.getUnknownLoc(), task.getResult().getType(),
                                          task.getBodyBlock()->getTerminator()->getOperands());
      task.getResult().replaceAllUsesWith(pack.getResult());
      rewriter.eraseOp(task.getBodyBlock()->getTerminator());
      rewriter.inlineBlockBefore(task.getBodyBlock(), task, task.getArgs());
      rewriter.eraseOp(task);
      rewriter.restoreInsertionPoint(ip);
      mlir::sortTopologically(block, spechls::topologicalSortCriterion);
    }
  });
}

class OptimizeControlLogicPass : public spechls::impl::OptimizeControlLogicPassBase<OptimizeControlLogicPass> {
public:
  using OptimizeControlLogicPassBase::OptimizeControlLogicPassBase;
  llvm::DenseSet<Operation *> sliceControl(Operation *firstOp);

  void runOnOperation() override;
};

} // namespace

llvm::DenseSet<Operation *> OptimizeControlLogicPass::sliceControl(Operation *firstOp) {
  auto isValidForMerging = [](Operation *op) {
    return !(llvm::isa<spechls::MuOp>(op) || llvm::isa<spechls::AlphaOp>(op) || llvm::isa<spechls::DelayOp>(op) ||
             llvm::isa<spechls::LoadOp>(op) || llvm::isa<spechls::CallOp>(op) || llvm::isa<spechls::FieldOp>(op) ||
             llvm::isa<spechls::CancellableDelayOp>(op) || llvm::isa<spechls::RollbackableDelayOp>(op));
  };
  llvm::DenseSet<Operation *> result;
  llvm::SmallVector<Operation *> workingList;
  workingList.push_back(firstOp);
  while (!workingList.empty()) {
    auto *current = workingList.back();
    workingList.pop_back();
    if (isValidForMerging(current)) {
      result.insert(current);
      for (auto operand : current->getOperands()) {
        auto *nextOp = operand.getDefiningOp();
        if (nextOp && !result.contains(nextOp))
          workingList.push_back(nextOp);
      }
    }
  }
  return result;
}

void OptimizeControlLogicPass::runOnOperation() {
  spechls::setupYosys();
  auto operation = getOperation();
  auto *ctx = operation->getContext();
  IRRewriter builder(ctx);

  operation.walk([&](spechls::KernelOp kernel) {
    unsigned index = 0;

    auto simplifyControl = [&](Operation *op, int input) {
      auto controlValue = op->getOperand(input);
      builder.setInsertionPointAfterValue(controlValue);
      auto controlType = llvm::dyn_cast<IntegerType>(controlValue.getType());
      auto signlessControlType = builder.getIntegerType(controlType.getWidth());
      auto inCast = builder.create<circt::hw::BitcastOp>(controlValue.getLoc(), signlessControlType, controlValue);
      auto outCast = builder.create<circt::hw::BitcastOp>(controlValue.getLoc(), controlType, inCast.getResult());
      auto outlineSet = sliceControl(inCast);
      op->setOperand(input, outCast.getResult());
      auto task = spechls::outlineControl(builder, inCast->getLoc(), "outline_control_" + std::to_string(index++),
                                          outlineSet, inCast.getResult());

      auto spechlsToHwPm = PassManager::on<spechls::KernelOp>(ctx);
      auto taskToHWPass = spechls::createSpecHLSTaskToHWPass();
      if (failed(taskToHWPass->initializeOptions("targetTask=" + std::string(task.getSymName()), [](const Twine &msg) {
            llvm::errs() << msg << '\n';
            return failure();
          })))
        return signalPassFailure();
      spechlsToHwPm.addPass(std::move(taskToHWPass));

      if (failed(spechlsToHwPm.run(kernel))) {
        return signalPassFailure();
      }
      // std::string funName = "outline_control_" + std::to_string(index++);
      // spechls::outlineOptFunc(builder, inCast->getLoc(), funName, outlineSet, inCast.getResult());
    };
    // Outline gamma control as HWModule
    kernel.walk([&](spechls::GammaOp gamma) { simplifyControl(gamma, 0); });
    kernel.walk([&](spechls::AlphaOp alpha) { simplifyControl(alpha, 3); });
    kernel.walk([&](spechls::PrintOp print) { simplifyControl(print, 1); });

    // auto spechlsToHwPm = PassManager::on<spechls::KernelOp>(ctx);
    // spechlsToHwPm.addPass(mlir::createCanonicalizerPass());
    // spechlsToHwPm.addPass(mlir::createCSEPass());
    // spechlsToHwPm.addNestedPass<spechls::OptimizedFuncOp>(spechls::createInlineRecursiveOptFuncPass());
    // spechlsToHwPm.addPass(spechls::createSpecHLSOptFunToHWPass());
    // if (failed(spechlsToHwPm.run(kernel))) {
    //   return signalPassFailure();
    // }

    std::string abcPath = YOSYS_PATH "/yosys-abc";

    auto lowerMoorePm = PassManager::on<ModuleOp>(ctx);
    lowerMoorePm.addNestedPass<circt::moore::SVModuleOp>(circt::moore::createLowerConcatRefPass());
    lowerMoorePm.addNestedPass<circt::moore::SVModuleOp>(circt::moore::createSimplifyProceduresPass());
    lowerMoorePm.addPass(mlir::createCanonicalizerPass());
    lowerMoorePm.addPass(circt::createConvertMooreToCorePass());
    lowerMoorePm.addNestedPass<::circt::hw::HWModuleOp>(circt::llhd::createSig2Reg());
    lowerMoorePm.addPass(mlir::createCanonicalizerPass());
    lowerMoorePm.addPass(spechls::createSimplifyCombPass());
    lowerMoorePm.addPass(mlir::createCanonicalizerPass());
    lowerMoorePm.addPass(mlir::createCSEPass());

    // Simplify HWModules
    auto newModuleOp = builder.create<ModuleOp>(builder.getUnknownLoc());
    auto *newModuleBody = newModuleOp.getBody();

    operation.walk([&](circt::hw::HWModuleOp hw) { builder.moveOpBefore(hw, newModuleBody, newModuleBody->end()); });

    auto generateVerilogPm = PassManager::on<ModuleOp>(ctx);
    std::string verilog;
    auto os = std::unique_ptr<llvm::raw_ostream>(new llvm::raw_string_ostream(verilog));
    generateVerilogPm.addPass(circt::createExportVerilogPass(std::move(os)));
    if (failed(generateVerilogPm.run(newModuleOp)))
      return signalPassFailure();

    Yosys::log_streams.clear();
    Yosys::log_error_stderr = true;

    auto design = std::make_unique<Yosys::Design>();
    std::istringstream inputStream(verilog);
    Yosys::Frontend::frontend_call(design.get(), &inputStream, "", "verilog -sv");

    Yosys::yosys_abc_executable = YOSYS_PATH "/yosys-abc";

    Yosys::Pass::call(design.get(), "proc");
    Yosys::Pass::call(design.get(), "flatten");
    Yosys::Pass::call(design.get(), "opt -full");
    Yosys::Pass::call(design.get(), "synth");
    Yosys::Pass::call(design.get(), "abc -g AND,OR,XOR");
    Yosys::Pass::call(design.get(), "opt_clean -purge");
    std::ostringstream outputStream;

    Yosys::Backend::backend_call(design.get(), &outputStream, "", "verilog -sv");

    // regenerated verilog is in outputStream.str()
    llvm::SourceMgr mgr;
    std::string newVer = outputStream.str();
    mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(newVer), SMLoc());
    auto simplifiedModule = builder.create<ModuleOp>(builder.getUnknownLoc());
    mlir::TimingScope timingScope;
    if (failed(circt::importVerilog(mgr, ctx, timingScope, simplifiedModule)))
      return signalPassFailure();
    if (failed(lowerMoorePm.run(simplifiedModule)))
      return signalPassFailure();

    llvm::SmallVector<circt::hw::HWModuleOp> hwModToMove;

    simplifiedModule.walk([&](circt::hw::HWModuleOp mod) { hwModToMove.push_back(mod); });
    for (auto &mod : hwModToMove) {
      builder.moveOpBefore(mod, operation.getBody(), operation.getBody()->end());
    }

    // builder.eraseBlock(newModuleOp.getBody());
    builder.eraseOp(newModuleOp);
    // builder.eraseBlock(simplifiedModule.getBody());
    builder.eraseOp(simplifiedModule);

    llvm::DenseSet<Operation *> toDelete;
    llvm::DenseMap<Operation *, Block *> toMove;

    llvm::DenseMap<llvm::SmallVector<Operation *>, Block *> toOutline;

    //  Inline HWModules as SpecHLSTask
    kernel->walk([&](circt::hw::InstanceOp instance) {
      llvm::SmallVector<Operation *> nodes;
      builder.setInsertionPoint(instance);
      auto args = instance.getInputs();
      auto module = llvm::dyn_cast<circt::hw::HWModuleOp>(operation.lookupSymbol(instance.getReferencedModuleName()));
      auto &body = module.getBody().front();
      auto moduleInputs = body.getArguments();

      // rewire input and output.
      for (auto &&op : body) {
        for (size_t i = 0; i < op.getNumOperands(); ++i) {
          for (size_t j = 0; j < moduleInputs.size(); ++j) {
            if (op.getOperand(i) == moduleInputs[j]) {
              op.setOperand(i, args[j]);
            }
          }
        }

        if (auto output = llvm::dyn_cast<circt::hw::OutputOp>(op)) {
          for (unsigned i = 0; i < instance->getNumResults(); ++i) {
            auto result = instance.getResult(i);
            auto out = output.getOperand(i);
            builder.replaceAllUsesWith(result, out);
          }
        }
      }

      body.walk([&](Operation *op) {
        if (!llvm::isa<circt::hw::OutputOp>(op)) {
          //   toDelete.insert(op);
          // } else {
          nodes.push_back(op);
          toMove.try_emplace(op, instance->getBlock());
        }
      });

      toDelete.insert(module);
      toDelete.insert(instance);
      toOutline.try_emplace(nodes, instance->getBlock());
    });
    for (auto [op, block] : toMove)
      builder.moveOpAfter(op, block, block->begin());
    for (auto &&op : toDelete) {
      builder.eraseOp(op);
    }

    unsigned idx = 0;
    for (auto [nodes, block] : toOutline) {
      if (!nodes.empty()) {
        auto ops = SmallPtrSet<mlir::Operation *, 32>();
        ops.insert(nodes.begin(), nodes.end());
        builder.setInsertionPointToStart(block);
        auto task = spechls::outlineTask(builder, builder.getUnknownLoc(), "ctrl_" + std::to_string(idx++), ops);
        task->setAttr("spechls.functionTask", builder.getUnitAttr());
      }
    }

    // lower bitcast to array type.
    llvm::SmallVector<circt::hw::BitcastOp> toChange;
    kernel.walk([&](circt::hw::BitcastOp bitcast) {
      if (auto aType = llvm::dyn_cast<circt::hw::ArrayType>(bitcast.getResult().getType())) {
        if (llvm::isa<circt::hw::ConstantOp>(bitcast.getInput().getDefiningOp())) {
          toChange.push_back(bitcast);
        }
      }
    });

    kernel.walk([&](mlir::Operation *op) {
      llvm::SmallVector<mlir::StringAttr> toRemove;
      for (auto attr : op->getAttrs()) {
        if (attr.getName().strref().starts_with("sv."))
          toRemove.push_back(attr.getName());
      }
      for (auto &attr : toRemove)
        op->removeAttr(attr);
    });

    for (auto bitcast : toChange) {
      builder.setInsertionPoint(bitcast);
      if (auto aType = llvm::dyn_cast<circt::hw::ArrayType>(bitcast.getResult().getType())) {
        if (auto cst = llvm::dyn_cast<circt::hw::ConstantOp>(bitcast.getInput().getDefiningOp())) {
          auto value = cst.getValue();
          size_t numElt = aType.getNumElements();
          llvm::SmallVector<mlir::Value> newInputs;
          newInputs.reserve(numElt);
          int eltBw = aType.getElementType().getIntOrFloatBitWidth();
          for (size_t i = 0; i < numElt; ++i) {
            auto newVal = value.extractBits(eltBw, (numElt - i - 1) * eltBw);
            newInputs.push_back(builder.create<circt::hw::ConstantOp>(cst.getLoc(), newVal));
          }
          builder.replaceOpWithNewOp<circt::hw::ArrayCreateOp>(bitcast, newInputs);
        }
      }
    }

    inlineControlTasks(kernel, builder);
  });
}