//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Conversion/SpecHLS/Passes.h"
#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Outlining.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <circt/Conversion/ExportVerilog.h>
#include <circt/Conversion/ImportVerilog.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Debug/DebugDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HW/HWTypes.h>
#include <circt/Dialect/LLHD/IR/LLHDDialect.h>
#include <circt/Dialect/LTL/LTLDialect.h>
#include <circt/Dialect/Moore/MooreDialect.h>
#include <circt/Dialect/SV/SVDialect.h>
#include <circt/Dialect/Sim/SimDialect.h>
#include <circt/Dialect/Verif/VerifDialect.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Inliner.h>
#include <mlir/Transforms/InliningUtils.h>
#include <string>

#define _YOSYS_
#include <kernel/log.h>
#include <kernel/register.h>
#include <kernel/yosys.h>
#undef _YOSYS_

#include <memory>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_OPTIMIZECONTROLLOGICPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

class OptimizeControlLogicPass : public spechls::impl::OptimizeControlLogicPassBase<OptimizeControlLogicPass> {
public:
  using OptimizeControlLogicPassBase::OptimizeControlLogicPassBase;
  llvm::DenseSet<Operation *> sliceControl(Operation *firstOp);

  void runOnOperation() override;
};

} // namespace

llvm::DenseSet<Operation *> OptimizeControlLogicPass::sliceControl(Operation *firstOp) {
  auto isValidForMerging = [](Operation *op) {
    return !(isa<spechls::MuOp>(op) || isa<spechls::AlphaOp>(op) || isa<spechls::DelayOp>(op) ||
             isa<spechls::LoadOp>(op) || isa<spechls::CallOp>(op) || isa<spechls::GammaOp>(op));
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
        workingList.push_back(operand.getDefiningOp());
      }
    }
  }
  return result;
}

void OptimizeControlLogicPass::runOnOperation() {
  auto operation = getOperation();
  auto *ctx = &getContext();
  IRRewriter builder(ctx);

  operation.walk([&](spechls::KernelOp kernel) {
    auto spechlsToHwPm = PassManager::on<spechls::TaskOp>(ctx);
    spechlsToHwPm.addPass(spechls::createSpecHLSTaskToHWPass());

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
      //    builder.insert(task);
      if (failed(spechlsToHwPm.run(task))) {
        return signalPassFailure();
      }
    };

    // Outline gamma control as HWModule
    kernel.walk([&](spechls::GammaOp gamma) { simplifyControl(gamma, 0); });

    auto lowerMoorePm = PassManager::on<ModuleOp>(ctx);
    lowerMoorePm.addPass(circt::createConvertMooreToCorePass());
    lowerMoorePm.addNestedPass<::circt::hw::HWModuleOp>(circt::llhd::createSig2Reg());
    lowerMoorePm.addPass(mlir::createCanonicalizerPass());

    // Simplify HWModules
    auto newModuleOp = builder.create<ModuleOp>(builder.getUnknownLoc());
    auto *newModuleBody = newModuleOp.getBody();
    Yosys::yosys_setup();
    operation.walk([&](circt::hw::HWModuleOp hw) { builder.moveOpBefore(hw, newModuleBody, newModuleBody->end()); });

    auto generateVerilogPm = PassManager::on<ModuleOp>(ctx);
    std::string verilog;
    auto os = std::unique_ptr<llvm::raw_ostream>(new llvm::raw_string_ostream(verilog));
    generateVerilogPm.addPass(circt::createExportVerilogPass(std::move(os)));
    if (failed(generateVerilogPm.run(newModuleOp)))
      return signalPassFailure();

    Yosys::log_streams.clear();
    Yosys::log_error_stderr = true;

    auto *design = new Yosys::Design;
    std::istringstream inputStream(verilog);
    Yosys::Frontend::frontend_call(design, &inputStream, "", "verilog -sv");

    std::string yosysAbc = "";
    yosysAbc += YOSYS_PATH;
    yosysAbc += "/yosys-abc";
    Yosys::yosys_abc_executable = yosysAbc;

    Yosys::Pass::call(design, "proc");
    Yosys::Pass::call(design, "flatten");
    Yosys::Pass::call(design, "opt -full");
    Yosys::Pass::call(design, "synth");
    Yosys::Pass::call(design, "abc -g AND,OR,XOR");
    std::ostringstream outputStream;

    Yosys::Backend::backend_call(design, &outputStream, "", "verilog -sv");

    // regenerated verilog is in outputStream.str()
    llvm::SourceMgr mgr;
    std::string newVer = outputStream.str();

    std::ofstream file("/tmp/ouput.sv");
    file << newVer;
    file.close();

    mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(newVer), SMLoc());
    auto simplifiedModule = builder.create<ModuleOp>(builder.getUnknownLoc());
    mlir::TimingScope timingScope;
    if (failed(circt::importVerilog(mgr, ctx, timingScope, simplifiedModule)))
      return signalPassFailure();

    if (failed(lowerMoorePm.run(simplifiedModule)))
      return signalPassFailure();

    simplifiedModule.walk(
        [&](circt::hw::HWModuleOp mod) { builder.moveOpBefore(mod, operation.getBody(), operation.getBody()->end()); });

    builder.eraseBlock(newModuleOp.getBody());
    builder.eraseOp(newModuleOp);
    builder.eraseBlock(simplifiedModule.getBody());
    builder.eraseOp(simplifiedModule);

    Yosys::yosys_shutdown();

    llvm::SmallVector<Operation *> toMove, toDelete;

    //  Inline HWModules as SpecHLSTask
    kernel->walk([&](circt::hw::InstanceOp instance) {
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
          builder.replaceAllOpUsesWith(instance, output.getOperand(0).getDefiningOp());
        }
      }

      body.walk([&](Operation *op) {
        if (llvm::dyn_cast<circt::hw::OutputOp>(op))
          toDelete.push_back(op);
        else
          toMove.push_back(op);
      });

      toDelete.push_back(module);
      toDelete.push_back(instance);
    });
    auto *kernelBody = &kernel.getBody().front();
    for (auto &&op : toMove)
      builder.moveOpAfter(op, kernelBody, kernelBody->begin());
    for (auto &&op : toDelete)
      builder.eraseOp(op);
  });
}