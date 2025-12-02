//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "CAPI/Dialect/Schedule.h"
#include "CAPI/Dialect/SpecHLS.h"
#include "CAPI/Dialect/Wcet.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep
#include "Dialect/Wcet/Transforms/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"

#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/Debug.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/LLHD.h>
#include <circt-c/Dialect/LTL.h>
#include <circt-c/Dialect/Moore.h>
#include <circt-c/Dialect/SSP.h>
#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/Seq.h>
#include <circt-c/Dialect/Sim.h>
#include <circt-c/Dialect/Synth.h>
#include <circt-c/Dialect/Verif.h>
#include <circt/Dialect/SSP/SSPDialect.h>
#include <circt/Dialect/SSP/SSPOps.h>
#include <mlir-c/Dialect/ControlFlow.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Math.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "CAPI/Heptane/Heptane.h"

extern "C" {

//===--------------------------------------------------------------------------------------------------------------===//
// Utility functions
//===--------------------------------------------------------------------------------------------------------------===//

// const char *getCStringDataFromMlirStringRef(MlirStringRef str) { return str.data; }
//
// size_t getCStringSizeFromMlirStringRef(MlirStringRef str) { return str.length; }
//
// const char *getCStringDataFromMlirIdentifier(MlirIdentifier identifier) {
//   MlirStringRef str = mlirIdentifierStr(identifier);
//   return str.data;
// }
//
// size_t getCStringSizeFromMlirIdentifier(MlirIdentifier identifier) {
//   MlirStringRef str = mlirIdentifierStr(identifier);
//   return str.length;
// }
//
// char getCharAt(const char *v, int offset) { return v[offset]; }
//
// MlirIdentifier mlirOperationGetAttributeNameAt(MlirOperation op, int64_t pos) {
//   return mlirOperationGetAttribute(op, pos).name;
// }
//
// MlirAttribute mlirOperationGetAttributeAt(MlirOperation op, int64_t pos) {
//   return mlirOperationGetAttribute(op, pos).attribute;
// }

//===--------------------------------------------------------------------------------------------------------------===//
// Passes
//===--------------------------------------------------------------------------------------------------------------===//

MlirModule parseMLIR(const char *str) {
  MlirContext context = mlirContextCreate();

  // Register required dialects.
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hw__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__comb__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__ssp__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__spechls__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__schedule__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__wcet__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__seq__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__synth__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sv__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llhd__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__cf__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__scf__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__math__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sim__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__verif__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__moore__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__func__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__ltl__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__debug__(), context);

  MlirStringRef wrapped = mlirStringRefCreateFromCString(str);
  return mlirModuleCreateParse(context, wrapped);
}

void destroyMLIR(MlirModule module) {
  MlirContext context = mlirModuleGetContext(module);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
}

#define DEFINE_WCET_API_PASS(func, namespace, name)                                                                    \
  MlirModule func(MlirModule mod) {                                                                                    \
    mlir::ModuleOp module = unwrap(mod);                                                                               \
    mlir::PassManager pm(module.getContext(), mlir::ModuleOp::getOperationName(),                                      \
                         mlir::PassManager::Nesting::Implicit);                                                        \
    pm.addPass(namespace ::create##name());                                                                            \
    if (mlir::failed(pm.run(module)))                                                                                  \
      llvm::errs() << "Unexpected failure running pass manager.\n";                                                    \
    return wrap(module);                                                                                               \
  }

void mlirDumpModule(MlirModule module) { unwrap(module)->dump(); }
size_t mlirWcetAnalysis(MlirModule module, mlir::SmallVector<size_t> &instrs) {
  //==== Setup module
  auto mod = unwrap(module);
  auto pm = mlir::PassManager::on<mlir::ModuleOp>(mod->getContext());
  pm.addPass(wcet::createSetupAnalysisPass());
  if (failed(pm.run(mod)))
    return 0;
  //==== Analyse each instructions
  for (size_t instr : instrs) {
    pm.clear();
    auto insertPass = wcet::createInsertInstrPass();
    if (mlir::failed(insertPass->initializeOptions("instrs=" + std::to_string(instr), [](const mlir::Twine &msg) {
          llvm::errs() << msg << "\n";
          return mlir::failure();
        }))) {
      return 0;
    }
    pm.addPass(std::move(insertPass));
    pm.addPass(wcet::createInlineCorePass());
    pm.addPass(wcet::createLongestPathPass());
    if (failed(pm.run(mod))) {
      return 0;
    }
  }
  //==== Retrieve the Wcet
  size_t wcet = 0;
  mod->walk([&](mlir::Operation *op) {
    if (op->hasAttr("wcet.penalties")) {
      wcet += mlir::cast<mlir::IntegerAttr>(op->getAttr("wcet.penalties")).getInt();
    }
  });

  //==== Clean up
  mlir::Operation *analysisCore;
  mod->walk([&](mlir::Operation *c) {
    if (c->hasAttr("wcet.analysis")) {
      analysisCore = c;
      return;
    }
  });
  analysisCore->erase();

  return wcet;
}
}
