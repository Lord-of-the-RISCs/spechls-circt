//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Analysis/Wcet/GraphWcetAnalysis.h"
#include "Analysis/Wcet/LinearWcetAnalysis.h"
#include "CAPI/Dialect/Schedule.h"
#include "CAPI/Dialect/SpecHLS.h"
#include "CAPI/Dialect/Wcet.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include <circt/Dialect/HW/HWOps.h>
#include <cstddef>
#include <kernel/rtlil.h>
#include <llvm/Support/Format.h>

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

using namespace mlir;

extern "C" {

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

size_t mlirWcetLongAnalysis(MlirModule module, mlir::SmallVector<size_t> &instr) {
  auto mod = unwrap(module);
  return wcet::GraphAnalysis(mod, instr).wcet;
}

size_t mlirWcetAnalysis(MlirModule module, mlir::SmallVector<size_t> &instrs) {
  auto mod = unwrap(module);
  return wcet::LinearAnalysis(mod, instrs).wcet;
}
}
