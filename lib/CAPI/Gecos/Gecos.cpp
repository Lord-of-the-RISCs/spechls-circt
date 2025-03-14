//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "CAPI/Dialect/Schedule.h"
#include "CAPI/Dialect/SpecHLS.h"
#include "Dialect/Schedule/Transforms/Passes.h"

#include <circt-c/Dialect/SSP.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>

extern "C" {

//===--------------------------------------------------------------------------------------------------------------===//
// Utility functions
//===--------------------------------------------------------------------------------------------------------------===//

const char *getCStringDataFromMlirStringRef(MlirStringRef str) { return str.data; }

size_t getCStringSizeFromMlirStringRef(MlirStringRef str) { return str.length; }

const char *getCStringDataFromMlirIdentifier(MlirIdentifier identifier) {
  MlirStringRef str = mlirIdentifierStr(identifier);
  return str.data;
}

size_t getCStringSizeFromMlirIdentifier(MlirIdentifier identifier) {
  MlirStringRef str = mlirIdentifierStr(identifier);
  return str.length;
}

char getCharAt(const char *v, int offset) { return v[offset]; }

MlirIdentifier mlirOperationGetAttributeNameAt(MlirOperation op, int64_t pos) {
  return mlirOperationGetAttribute(op, pos).name;
}

MlirAttribute mlirOperationGetAttributeAt(MlirOperation op, int64_t pos) {
  return mlirOperationGetAttribute(op, pos).attribute;
}

//===--------------------------------------------------------------------------------------------------------------===//
// Passes
//===--------------------------------------------------------------------------------------------------------------===//

MlirModule parseMLIR(const char *str) {
  MlirContext context = mlirContextCreate();

  // Register required dialects.
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__ssp__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__spechls__(), context);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__schedule__(), context);

  MlirStringRef wrapped = mlirStringRefCreateFromCString(str);
  return mlirModuleCreateParse(context, wrapped);
}

void destroyMLIR(MlirModule module) {
  MlirContext context = mlirModuleGetContext(module);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
}

#define DEFINE_GECOS_API_PASS(func, namespace, name)                                                                   \
  MlirModule func(MlirModule mod) {                                                                                    \
    mlir::ModuleOp module = unwrap(mod);                                                                               \
    mlir::PassManager pm(module.getContext(), mlir::ModuleOp::getOperationName(),                                      \
                         mlir::PassManager::Nesting::Implicit);                                                        \
    pm.addPass(namespace ::create##name());                                                                            \
    if (mlir::failed(pm.run(module)))                                                                                  \
      llvm::errs() << "Unexpected failure running pass manager.\n";                                                    \
    return wrap(module);                                                                                               \
  }

DEFINE_GECOS_API_PASS(configurationExcluderMLIR, schedule, ConfigurationExcluderPass)
DEFINE_GECOS_API_PASS(mobilityMLIR, schedule, MobilityPass)
DEFINE_GECOS_API_PASS(scheduleMLIR, schedule, SchedulePass)
}
