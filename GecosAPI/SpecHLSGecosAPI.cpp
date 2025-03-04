//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/MemRef.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/IR.h>

#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/FSM.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/HWArith.h>
#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/Seq.h>

#include <CAPI/SpecHLS.h>

#include <CAPI/SSP.h>

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

#define BUFFERLEN 0x800000
char buffer[BUFFERLEN];
int offset = 0;

void resetBuffer() { offset = 0; }

void printToBuffer(MlirStringRef str, void *userData) {
  memcpy(&buffer[offset], str.data, str.length);
  offset += str.length;
}

} // namespace

extern "C" {

void registerAllUpstreamDialects(MlirContext ctx) {
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__spechls__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__schedule__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__transform__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__func__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__arith__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__comb__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__seq__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hw__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sv__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hwarith__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__scf__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__ssp__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__memref__(), ctx);
}

int getCStringSizeFromMlirIdentifier(MlirIdentifier ident) {
  MlirStringRef identStr = mlirIdentifierStr(ident);
  return identStr.length;
}

const char *getCStringDataFromMlirIdentifier(MlirIdentifier ident) {
  MlirStringRef identStr = mlirIdentifierStr(ident);
  return identStr.data;
}

const char *getCStringDataFromMlirStringRef(MlirStringRef ident) {
  return ident.data;
}

size_t getCStringSizeFromMlirStringRef(MlirStringRef ident) {
  return ident.length;
}

MlirIdentifier mlirNamedAttributeGetName(MlirNamedAttribute p) {
  return p.name;
}

MlirAttribute mlirNamedAttributeGetAttribute(MlirNamedAttribute p) {
  return p.attribute;
}

char *mlirBlockToString(MlirBlock b) {
  resetBuffer();
  mlirBlockPrint(b, printToBuffer, NULL);
  return buffer;
}

char *mlirOperationToString(MlirOperation b) {
  resetBuffer();
  mlirOperationPrint(b, printToBuffer, NULL);
  buffer[offset] = '\0';
  return buffer;
}

char *mlirModuleToString(MlirModule b) {
  resetBuffer();
  mlirOperationToString(mlirModuleGetOperation(b));
  buffer[offset] = '\0';
  return buffer;
}

char *mlirValueToString(MlirValue b) {
  resetBuffer();
  mlirValuePrint(b, printToBuffer, NULL);
  buffer[offset] = '\0';
  return buffer;
}

char *mlirAttributeToString(MlirAttribute b) {
  resetBuffer();
  if (b.ptr != NULL) {
    mlirAttributePrint(b, printToBuffer, NULL);
    buffer[offset] = '\0';
    return buffer;
  }
  return nullptr;
}

MlirIdentifier mlirOperationGetAttributeNameAt(MlirOperation op, intptr_t pos) {
  return mlirOperationGetAttribute(op, pos).name;
}

MlirAttribute mlirOperationGetAttributeAt(MlirOperation op, intptr_t pos) {

  return mlirOperationGetAttribute(op, pos).attribute;
}

void mlirPrintNamedAttribute(MlirNamedAttribute b) {
  fprintf(stderr, "%s->%s", getCStringDataFromMlirIdentifier(b.name),
          mlirAttributeToString(b.attribute));
}

char *mlirTypeToString(MlirType b) {
  resetBuffer();
  mlirTypePrint(b, printToBuffer, NULL);
  buffer[offset] = '\0';
  return buffer;
}

MlirModule parseMLIR(const char *mlir) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);
  MlirStringRef str = mlirStringRefCreateFromCString(mlir);
  MlirModule module = mlirModuleCreateParse(ctx, str);
  return module;
}

void destroyMLIR(MlirModule module) {
  MlirContext ctx = mlirModuleGetContext(module);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}
}
