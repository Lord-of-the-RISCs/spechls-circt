//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/ScheduleDialect/ScheduleDialectDialect.h"
#include "Dialect/ScheduleDialect/ScheduleDialectOps.h"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

// #include "circt/Dialect/Comb/CombOps.h"
// #include "circt/Dialect/HW/HWOpInterfaces.h"
// #include "circt/Dialect/HW/HWOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Transforms/DialectConversion.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/ScheduleDialect/ScheduleDialectOps.cpp.inc"
#include "Dialect/SpecHLS/SpecHLSOpsDialect.cpp.inc"
#include "mlir/CAPI/Registration.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

extern "C" {

// bool mlirAttributeIsAArray(MlirAttribute attr) {
//   return llvm::isa<mlir::ArrayAttr>(unwrap(attr));
// }

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SpecHLS, spechls, SpecHLS::SpecHLSDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ScheduleDialect, scheduledialect,
                                      SpecHLS::ScheduleDialectDialect)
}

// void registerSeqPasses() { SpecHLS::registerTransformsPasses(); }
