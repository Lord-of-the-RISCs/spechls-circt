//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Wcet/IR/WcetOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>

#include "Dialect/Wcet/IR/WcetDialect.cpp.inc"

using namespace mlir;

//===--------------------------------------------------------------------------------------------------------------===//
// Wcet dialect
//===--------------------------------------------------------------------------------------------------------------===//

//void wcet::WcetDialect::initialize() {
//  addOperations<
//#define GET_OP_LIST
//#include "Dialect/Wcet/IR/Wcet.cpp.inc"
//      >();
//  addTypes<
//#define GET_TYPEDEF_LIST
//#include "Dialect/Wcet/IR/WcetTypes.cpp.inc"
//      >();
//}
//
//LogicalResult wcet::OperationOp::verify() {
//  if (getNumOperands() != getDistances().size())
//    return emitOpError("expects ") << getNumOperands() << " distances, but got " << getDistances().size();
//  return success();
//}

//===--------------------------------------------------------------------------------------------------------------===//
// TableGen'd types and op method definitions
//===--------------------------------------------------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Dialect/Wcet/IR/WcetTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Wcet/IR/Wcet.cpp.inc"
