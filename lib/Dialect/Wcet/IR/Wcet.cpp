//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Wcet/IR/WcetOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>

#include "Dialect/Wcet/IR/WcetDialect.cpp.inc"

using namespace mlir;

//===--------------------------------------------------------------------------------------------------------------===//
// Wcet dialect
//===--------------------------------------------------------------------------------------------------------------===//

void wcet::WcetDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Wcet/IR/Wcet.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Wcet/IR/WcetTypes.cpp.inc"
      >();
}
//
//LogicalResult wcet::OperationOp::verify() {
//  if (getNumOperands() != getDistances().size())
//    return emitOpError("expects ") << getNumOperands() << " distances, but got " << getDistances().size();
//  return success();
//}

//===--------------------------------------------------------------------------------------------------------------===//
// TableGen'd types and op method definitions
//===--------------------------------------------------------------------------------------------------------------===//

ParseResult wcet::PenaltyOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand input, enable, init;
  Type type;
  uint32_t delay;

  if (parser.parseOperand(input) || parser.parseKeyword("by") || parser.parseInteger(delay))
    return failure();
  result.addAttribute(getDepthAttrName(result.name), builder.getUI32IntegerAttr(delay));

  bool hasEnable = false;
  if (parser.parseOptionalKeyword("if").succeeded()) {
    if (parser.parseOperand(enable))
      return failure();
    hasEnable = true;
  }
  bool hasInit = false;
  if (parser.parseOptionalKeyword("init").succeeded()) {
    if (parser.parseOperand(init))
      return failure();
    hasInit = true;
  }
  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      builder.getDenseI32ArrayAttr({1, hasEnable, hasInit}));

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type) ||
      parser.resolveOperand(input, type, result.operands) || parser.addTypeToList(type, result.types))
    return failure();
  if (hasEnable && parser.resolveOperand(enable, builder.getI1Type(), result.operands))
    return failure();
  if (hasInit && parser.resolveOperand(init, type, result.operands))
    return failure();

  return success();
}

void wcet::PenaltyOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput() << " by " << getDepth();
  if (getEnable())
    printer << " if " << getEnable();
  if (getInit())
    printer << " init " << getInit();
  printer << " : " << getType();
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/Wcet/IR/WcetTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Wcet/IR/Wcet.cpp.inc"
