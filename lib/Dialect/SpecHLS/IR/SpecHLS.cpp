//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>

#include "Dialect/SpecHLS/IR/SpecHLSDialect.cpp.inc"

using namespace mlir;

//===--------------------------------------------------------------------------------------------------------------===//
// SpecHLS dialect
//===--------------------------------------------------------------------------------------------------------------===//

void spechls::SpecHLSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/SpecHLS/IR/SpecHLS.cpp.inc"
      >();
}

//===--------------------------------------------------------------------------------------------------------------===//
// Operations
//===--------------------------------------------------------------------------------------------------------------===//

namespace {

template <typename T>
ParseResult parseTaskLikeOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, T::getSymNameAttrName(result.name), result.attributes))
    return failure();

  // Parse the signature.
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignatureWithArguments(parser, false, entryArgs, isVariadic, resultTypes,
                                                                   resultAttrs))
    return failure();

  SmallVector<Type> argTypes;
  for (auto const &arg : entryArgs)
    argTypes.push_back(arg.type);
  result.addAttribute(T::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(builder.getFunctionType(argTypes, resultTypes)));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Add attributes to the arguments and results.
  assert(resultAttrs.size() == resultTypes.size());
  call_interface_impl::addArgAndResultAttrs(builder, result, entryArgs, resultAttrs,
                                            T::getArgAttrsAttrName(result.name), T::getResAttrsAttrName(result.name));

  // Parse the kernel body.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, entryArgs);
}

template <typename T>
void printTaskLikeOp(T &instance, OpAsmPrinter &printer) {
  // Print the kernel signature and attributes.
  printer << ' ';
  printer.printSymbolName(instance.getName());
  auto functionType = instance.getFunctionType();
  function_interface_impl::printFunctionSignature(printer, instance, functionType.getInputs(), false,
                                                  functionType.getResults());
  function_interface_impl::printFunctionAttributes(
      printer, instance,
      {instance.getFunctionTypeAttrName(), instance.getArgAttrsAttrName(), instance.getResAttrsAttrName()});

  // Print the kernel body.
  printer << ' ';
  printer.printRegion(instance.getBody(), false, true);
}

} // namespace

ParseResult spechls::HKernelOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTaskLikeOp<HKernelOp>(parser, result);
}

void spechls::HKernelOp::print(OpAsmPrinter &printer) { printTaskLikeOp(*this, printer); }

LogicalResult spechls::ExitOp::verify() {
  auto task = cast<HKernelOp>((*this)->getParentOp());

  // The number of committed values must match the task signature.
  auto const &results = task.getResultTypes();
  if (getNumOperands() != results.size())
    return emitOpError("has ") << getNumOperands() << " operands, but enclosing hkernel (@" << task.getName()
                               << ") returns " << results.size();

  for (size_t i = 0, e = results.size(); i != e; ++i) {
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of exit operand " << i << " (" << getOperand(i).getType()
                         << ") doesn't match result type (" << results[i] << ") in hkernel @" << task.getName();
  }

  return success();
}

ParseResult spechls::HTaskOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTaskLikeOp<HTaskOp>(parser, result);
}

void spechls::HTaskOp::print(OpAsmPrinter &printer) { printTaskLikeOp(*this, printer); }

LogicalResult spechls::CommitOp::verify() {
  auto task = cast<HTaskOp>((*this)->getParentOp());

  // The number of committed values must match the task signature.
  auto const &results = task.getResultTypes();
  if (getNumOperands() != results.size())
    return emitOpError("has ") << getNumOperands() << " operands, but enclosing htask (@" << task.getName()
                               << ") returns " << results.size();

  for (size_t i = 0, e = results.size(); i != e; ++i) {
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of commit operand " << i << " (" << getOperand(i).getType()
                         << ") doesn't match result type (" << results[i] << ") in htask @" << task.getName();
  }

  return success();
}

CallInterfaceCallable spechls::LaunchOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>(getCalleeAttrName());
}

void spechls::LaunchOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr(getCalleeAttrName(), cast<SymbolRefAttr>(callee));
}

Operation::operand_range spechls::LaunchOp::getArgOperands() { return getArguments(); }

MutableOperandRange spechls::LaunchOp::getArgOperandsMutable() { return getArgumentsMutable(); }

//===--------------------------------------------------------------------------------------------------------------===//
// TableGen'd op method definitions
//===--------------------------------------------------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/SpecHLS/IR/SpecHLS.cpp.inc"
