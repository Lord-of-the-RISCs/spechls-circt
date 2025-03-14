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

ParseResult spechls::GammaOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the symbol name specifier.
  StringAttr symbolNameAttr;
  if (parser.parseLess() || parser.parseAttribute(symbolNameAttr, getSymNameAttrName(result.name), result.attributes) ||
      parser.parseGreater() || parser.parseLParen())
    return failure();

  // Parse operands.
  OpAsmParser::Argument selectArgInfo;
  if (parser.parseOperand(selectArgInfo.ssaName))
    return failure();
  SmallVector<OpAsmParser::UnresolvedOperand> inputs;
  SMLoc inputsLoc = parser.getCurrentLocation();
  if (parser.parseTrailingOperandList(inputs) || parser.parseRParen())
    return failure();

  // Parse the attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the type specifiers.
  Type selectType;
  Type argType;
  if (parser.parseColon() || parser.parseType(selectType) || parser.parseComma() || parser.parseType(argType))
    return failure();

  // Resolve operands.
  SmallVector<Type> inputTypes(inputs.size(), argType);
  if (parser.resolveOperand(selectArgInfo.ssaName, selectType, result.operands) ||
      parser.resolveOperands(inputs, inputTypes, inputsLoc, result.operands))
    return failure();

  result.addTypes(argType);
  return success();
}

void spechls::GammaOp::print(OpAsmPrinter &printer) {
  auto select = getSelect();
  auto inputs = getInputs();

  printer << "<\"" << getSymName() << "\">(" << select << ", ";
  printer.printOperands(inputs);
  printer << ") : ";
  printer.printType(select.getType());
  printer << ", ";
  printer.printType(inputs.front().getType());
}

LogicalResult spechls::GammaOp::verify() {
  auto inputs = getInputs();
  if (inputs.size() < 2)
    return emitOpError("expects at least two data inputs");

  unsigned int selectWidth = getSelect().getType().getWidth();
  if ((1ull << selectWidth) <= inputs.size())
    return emitOpError("has a select signal too narrow (")
           << selectWidth << " bit" << ((selectWidth > 1) ? "s" : "") << ") to select all of its inputs";

  return success();
}

LogicalResult spechls::GammaOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                                 ValueRange operands, DictionaryAttr attributes,
                                                 OpaqueProperties properties, RegionRange regions,
                                                 SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands.back().getType());
  return success();
}

//===--------------------------------------------------------------------------------------------------------------===//
// TableGen'd op method definitions
//===--------------------------------------------------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/SpecHLS/IR/SpecHLS.cpp.inc"
