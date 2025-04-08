//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>

#include "Dialect/SpecHLS/IR/SpecHLSDialect.cpp.inc"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"

using namespace mlir;

//===--------------------------------------------------------------------------------------------------------------===//
// SpecHLS dialect
//===--------------------------------------------------------------------------------------------------------------===//

void spechls::SpecHLSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/SpecHLS/IR/SpecHLS.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/SpecHLS/IR/SpecHLSTypes.cpp.inc"
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
  ArrayRef<Type> results;
  StringRef taskName;

  if (auto task = dyn_cast<HKernelOp>((*this)->getParentOp())) {
    results = task.getResultTypes();
    taskName = task.getName();
  } else if (auto task = dyn_cast<HTaskOp>((*this)->getParentOp())) {
    results = task.getResultTypes();
    taskName = task.getName();
  } else
    return emitOpError("expected parent to be an hkernel or an htask op");

  // The number of committed values must match the task signature.
  if (getNumOperands() - 1 != results.size())
    return emitOpError("has ") << getNumOperands() - 1 << " operands, but enclosing hkernel (@" << taskName
                               << ") returns " << results.size();

  for (size_t i = 0, e = results.size(); i != e; ++i) {
    if (getOperand(i + 1).getType() != results[i])
      return emitError() << "type of exit operand " << i << " (" << getOperand(i + 1).getType()
                         << ") doesn't match result type (" << results[i] << ") in hkernel @" << taskName;
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

ParseResult spechls::ExitOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand guard;
  if (parser.parseKeyword("if") || parser.parseOperand(guard) ||
      parser.resolveOperand(guard, parser.getBuilder().getI1Type(), result.operands))
    return failure();

  if (parser.parseOptionalKeyword("with").succeeded()) {
    SmallVector<OpAsmParser::UnresolvedOperand> values;
    SmallVector<Type> valueTypes;
    SMLoc valueLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(values) || parser.parseColonTypeList(valueTypes) ||
        parser.resolveOperands(values, valueTypes, valueLoc, result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void spechls::ExitOp::print(OpAsmPrinter &printer) {
  printer << " if " << getGuard();

  if (getValues().size() > 0)
    printer << " with " << getValues() << " : " << getValues().getTypes();
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
  OpAsmParser::UnresolvedOperand select;
  if (parser.parseOperand(select))
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
  if (parser.resolveOperand(select, selectType, result.operands) ||
      parser.resolveOperands(inputs, inputTypes, inputsLoc, result.operands) ||
      parser.addTypeToList(argType, result.types))
    return failure();

  return success();
}

void spechls::GammaOp::print(OpAsmPrinter &printer) {
  auto select = getSelect();
  auto inputs = getInputs();

  printer << "<\"" << getSymName() << "\">(" << select << ", " << inputs << ") : " << select.getType() << ", "
          << inputs.front().getType();
}

LogicalResult spechls::GammaOp::verify() {
  auto inputs = getInputs();
  if (inputs.size() < 2)
    return emitOpError("expects at least two data inputs");

  unsigned int selectWidth = getSelect().getType().getWidth();
  if ((1ull << selectWidth) + 1 <= inputs.size())
    return emitOpError("has a select signal too narrow (")
           << selectWidth << " bit" << ((selectWidth > 1) ? "s" : "") << ") to select all of its inputs";

  return success();
}

ParseResult spechls::MuOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the symbol name specifier.
  StringAttr symbolNameAttr;
  if (parser.parseLess() || parser.parseAttribute(symbolNameAttr, getSymNameAttrName(result.name), result.attributes) ||
      parser.parseGreater())
    return failure();

  // Parse operands.
  SmallVector<OpAsmParser::UnresolvedOperand> inputs;
  SMLoc inputsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputs, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse the attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the type specifiers.
  Type argType;
  if (parser.parseColonType(argType))
    return failure();

  // Resolve operands.
  SmallVector<Type> inputTypes(inputs.size(), argType);
  if (parser.resolveOperands(inputs, inputTypes, inputsLoc, result.operands))
    return failure();

  result.addTypes(argType);
  return success();
}

void spechls::MuOp::print(OpAsmPrinter &printer) {
  printer << "<\"" << getSymName() << "\">(" << getInitValue() << ", " << getLoopValue()
          << ") : " << getInitValue().getType();
}

LogicalResult spechls::MuOp::verify() {
  auto init = getInitValue();
  if (!isa<BlockArgument>(init))
    return emitOpError("init value must be a task argument");

  return success();
}

CallInterfaceCallable spechls::CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>(getCalleeAttrName());
}

void spechls::CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr(getCalleeAttrName(), cast<SymbolRefAttr>(callee));
}

Operation::operand_range spechls::CallOp::getArgOperands() { return getArguments(); }

MutableOperandRange spechls::CallOp::getArgOperandsMutable() { return getArgumentsMutable(); }

ParseResult spechls::AlphaOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand array, index, value, we;
  Type indexType;
  spechls::ArrayType arrayType;

  if (parser.parseOperand(array) || parser.parseLSquare() || parser.parseOperand(index) ||
      parser.parseColonType(indexType) || parser.parseRSquare() || parser.parseComma() || parser.parseOperand(value) ||
      parser.parseKeyword("if") || parser.parseOperand(we) || parser.parseColonType(arrayType))
    return failure();

  // Resolve operands and results.
  if (parser.resolveOperand(array, arrayType, result.operands) ||
      parser.resolveOperand(index, indexType, result.operands) ||
      parser.resolveOperand(value, arrayType.getElementType(), result.operands) ||
      parser.resolveOperand(we, parser.getBuilder().getI1Type(), result.operands) ||
      parser.addTypeToList(arrayType, result.types))
    return failure();

  return success();
}

void spechls::AlphaOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getArray() << '[' << getIndex() << ": " << getIndex().getType() << "], " << getValue() << " if "
          << getWe() << " : " << getType();
}

LogicalResult spechls::AlphaOp::verify() {
  auto arrayType = getArray().getType();
  if (arrayType != getType())
    return emitOpError("has inconsistent input (")
           << getArray().getType() << ") and output (" << getType() << ") array types";
  if (arrayType.getElementType() != getValue().getType())
    return emitOpError("has inconsistent write value type (")
           << arrayType.getElementType() << ") and output array element type (" << getType().getElementType() << ")";

  return success();
}

LogicalResult spechls::LoadOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc, ValueRange operands,
                                                DictionaryAttr attributes, OpaqueProperties properties,
                                                RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  LoadOpAdaptor adaptor(operands, attributes, properties, regions);
  inferredReturnTypes.push_back(cast<spechls::ArrayType>(adaptor.getArray().getType()).getElementType());
  return success();
}

LogicalResult spechls::LUTOp::verify() {
  auto indexWidth = getIndex().getType().getWidth();
  if ((1ull << indexWidth) + 1 <= getContents().size())
    return emitOpError("has an index too narrow (")
           << indexWidth << " bit" << ((indexWidth > 1) ? "s" : "") << ") to select all of its inputs";

  // Make sure that the result type is wide enough to represent all of the LUT's possible values.
  unsigned requiredBits = 0;
  for (int64_t value : getContents()) {
    unsigned neededBits = APInt::getBitsNeeded(std::to_string(value), 10);
    if (neededBits > requiredBits)
      requiredBits = neededBits;
  }
  auto resultWidth = getResult().getType().getWidth();
  if (resultWidth < requiredBits)
    return emitOpError("has a result type too narrow to represent all possible values (required at least ")
           << requiredBits << " bits, but got " << resultWidth << ")";

  return success();
}

ParseResult spechls::DelayOp::parse(OpAsmParser &parser, OperationState &result) {
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

void spechls::DelayOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput() << " by " << getDepth();
  if (getEnable())
    printer << " if " << getEnable();
  if (getInit())
    printer << " init " << getInit();
  printer << " : " << getType();
}

LogicalResult spechls::FIFOOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc, ValueRange operands,
                                                DictionaryAttr attributes, OpaqueProperties properties,
                                                RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  // FIFO inputs have a write enable signal, that is not forwarded to their output.
  FIFOOpAdaptor adaptor(operands, attributes, properties, regions);
  StructType inputType = cast<StructType>(adaptor.getInput().getType());
  inferredReturnTypes.push_back(inputType.getFields().back());
  return success();
}

LogicalResult spechls::UnpackOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                                                  ValueRange operands, DictionaryAttr attributes,
                                                  OpaqueProperties properties, RegionRange regions,
                                                  SmallVectorImpl<Type> &inferredReturnTypes) {
  UnpackOpAdaptor adaptor(operands, attributes, properties, regions);
  StructType inputType = cast<StructType>(adaptor.getInput().getType());
  const auto &fields = inputType.getFields();
  inferredReturnTypes.append(fields.begin(), fields.end());
  return success();
}

//===--------------------------------------------------------------------------------------------------------------===//
// TableGen'd types and op method definitions
//===--------------------------------------------------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Dialect/SpecHLS/IR/SpecHLSTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/SpecHLS/IR/SpecHLS.cpp.inc"
