//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
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
#include <string>

#include "Dialect/SpecHLS/IR/SpecHLSDialect.cpp.inc"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Utils.h"
#include "llvm/Support/LogicalResult.h"

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

Type spechls::StructType::parse(AsmParser &parser) {
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());

  std::string name;
  if (parser.parseLess() || parser.parseString(&name))
    return {};

  SmallVector<Type, 4> fieldTypes;
  SmallVector<std::string, 4> fieldNames;

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces, [&]() {
        std::string fieldName;
        Type fieldType;
        if (parser.parseString(&fieldName) || parser.parseColonType(fieldType))
          return failure();
        fieldTypes.push_back(std::move(fieldType));
        fieldNames.push_back(std::move(fieldName));
        return success();
      })) {
    return {};
  }

  if (parser.parseGreater())
    return {};

  return StructType::getChecked([loc] { return emitError(loc); }, loc.getContext(), name, fieldNames, fieldTypes);
}

void spechls::StructType::print(AsmPrinter &printer) const {
  printer << "<\"" << getName() << "\" { ";
  llvm::interleaveComma(llvm::zip(getFieldNames(), getFieldTypes()), printer,
                        [&](auto p) { printer << '"' << std::get<0>(p) << "\" : " << std::get<1>(p); });
  printer << " }>";
}

LogicalResult spechls::StructType::verify(function_ref<InFlightDiagnostic()> emitError, StringRef name,
                                          ArrayRef<std::string> fieldNames, ArrayRef<Type> fieldTypes) {
  if (fieldNames.size() != fieldTypes.size())
    return emitError() << "field name and field type count mismatch";
  return success();
}

ParseResult spechls::KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name), result.attributes))
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
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(builder.getFunctionType(argTypes, resultTypes)));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Add attributes to the arguments and results.
  assert(resultAttrs.size() == resultTypes.size());
  call_interface_impl::addArgAndResultAttrs(builder, result, entryArgs, resultAttrs, getArgAttrsAttrName(result.name),
                                            getResAttrsAttrName(result.name));

  // Parse the kernel body.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, entryArgs);
}

void spechls::KernelOp::print(OpAsmPrinter &printer) {
  // Print the kernel signature and attributes.
  printer << ' ';
  printer.printSymbolName(getName());
  auto functionType = getFunctionType();
  function_interface_impl::printFunctionSignature(printer, *this, functionType.getInputs(), false,
                                                  functionType.getResults());
  function_interface_impl::printFunctionAttributes(
      printer, *this, {getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName()});

  // Print the kernel body.
  printer << ' ';
  printer.printRegion(getBody(), false, true);
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

LogicalResult spechls::ExitOp::verify() {
  ArrayRef<Type> results;
  StringRef taskName;

  auto kernel = cast<KernelOp>((*this)->getParentOp());
  results = kernel.getResultTypes();
  taskName = kernel.getName();

  // The number of committed values must match the task signature.
  if (getNumOperands() - 1 != results.size())
    return emitOpError("has ") << getNumOperands() - 1 << " operands, but enclosing kernel (@" << taskName
                               << ") returns " << results.size();

  for (size_t i = 0, e = results.size(); i != e; ++i) {
    if (getOperand(i + 1).getType() != results[i])
      return emitError() << "type of exit operand " << i << " (" << getOperand(i + 1).getType()
                         << ") doesn't match result type (" << results[i] << ") in kernel @" << taskName;
  }

  return success();
}

ParseResult spechls::TaskOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;

  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

  FunctionType funcType;
  if (parser.parseAttribute(nameAttr, getSymNameAttrName(result.name), result.attributes) ||
      parser.parseAssignmentList(regionArgs, operands) || parser.parseColonType(funcType) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  result.addTypes(funcType.getResults());

  if (regionArgs.size() != funcType.getInputs().size())
    return parser.emitError(parser.getNameLoc(), "missing types for task arguments");

  for (auto [arg, type] : llvm::zip_equal(regionArgs, funcType.getInputs()))
    arg.type = type;

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  for (auto [operand, type] : llvm::zip_equal(operands, funcType.getInputs())) {
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();
  }

  return success();
}

void spechls::TaskOp::print(OpAsmPrinter &printer) {
  printer << " \"" << getSymName() << "\"(";
  llvm::interleaveComma(llvm::zip(getRegion().getArguments(), getArgs()), printer,
                        [&](auto it) { printer << std::get<0>(it) << " = " << std::get<1>(it); });
  printer << ") : (";
  llvm::interleaveComma(getArgs(), printer, [&](auto it) { printer.printType(it.getType()); });
  printer << ") -> " << getResult().getType() << " ";
  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {getSymNameAttrName()});
  printer.printRegion(getBody(), false);
}

LogicalResult spechls::CommitOp::verify() {
  auto task = cast<TaskOp>((*this)->getParentOp());
  auto taskType = task.getResult().getType();
  if (taskType.getFieldTypes().size() != getNumOperands()) {
    return emitError() << "Bad number of commit operands. Got" << getNumOperands() << ", but "
                       << taskType.getFieldTypes().size() << "was expected.\n";
  }
  for (unsigned i = 0; i < getNumOperands(); ++i) {
    if (getOperand(i).getType() != taskType.getFieldTypes()[i]) {
      return emitError() << "Bad commit operand type at index " << i << ". Operand of type " << getOperand(i).getType()
                         << ", but type" << taskType.getFieldTypes()[i] << "was expected.\n";
    }
  }
  return success();
}

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

  printer << "<\"" << getSymName() << "\">(" << select << ", " << inputs << ") "
          << (*this)->getDiscardableAttrDictionary() << ": " << select.getType() << ", " << inputs.front().getType();
}

LogicalResult spechls::GammaOp::verify() {
  auto inputs = getInputs();
  if (inputs.size() < 2)
    return emitOpError("expects at least two data inputs");

  unsigned int selectWidth = getSelect().getType().getWidth();
  if (selectWidth < utils::getMinBitwidth(inputs.size() - 1))
    return emitOpError("has a select signal too narrow (")
           << selectWidth << " bit" << ((selectWidth > 1) ? "s" : "") << ") to select all of its inputs (required "
           << utils::getMinBitwidth(inputs.size() - 1) << ")";

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
  printer << "<\"" << getSymName() << "\">(" << getInitValue() << ", " << getLoopValue() << ") "
          << (*this)->getDiscardableAttrDictionary() << ": " << getInitValue().getType();
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
      parser.parseKeyword("if") || parser.parseOperand(we) || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(arrayType))
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
          << getWe();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getType();
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
  size_t contentSize = getContents().size();
  if (!(contentSize > 0 && (contentSize & (contentSize - 1)) == 0)) {
    return emitOpError("contents size sould be a power of two");
  }

  auto indexWidth = getIndex().getType().getWidth();
  size_t contentWidth = utils::getMinBitwidth(getContents().size() - 1);
  if (indexWidth != contentWidth) {
    return emitOpError("has an index too ") << ((indexWidth < contentWidth) ? "narrow" : "wide") << " (" << indexWidth
                                            << " bit" << ((indexWidth > 1) ? "s" : "") << ")";
  }

  // Make sure that the result type is wide enough to represent all of the LUT's possible values.
  unsigned requiredBits = 0;
  for (int64_t value : getContents()) {
    unsigned neededBits = utils::getMinBitwidth(value);
    if (neededBits > requiredBits)
      requiredBits = neededBits;
  }
  auto resultWidth = getResult().getType().getWidth();
  if (resultWidth < requiredBits) {
    return emitOpError("has a result type too narrow to represent all possible values (required at least ")
           << requiredBits << " bits, but got " << resultWidth << ")";
  }

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
  printer << " " << (*this)->getDiscardableAttrDictionary() << " : " << getType();
}

ParseResult spechls::CancellableDelayOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand input, enable, init, cancel, cancelWe;
  Type type, cancelType;
  uint32_t delay, offset;

  if (parser.parseOperand(input) || parser.parseKeyword("by") || parser.parseInteger(delay) ||
      parser.parseKeyword("cancel") || parser.parseOperand(cancel) || parser.parseColonType(cancelType) ||
      parser.parseKeyword("at") || parser.parseInteger(offset) || parser.parseOperand(cancelWe))
    return failure();
  result.addAttribute(getDepthAttrName(result.name), builder.getUI32IntegerAttr(delay));
  result.addAttribute(getOffsetAttrName(result.name), builder.getUI32IntegerAttr(offset));

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
                      builder.getDenseI32ArrayAttr({1, 1, 1, hasEnable, hasInit}));

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type) ||
      parser.resolveOperand(input, type, result.operands) || parser.addTypeToList(type, result.types))
    return failure();
  if (parser.resolveOperand(cancel, cancelType, result.operands))
    return failure();
  if (parser.resolveOperand(cancelWe, builder.getI1Type(), result.operands))
    return failure();
  if (hasEnable && parser.resolveOperand(enable, builder.getI1Type(), result.operands))
    return failure();
  if (hasInit && parser.resolveOperand(init, type, result.operands))
    return failure();

  return success();
}

void spechls::CancellableDelayOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput() << " by " << getDepth();
  printer << " cancel " << getCancel() << " : " << getCancel().getType() << " at " << getOffset() << " "
          << getCancelWe();
  if (getEnable())
    printer << " if " << getEnable();
  if (getInit())
    printer << " init " << getInit();
  printer << " " << (*this)->getDiscardableAttrDictionary() << " : " << getType();
}

ParseResult spechls::RollbackableDelayOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand input, enable, init, rollback, rbWe;
  Type type, rollbackType;
  uint32_t delay, offset;

  if (parser.parseOperand(input) || parser.parseKeyword("by") || parser.parseInteger(delay) ||
      parser.parseKeyword("rollback") || parser.parseOperand(rollback) || parser.parseColonType(rollbackType) ||
      parser.parseKeyword("at") || parser.parseInteger(offset) || parser.parseOperand(rbWe))
    return failure();
  llvm::SmallVector<int64_t> rollbackDepths;
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
        int64_t depth;
        if (parser.parseInteger(depth))
          return failure();
        rollbackDepths.push_back(depth);
        return success();
      }))
    return failure();

  result.addAttribute(getRollbackDepthsAttrName(result.name), builder.getDenseI64ArrayAttr(rollbackDepths));
  result.addAttribute(getDepthAttrName(result.name), builder.getUI32IntegerAttr(delay));
  result.addAttribute(getOffsetAttrName(result.name), builder.getUI32IntegerAttr(offset));

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
                      builder.getDenseI32ArrayAttr({1, 1, 1, hasEnable, hasInit}));

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type) ||
      parser.resolveOperand(input, type, result.operands) || parser.addTypeToList(type, result.types))
    return failure();
  if (parser.resolveOperand(rollback, rollbackType, result.operands))
    return failure();
  if (parser.resolveOperand(rbWe, builder.getI1Type(), result.operands))
    return failure();
  if (hasEnable && parser.resolveOperand(enable, builder.getI1Type(), result.operands))
    return failure();
  if (hasInit && parser.resolveOperand(init, type, result.operands))
    return failure();

  return success();
}

void spechls::RollbackableDelayOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput() << " by " << getDepth();
  printer << " rollback " << getRollback() << " : " << getRollback().getType() << " at " << getOffset() << " "
          << getRbWe() << " [" << getRollbackDepths() << "]";
  if (getEnable())
    printer << " if " << getEnable();
  if (getInit())
    printer << " init " << getInit();
  printer << " " << (*this)->getDiscardableAttrDictionary() << " : " << getType();
}

LogicalResult spechls::FIFOOp::verify() {
  StructType structType = getInput().getType();
  auto fields = structType.getFieldTypes();
  if (!fields.front().isInteger(1)) {
    return emitOpError(
               "FIFO input type expected to be a structure of the form !spechls.struct<i1, output-type...>, but got ")
           << structType;
  }
  return success();
}

LogicalResult spechls::UnpackOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                                                  ValueRange operands, DictionaryAttr attributes,
                                                  OpaqueProperties properties, RegionRange regions,
                                                  SmallVectorImpl<Type> &inferredReturnTypes) {
  UnpackOpAdaptor adaptor(operands, attributes, properties, regions);
  StructType inputType = cast<StructType>(adaptor.getInput().getType());
  const auto &fields = inputType.getFieldTypes();
  inferredReturnTypes.append(fields.begin(), fields.end());
  return success();
}

LogicalResult spechls::SyncOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc, ValueRange operands,
                                                DictionaryAttr attributes, OpaqueProperties properties,
                                                RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  SyncOpAdaptor adaptor(operands, attributes, properties, regions);
  inferredReturnTypes.push_back(adaptor.getInputs().getTypes().front());
  return success();
}

LogicalResult spechls::FieldOp::verify() {
  StringRef fieldName = getName();
  StructType inputType = getInput().getType();
  auto fieldNames = inputType.getFieldNames();
  if (std::find(fieldNames.begin(), fieldNames.end(), fieldName) == fieldNames.end())
    return emitOpError("invalid field name '") << fieldName << "' for struct " << inputType;
  return success();
}

LogicalResult spechls::FieldOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc, ValueRange operands,
                                                 DictionaryAttr attributes, OpaqueProperties properties,
                                                 RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  FieldOpAdaptor adaptor(operands, attributes, properties, regions);
  StructType inputType = cast<StructType>(adaptor.getInput().getType());

  auto fieldNames = inputType.getFieldNames();
  auto fieldTypes = inputType.getFieldTypes();
  size_t i = 0;
  for (auto &&name : fieldNames) {
    if (name == adaptor.getName()) {
      inferredReturnTypes.push_back(fieldTypes[i]);
      break;
    }
    ++i;
  }
  return success();
}

mlir::OpFoldResult spechls::FieldOp::fold(FoldAdaptor adaptor) {
  auto *pred = getInput().getDefiningOp();
  if (pred != nullptr) {
    if (auto pack = llvm::dyn_cast<spechls::PackOp>(pred)) {
      auto structType = pack.getType();
      for (unsigned i = 0; i < structType.getFieldNames().size(); ++i) {
        if (structType.getFieldNames()[i] == getName()) {
          return pack.getOperand(i);
        }
      }
    }
  }
  return nullptr;
}

//===--------------------------------------------------------------------------------------------------------------===//
// TableGen'd types and op method definitions
//===--------------------------------------------------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Dialect/SpecHLS/IR/SpecHLSTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/SpecHLS/IR/SpecHLS.cpp.inc"
