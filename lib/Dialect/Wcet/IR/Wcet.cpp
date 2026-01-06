//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Wcet/IR/WcetOps.h"

#include "Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include <circt/Dialect/HW/HWOps.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <type_traits>

#include "Dialect/Wcet/IR/WcetDialect.cpp.inc"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===--------------------------------------------------------------------------------------------------------------===//
// Wcet Inliner Interface
//===--------------------------------------------------------------------------------------------------------------===//

struct WcetInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final { return true; }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned, IRMapping &valueMapping) const final {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto commit = cast<wcet::CommitOp>(op);
    for (const auto &it : llvm::enumerate(commit->getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  Operation *materializeCallConversion(OpBuilder &builder, Value input, Type resultType,
                                       Location conversionLoc) const final {
    return wcet::CastOp::create(builder, conversionLoc, resultType, input);
  }
};

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
  addInterface<WcetInlinerInterface>();
}

//
// LogicalResult wcet::OperationOp::verify() {
//  if (getNumOperands() != getDistances().size())
//    return emitOpError("expects ") << getNumOperands() << " distances, but got " << getDistances().size();
//  return success();
//}

//===--------------------------------------------------------------------------------------------------------------===//
// DummyOp
//===--------------------------------------------------------------------------------------------------------------===//

LogicalResult wcet::DummyOp::canonicalize(wcet::DummyOp op, PatternRewriter &rewriter) {
  // patterns and rewrites go here.
  auto inputs = op.getInputs();
  auto outputs = op.getOutputs();
  SmallVector<Value> newIn = SmallVector<Value>();
  SmallVector<Type> newInType = SmallVector<Type>();
  bool change = false;
  for (size_t i = 0; i < inputs.size(); i++) {
    newIn.push_back(inputs[i]);
    newInType.push_back(inputs[i].getType());
    auto *in = inputs[i].getDefiningOp();
    if (!in)
      continue;
    if (in->hasTrait<mlir::OpTrait::ConstantLike>()) {
      rewriter.replaceAllUsesWith(outputs[i], inputs[i]);
      // newIn.pop_back();
      // newInType.pop_back();
      change = true;
    }
  }
  if (!change)
    return failure();

  op->setOperands(newIn);
  return success();
}

//===--------------------------------------------------------------------------------------------------------------===//
// PenaltyOp
//===--------------------------------------------------------------------------------------------------------------===//

LogicalResult wcet::PenaltyOp::canonicalize(wcet::PenaltyOp op, PatternRewriter &rewriter) {
  if (!op.getInput().getDefiningOp())
    return failure();
  if (!op.getInput().getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>())
    return failure();
  for (auto *us : op.getInput().getUsers()) {
    us->setAttr("wcet.delay", rewriter.getI32IntegerAttr(op.getDepth()));
  }
  rewriter.replaceAllOpUsesWith(op, op.getInput());
  rewriter.eraseOp(op);
  return success();
}

//===--------------------------------------------------------------------------------------------------------------===//
// CastOp
//===--------------------------------------------------------------------------------------------------------------===//

bool wcet::CastOp::areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs) { return true; }

//===--------------------------------------------------------------------------------------------------------------===//
// CoreOp
//===--------------------------------------------------------------------------------------------------------------===//

void wcet::CoreOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                         mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult wcet::CoreOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  auto buildFuncType = [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
                          llvm::ArrayRef<mlir::Type> results, mlir::function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(parser, result, false, getFunctionTypeAttrName(result.name),
                                                        buildFuncType, getArgAttrsAttrName(result.name),
                                                        getResAttrsAttrName(result.name));
}

void wcet::CoreOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(p, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
                                                 getResAttrsAttrName());
}

//===--------------------------------------------------------------------------------------------------------------===//
// CoreInstanceOp
//===--------------------------------------------------------------------------------------------------------------===//

CallInterfaceCallable wcet::CoreInstanceOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void wcet::CoreInstanceOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

Operation::operand_range wcet::CoreInstanceOp::getArgOperands() { return getInputs(); }

MutableOperandRange wcet::CoreInstanceOp::getArgOperandsMutable() { return getInputsMutable(); }

//===--------------------------------------------------------------------------------------------------------------===//
// CommitOp
//===--------------------------------------------------------------------------------------------------------------===//

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

ParseResult wcet::GammaOp::parse(OpAsmParser &parser, OperationState &result) {
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

void wcet::GammaOp::print(OpAsmPrinter &printer) {
  auto select = getSelect();
  auto inputs = getInputs();

  printer << "<\"" << getSymName() << "\">(" << select << ", " << inputs << ") "
          << (*this)->getDiscardableAttrDictionary() << ": " << select.getType() << ", " << inputs.front().getType();
}

LogicalResult wcet::GammaOp::verify() {
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

LogicalResult wcet::GammaOp::canonicalize(wcet::GammaOp gamma, PatternRewriter &rewriter) {
  auto selectOp = dyn_cast<circt::hw::ConstantOp>(gamma.getSelect().getDefiningOp());
  if (!selectOp)
    return failure();
  auto idxAttr = selectOp.getValueAttr();
  auto idx = idxAttr.getInt();
  auto sidx = (size_t)(idx & ((1 << idxAttr.getType().getIntOrFloatBitWidth()) - 1));
  if (sidx >= gamma.getInputs().size()) {
    return failure();
  }
  rewriter.replaceAllOpUsesWith(gamma, gamma.getInputs()[sidx]);
  return success();
}

LogicalResult wcet::LUTOp::verify() {
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

LogicalResult wcet::LUTOp::canonicalize(wcet::LUTOp lut, PatternRewriter &rewriter) {

  auto selectOp = dyn_cast<circt::hw::ConstantOp>(lut.getIndex().getDefiningOp());
  if (!selectOp)
    return failure();

  auto idxAttr = selectOp.getValueAttr();
  size_t idx = (size_t)(idxAttr.getInt() & ((1 << selectOp.getType().getIntOrFloatBitWidth()) - 1));
  auto cnst = rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), lut.getType(), lut.getContents()[idx]);
  rewriter.replaceAllOpUsesWith(lut, cnst);
  return success();
}

LogicalResult wcet::ConstArrayRead::canonicalize(wcet::ConstArrayRead read, PatternRewriter &rewriter) {

  auto selectOp = dyn_cast<circt::hw::ConstantOp>(read.getIndex().getDefiningOp());
  if (!selectOp)
    return failure();

  auto idxAttr = selectOp.getValueAttr();
  size_t idx = (size_t)(idxAttr.getInt() & (((size_t)1 << selectOp.getType().getIntOrFloatBitWidth()) - 1));
  auto step = cast<IntegerAttr>(read.getStepAttr()).getInt();
  if (step < 0) {
    llvm::errs() << step << "\n";
    return failure();
  }
  if (idx / step >= read.getContents().size())
    return failure();
  auto cnst =
      rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), read.getType(), read.getContents()[idx / step]);
  rewriter.replaceAllOpUsesWith(read, cnst);
  return success();
}
#define GET_TYPEDEF_CLASSES
#include "Dialect/Wcet/IR/WcetTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Wcet/IR/Wcet.cpp.inc"
