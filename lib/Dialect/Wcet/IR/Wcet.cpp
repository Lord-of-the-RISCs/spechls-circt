//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Wcet/IR/WcetOps.h"

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

#include "Dialect/Wcet/IR/WcetDialect.cpp.inc"
#include "llvm/ADT/ArrayRef.h"

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

// LogicalResult wcet::DummyOp::canonicalize(wcet::DummyOp op, PatternRewriter &rewriter) {
//   // patterns and rewrites go here.
//   auto inputs = op.getInputs();
//   auto outputs = op.getOutputs();
//   SmallVector<Value> newIn = SmallVector<Value>();
//   SmallVector<Type> newInType = SmallVector<Type>();
//   bool change = false;
//   for (size_t i = 0; i < inputs.size(); i++) {
//     newIn.push_back(inputs[i]);
//     newInType.push_back(inputs[i].getType());
//     auto *in = inputs[i].getDefiningOp();
//     if (!in)
//       continue;
//     if (in->hasTrait<mlir::OpTrait::ConstantLike>()) {
//       rewriter.replaceAllUsesWith(outputs[i], inputs[i]);
//       newIn.pop_back();
//       newInType.pop_back();
//       change = true;
//     }
//   }
//   if (!change)
//     return failure();
//
//   op->setOperands(newIn);
//   return success();
// }

//===--------------------------------------------------------------------------------------------------------------===//
// PenaltyOp
//===--------------------------------------------------------------------------------------------------------------===//

// LogicalResult wcet::PenaltyOp::canonicalize(wcet::PenaltyOp op, PatternRewriter &rewriter) {
//   if (!op.getInput().getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>())
//     return failure();
//   for (auto *us : op.getInput().getUsers()) {
//     llvm::errs() << us->getName() << "\n";
//     us->setAttr("wcet.delay", rewriter.getI32IntegerAttr(op.getDepth()));
//   }
//   rewriter.replaceAllOpUsesWith(op, op.getInput());
//   rewriter.eraseOp(op);
//   return success();
// }

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

#define GET_TYPEDEF_CLASSES
#include "Dialect/Wcet/IR/WcetTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Wcet/IR/Wcet.cpp.inc"
