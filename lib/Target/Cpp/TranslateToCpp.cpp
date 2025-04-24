//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Target/Cpp/Export.h"
#include "circt/Dialect/Comb/CombDialect.h"

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LLVM.h>

#include <stack>

using namespace mlir;

namespace {
class CppEmitter {
  raw_indented_ostream os;

public:
  explicit CppEmitter(raw_ostream &os, bool declareStructTypes = false);

  raw_indented_ostream &ostream() { return os; }

  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  LogicalResult emitOperand(Value value);
  LogicalResult emitOperands(Operation &op);

  LogicalResult emitType(Location loc, Type type);
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);
  LogicalResult emitStructDefinition(Location loc, spechls::StructType structType);

  LogicalResult emitAttribute(Location loc, Attribute attr);

  LogicalResult emitVariableDeclaration(OpResult result, bool trailingSemicolon);
  LogicalResult emitVariableDeclaration(Location loc, Type type, StringRef name, bool trailingSemicolon);

  LogicalResult emitVariableAssignment(OpResult result);

  LogicalResult emitAssignPrefix(Operation &op);

  StringRef getOrCreateName(Value value);
  StringRef getExitVariableName() { return "exit_"; }

  bool shouldDeclareStructTypes() { return declareStructTypes; }

  class Scope {
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    CppEmitter &emitter;

  public:
    explicit Scope(CppEmitter &emitter) : valueMapperScope(emitter.valueMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
    }
    ~Scope() { emitter.valueInScopeCount.pop(); }
  };

  bool hasValueInScope(Value value) { return valueMapper.count(value); }

private:
  static bool shouldMapToUnsigned(IntegerType::SignednessSemantics semantics);
  static StringRef getValueNamePrefix(Value value);

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  ValueMapper valueMapper;

  std::stack<int64_t> valueInScopeCount;
  bool declareStructTypes;
};

template <typename ForwardIterator, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(ForwardIterator begin, ForwardIterator end, UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c, UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c, raw_ostream &os, UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

LogicalResult printFunctionArgs(CppEmitter &emitter, Operation *taskLikeOp, Region::BlockArgListType arguments) {
  raw_indented_ostream &os = emitter.ostream();

  return interleaveCommaWithError(arguments, os, [&](BlockArgument arg) -> LogicalResult {
    return emitter.emitVariableDeclaration(taskLikeOp->getLoc(), arg.getType(), emitter.getOrCreateName(arg), false);
  });
}

LogicalResult printAllVariables(CppEmitter &emitter, Operation *taskLikeOp) {
  WalkResult result = taskLikeOp->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    for (OpResult result : op->getResults()) {
      if (failed(emitter.emitVariableDeclaration(result, true)))
        return WalkResult(op->emitError("unable to declare result variable for op"));
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  return success();
}

LogicalResult printAllStructTypes(CppEmitter &emitter, ModuleOp moduleOp) {
  DenseSet<StringRef> generatedStructs{};

  WalkResult result = moduleOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    for (Type type : op->getResultTypes()) {
      if (auto sType = dyn_cast<spechls::StructType>(type)) {
        if (!generatedStructs.contains(sType.getName())) {
          if (failed(emitter.emitStructDefinition(op->getLoc(), sType)))
            return WalkResult(op->emitError("unable to declare structure type for op"));
          generatedStructs.insert(sType.getName());
        }
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

LogicalResult printFunctionBody(CppEmitter &emitter, Operation *taskLikeOp, Region::BlockListType &blocks,
                                bool indent = true) {
  raw_indented_ostream &os = emitter.ostream();
  if (indent)
    os.indent();

  for (auto &&block : blocks) {
    for (auto &&op : block.getOperations()) {
      if (failed(emitter.emitOperation(op, true)))
        return failure();
    }
  }

  if (indent)
    os.unindent();
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, ModuleOp moduleOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "#include <tuple>\n";
  os << "#include <ap_int.h>\n";
  os << "#include <io_printf.h>\n";
  os << "\n";

  if (emitter.shouldDeclareStructTypes()) {
    if (failed(printAllStructTypes(emitter, moduleOp)))
      return failure();
    os << "\n";
  }

  for (auto &&op : moduleOp) {
    if (failed(emitter.emitOperation(op, false)))
      return failure();
  }
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::HKernelOp hkernelOp) {
  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitTypes(hkernelOp.getLoc(), hkernelOp.getFunctionType().getResults())))
    return failure();
  os << ' ' << hkernelOp.getName() << "(";
  Operation *operation = hkernelOp.getOperation();
  if (failed(printFunctionArgs(emitter, operation, hkernelOp.getArguments())))
    return failure();
  os << ") {\n";
  os.indent();

  if (failed(printAllVariables(emitter, operation)))
    return failure();

  auto exit = cast<spechls::ExitOp>(hkernelOp.getBody().front().getTerminator());
  if (failed(emitter.emitVariableDeclaration(exit.getLoc(), exit.getGuard().getType(), emitter.getExitVariableName(),
                                             false)))
    return failure();
  os << " = false;\n\n";
  os << "while (!" << emitter.getExitVariableName() << ") {\n";

  if (failed(printFunctionBody(emitter, operation, hkernelOp.getBlocks())))
    return failure();

  os << "}\n";

  os << "return";
  switch (exit.getValues().size()) {
  case 0:
    break;
  case 1:
    os << ' ';
    if (failed(emitter.emitOperand(exit.getValues().front())))
      return failure();
    break;
  default:
    os << " std::make_tuple(";
    if (failed(interleaveCommaWithError(exit.getValues(), os,
                                        [&](Value operand) { return emitter.emitOperand(operand); }))) {
      return failure();
    }
    os << ')';
  }
  os << ";\n";

  os.unindent();
  os << "}";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::HTaskOp htaskOp) {
  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitTypes(htaskOp.getLoc(), htaskOp.getFunctionType().getResults())))
    return failure();
  os << ' ' << htaskOp.getName() << "(";
  Operation *operation = htaskOp.getOperation();
  if (failed(printFunctionArgs(emitter, operation, htaskOp.getArguments())))
    return failure();
  os << ") {\n";
  os.indent();

  if (failed(printAllVariables(emitter, operation)))
    return failure();
  os << "\n";

  if (failed(printFunctionBody(emitter, operation, htaskOp.getBlocks(), false)))
    return failure();

  os.unindent();
  os << "}";
  return success();
}

LogicalResult printCallOp(CppEmitter &emitter, Operation *operation, StringRef callee) {
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << callee << "(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::LaunchOp launchOp) {
  Operation *operation = launchOp.getOperation();
  StringRef callee = launchOp.getCallee();
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::DelayOp delayOp) {
  Operation *operation = delayOp.getOperation();
  StringRef callee = llvm::formatv("delay<{0}>", delayOp.getDepth()).sstr<16>();
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::ExitOp exitOp) {
  raw_ostream &os = emitter.ostream();
  os << emitter.getExitVariableName() << " = " << emitter.getOrCreateName(exitOp.getGuard());
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::MuOp muOp) {
  Operation *operation = muOp.getOperation();
  StringRef callee = "mu";
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::PackOp packOp) {
  Operation *operation = packOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitVariableAssignment(operation->getResult(0))))
    return failure();
  os << "std::make_tuple(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::CallOp callOp) {
  Operation *operation = callOp.getOperation();
  StringRef callee = callOp.getCallee();
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::GammaOp gammaOp) {
  Operation *operation = gammaOp.getOperation();
  StringRef callee = "gamma";
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::PrintOp printOp) {
  Operation *operation = printOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "io_printf(";
  if (failed(emitter.emitOperand(printOp.getState())))
    return failure();
  os << ", ";
  if (failed(emitter.emitOperand(printOp.getEnable())))
    return failure();
  os << ", ";
  if (failed(emitter.emitAttribute(operation->getLoc(), printOp.getFormatAttr())))
    return failure();
  if (printOp.getArgs().size() > 0)
    os << ", ";
  if (failed(
          interleaveCommaWithError(printOp.getArgs(), os, [&](Value operand) { return emitter.emitOperand(operand); })))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::CommitOp commitOp) {
  raw_indented_ostream &os = emitter.ostream();

  if (commitOp.getValues().size() == 0) {
    os << "return";
    return success();
  }

  os << "if (";
  if (failed(emitter.emitOperand(commitOp.getEnable())))
    return failure();
  os << ") {\n";
  os.indent();
  os << "return ";

  if (commitOp.getValues().size() == 1) {
    if (failed(emitter.emitOperand(commitOp.getValues().front())))
      return failure();
  } else {
    os << "std::make_tuple(";
    if (failed(interleaveCommaWithError(commitOp.getValues(), os,
                                        [&](Value operand) { return emitter.emitOperand(operand); }))) {
      return failure();
    }
    os << ")";
  }

  os << ";\n";
  os.unindent();
  os << "}\n";

  os << "return ";
  if (commitOp.getValues().size() == 1) {
    if (failed(emitter.emitType(commitOp.getLoc(), commitOp.getValues().front().getType())))
      return failure();
  } else {
    os << "std::tuple<";
    if (failed(interleaveCommaWithError(commitOp.getValues().getTypes(), os,
                                        [&](Type type) { return emitter.emitType(commitOp.getLoc(), type); }))) {
      return failure();
    }
    os << ">";
  }
  os << "{}";

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::FSMCommandOp fsmCommandOp) {
  Operation *operation = fsmCommandOp.getOperation();
  StringRef callee = "fsm_command";
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::FSMOp fsmOp) {
  Operation *operation = fsmOp.getOperation();
  StringRef callee = "fsm";
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::FieldOp fieldOp) {
  Operation *operation = fieldOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitVariableAssignment(operation->getResult(0))))
    return failure();
  if (failed(emitter.emitOperand(fieldOp.getInput())))
    return failure();
  os << "." << fieldOp.getName();

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::RewindOp rewindOp) {
  Operation *operation = rewindOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "rewind<";
  interleaveComma(rewindOp.getDepths(), os);
  os << ">(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::RollbackOp rollbackOp) {
  Operation *operation = rollbackOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "rollback<";
  interleaveComma(rollbackOp.getDepths(), os);
  os << ">(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::FIFOOp fifoOp) {
  Operation *operation = fifoOp.getOperation();
  StringRef callee = llvm::formatv("fifo<{0}>", fifoOp.getDepth()).sstr<16>();
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::UnpackOp unpackOp) {
  Operation *operation = unpackOp.getOperation();
  raw_ostream &os = emitter.ostream();

  os << "std::tie(" << emitter.getOrCreateName(operation->getResult(0)) << ") = ";
  if (failed(emitter.emitOperand(unpackOp.getOperand())))
    return failure();
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::LoadOp loadOp) {
  Operation *operation = loadOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitVariableAssignment(operation->getResult(0))))
    return failure();
  if (failed(emitter.emitOperand(loadOp.getArray())))
    return failure();
  os << "[";
  if (failed(emitter.emitOperand(loadOp.getIndex())))
    return failure();
  os << "]";

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::LUTOp lutOp) {
  Operation *operation = lutOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "lut<";
  interleaveComma(lutOp.getContents(), os);
  os << ">(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::AlphaOp alphaOp) {
  Operation *operation = alphaOp.getOperation();
  StringRef callee = "alpha";
  return printCallOp(emitter, operation, callee);
}

LogicalResult printBinaryOperation(CppEmitter &emitter, Operation *operation, StringRef binaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  if (failed(emitter.emitOperand(operation->getOperand(0))))
    return failure();
  os << ' ' << binaryOperator << ' ';
  if (failed(emitter.emitOperand(operation->getOperand(1))))
    return failure();
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::AddOp addOp) {
  Operation *operation = addOp.getOperation();
  return printBinaryOperation(emitter, operation, "+");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::AndOp andOp) {
  Operation *operation = andOp.getOperation();
  return printBinaryOperation(emitter, operation, "&");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::DivSOp divOp) {
  Operation *operation = divOp.getOperation();
  return printBinaryOperation(emitter, operation, "/");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::DivUOp divOp) {
  Operation *operation = divOp.getOperation();
  return printBinaryOperation(emitter, operation, "/");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::MulOp mulOp) {
  Operation *operation = mulOp.getOperation();
  return printBinaryOperation(emitter, operation, "*");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::OrOp orOp) {
  Operation *operation = orOp.getOperation();
  return printBinaryOperation(emitter, operation, "|");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::SubOp subOp) {
  Operation *operation = subOp.getOperation();
  return printBinaryOperation(emitter, operation, "-");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::XorOp xorOp) {
  Operation *operation = xorOp.getOperation();
  return printBinaryOperation(emitter, operation, "^");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ICmpOp icmpOp) {
  Operation *operation = icmpOp.getOperation();
  StringRef comparisonOperator = "";
  switch (icmpOp.getPredicate()) {
  case circt::comb::ICmpPredicate::eq:
    comparisonOperator = "==";
    break;
  case circt::comb::ICmpPredicate::ne:
    comparisonOperator = "!=";
    break;
  case circt::comb::ICmpPredicate::slt:
  case circt::comb::ICmpPredicate::ult:
    comparisonOperator = "<";
    break;
  case circt::comb::ICmpPredicate::sle:
  case circt::comb::ICmpPredicate::ule:
    comparisonOperator = "<=";
    break;
  case circt::comb::ICmpPredicate::sgt:
  case circt::comb::ICmpPredicate::ugt:
    comparisonOperator = ">";
    break;
  case circt::comb::ICmpPredicate::sge:
  case circt::comb::ICmpPredicate::uge:
    comparisonOperator = ">=";
    break;
  default:
    return operation->emitOpError("Unexpected icmp predicate ")
           << circt::comb::stringifyICmpPredicate(icmpOp.getPredicate());
  }
  return printBinaryOperation(emitter, operation, comparisonOperator);
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ReplicateOp replicateOp) {
  Operation *operation = replicateOp.getOperation();
  raw_ostream &os = emitter.ostream();

  size_t count = replicateOp.getResult().getType().getIntOrFloatBitWidth() /
                 replicateOp.getInput().getType().getIntOrFloatBitWidth();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "replicate<" << count << ">(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::MuxOp muxOp) {
  Operation *operation = muxOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitVariableAssignment(operation->getResult(0))))
    return failure();
  if (failed(emitter.emitOperand(muxOp.getCond())))
    return failure();
  os << " ? ";
  if (failed(emitter.emitOperand(muxOp.getTrueValue())))
    return failure();
  os << " : ";
  if (failed(emitter.emitOperand(muxOp.getFalseValue())))
    return failure();
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ExtractOp extractOp) {
  Operation *operation = extractOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitVariableAssignment(operation->getResult(0))))
    return failure();
  if (failed(emitter.emitOperand(extractOp.getOperand())))
    return failure();

  uint32_t lowBit = extractOp.getLowBit();
  uint32_t highBit = lowBit + extractOp.getType().getIntOrFloatBitWidth();
  os << ".range(" << lowBit << ", " << highBit << ")";

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ConcatOp concatOp) {
  Operation *operation = concatOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitVariableAssignment(operation->getResult(0))))
    return failure();
  if (failed(emitter.emitOperand(concatOp.getOperand(0))))
    return failure();
  os << ".concat(";
  if (failed(emitter.emitOperand(concatOp.getOperand(1))))
    return failure();
  os << ")";

  return success();
}

LogicalResult printConstantOp(CppEmitter &emitter, Operation *operation, Attribute value) {
  if (failed(emitter.emitVariableAssignment(operation->getResult(0))))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

LogicalResult printOperation(CppEmitter &emitter, circt::hw::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValueAttr();
  return printConstantOp(emitter, operation, value);
}
} // namespace

CppEmitter::CppEmitter(raw_ostream &os, bool declareStructTypes) : os(os), declareStructTypes(declareStructTypes) {
  valueInScopeCount.push(0);
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  LogicalResult status =
      TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // Comb ops.
          .Case<circt::comb::AddOp, circt::comb::AndOp, circt::comb::ConcatOp, circt::comb::DivSOp, circt::comb::DivUOp,
                circt::comb::ExtractOp, circt::comb::ICmpOp, circt::comb::MulOp, circt::comb::MuxOp, circt::comb::OrOp,
                circt::comb::SubOp, circt::comb::ReplicateOp, circt::comb::XorOp>(
              [&](auto op) { return printOperation(*this, op); })
          // HW ops.
          .Case<circt::hw::ConstantOp>([&](auto op) { return printOperation(*this, op); })
          // SpecHLS ops.
          .Case<spechls::AlphaOp, spechls::CallOp, spechls::CommitOp, spechls::DelayOp, spechls::ExitOp,
                spechls::FieldOp, spechls::FIFOOp, spechls::FSMCommandOp, spechls::FSMOp, spechls::GammaOp,
                spechls::HKernelOp, spechls::HTaskOp, spechls::LaunchOp, spechls::LoadOp, spechls::LUTOp, spechls::MuOp,
                spechls::PackOp, spechls::PrintOp, spechls::RewindOp, spechls::RollbackOp, spechls::UnpackOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Default([&](Operation *) { return op.emitOpError("unable to find printer for op"); });

  if (failed(status))
    return failure();

  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitOperand(Value value) {
  os << getOrCreateName(value);
  return success();
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  return interleaveCommaWithError(op.getOperands(), os, [&](Value operand) { return emitOperand(operand); });
}

LogicalResult CppEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    if (shouldMapToUnsigned(iType.getSignedness()))
      os << "ap_uint<" << iType.getWidth() << '>';
    else
      os << "ap_int<" << iType.getWidth() << '>';
    return success();
  }
  if (auto tType = dyn_cast<TupleType>(type)) {
    return emitTupleType(loc, tType.getTypes());
  }
  if (auto sType = dyn_cast<spechls::StructType>(type)) {
    os << sType.getName();
    return success();
  }
  if (auto aType = dyn_cast<spechls::ArrayType>(type)) {
    if (failed(emitType(loc, aType.getElementType())))
      return failure();
    os << "*";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type;
}

LogicalResult CppEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult CppEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult CppEmitter::emitStructDefinition(Location loc, spechls::StructType structType) {
  os << "struct " << structType.getName() << " {\n";
  os.indent();

  auto fieldNames = structType.getFieldNames();
  auto fieldTypes = structType.getFieldTypes();

  for (size_t i = 0; i < fieldTypes.size(); ++i) {
    if (failed(emitType(loc, fieldTypes[i])))
      return failure();
    os << " " << fieldNames[i] << ";\n";
  }

  os.unindent();
  os << "};\n";
  return success();
}

LogicalResult CppEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
  }
  if (auto sAttr = dyn_cast<StringAttr>(attr)) {
    os << '"';
    os.write_escaped(sAttr.strref());
    os << '"';
    return success();
  }

  return emitError(loc, "cannot emit attribute: ") << attr;
}

bool CppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics semantics) {
  switch (semantics) {
  case IntegerType::Signless:
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

StringRef CppEmitter::getValueNamePrefix(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return "arg";
  // Comb ops.
  if (isa<circt::comb::AddOp>(op))
    return "add";
  if (isa<circt::comb::AndOp>(op))
    return "and";
  if (isa<circt::comb::ConcatOp>(op))
    return "concat";
  if (isa<circt::comb::DivSOp>(op) || isa<circt::comb::DivUOp>(op))
    return "div";
  if (isa<circt::comb::ExtractOp>(op))
    return "extract";
  if (auto icmpOp = dyn_cast<circt::comb::ICmpOp>(op)) {
    switch (icmpOp.getPredicate()) {
    case circt::comb::ICmpPredicate::eq:
      return "eq";
    case circt::comb::ICmpPredicate::ne:
      return "ne";
    case circt::comb::ICmpPredicate::slt:
    case circt::comb::ICmpPredicate::ult:
      return "lt";
    case circt::comb::ICmpPredicate::sle:
    case circt::comb::ICmpPredicate::ule:
      return "le";
    case circt::comb::ICmpPredicate::sgt:
    case circt::comb::ICmpPredicate::ugt:
      return "gt";
    case circt::comb::ICmpPredicate::sge:
    case circt::comb::ICmpPredicate::uge:
      return "ge";
    default:
      return "v";
    }
  }
  if (isa<circt::comb::MulOp>(op))
    return "mul";
  if (isa<circt::comb::MuxOp>(op))
    return "mux";
  if (isa<circt::comb::OrOp>(op))
    return "or";
  if (isa<circt::comb::ReplicateOp>(op))
    return "replicate";
  if (isa<circt::comb::SubOp>(op))
    return "sub";
  if (isa<circt::comb::XorOp>(op))
    return "xor";
  // HW ops.
  if (auto constantOp = dyn_cast<circt::hw::ConstantOp>(op)) {
    if (constantOp.getValue().getBitWidth() == 1)
      return constantOp.getValue().getBoolValue() ? "true" : "false";
    return constantOp.getValue().isZero() ? "zero" : "cst";
  }
  // SpecHLS ops.
  if (isa<spechls::AlphaOp>(op))
    return "alpha";
  if (auto call = dyn_cast<spechls::CallOp>(op))
    return call.getCallee();
  if (isa<spechls::DelayOp>(op))
    return "delay";
  if (auto field = dyn_cast<spechls::FieldOp>(op))
    return field.getName();
  if (isa<spechls::FIFOOp>(op))
    return "fifo";
  if (isa<spechls::FSMOp>(op))
    return "fsm";
  if (isa<spechls::FSMCommandOp>(op))
    return "fsm_command";
  if (isa<spechls::GammaOp>(op))
    return "gamma";
  if (auto launch = dyn_cast<spechls::LaunchOp>(op))
    return launch.getCallee();
  if (isa<spechls::LoadOp>(op))
    return "load";
  if (isa<spechls::LUTOp>(op))
    return "lut";
  if (isa<spechls::MuOp>(op))
    return "mu";
  if (isa<spechls::PrintOp>(op))
    return "print_state";
  if (isa<spechls::RewindOp>(op))
    return "rewind";
  if (isa<spechls::RollbackOp>(op))
    return "rollback";
  return "v";
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result, bool trailingSemicolon) {
  if (hasValueInScope(result))
    return result.getDefiningOp()->emitError("result variable for the operation already declared");

  return emitVariableDeclaration(result.getOwner()->getLoc(), result.getType(), getOrCreateName(result),
                                 trailingSemicolon);
}

LogicalResult CppEmitter::emitVariableDeclaration(Location loc, Type type, StringRef name, bool trailingSemicolon) {
  if (failed(emitType(loc, type)))
    return failure();
  os << ' ' << name;
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult CppEmitter::emitVariableAssignment(OpResult result) {
  if (!hasValueInScope(result))
    return result.getDefiningOp()->emitOpError("result variable for the operation has not been declared");
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (failed(emitVariableAssignment(result)))
      return failure();
    break;
  }
  default:
    os << "std::tie(";
    interleaveComma(op.getResults(), os, [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

StringRef CppEmitter::getOrCreateName(Value value) {
  if (!valueMapper.count(value)) {
    valueMapper.insert(value, llvm::formatv("{0}_{1}", getValueNamePrefix(value), ++valueInScopeCount.top()));
  }
  return *valueMapper.begin(value);
}

LogicalResult spechls::translateToCpp(Operation *op, raw_ostream &os, bool declareStructTypes) {
  CppEmitter emitter(os, declareStructTypes);
  return emitter.emitOperation(*op, false);
}
