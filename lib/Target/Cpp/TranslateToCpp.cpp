//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Target/Cpp/Export.h"

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <stack>

using namespace mlir;

namespace {
class CppEmitter {
  raw_indented_ostream os;

public:
  explicit CppEmitter(raw_ostream &os, bool declareStructTypes = false, bool declareFunctions = false);

  raw_indented_ostream &ostream() { return os; }

  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  LogicalResult emitOperand(Value value);
  LogicalResult emitOperands(Operation &op);

  LogicalResult emitType(Location loc, Type type);
  LogicalResult emitReference(Location loc, Type type);
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);
  LogicalResult emitStructDefinition(Location loc, spechls::StructType structType);
  LogicalResult emitFunctionPrototype(Location loc, StringRef callee, ArrayRef<Type> argumentTypes, Type returnType);

  LogicalResult emitAttribute(Location loc, Attribute attr);

  LogicalResult emitVariableDeclaration(OpResult result, bool trailingSemicolon);
  LogicalResult emitVariableDeclaration(Location loc, Type type, StringRef name, bool trailingSemicolon,
                                        bool isReference = false);

  LogicalResult emitVariableAssignment(OpResult result);

  LogicalResult emitAssignPrefix(Operation &op);

  StringRef getOrCreateName(Value value);
  StringRef getInitVariableName() { return "init_"; }
  StringRef getExitVariableName() { return "exit_"; }

  bool shouldDeclareStructTypes() { return declareStructTypes; }
  bool shouldDeclareFunctions() { return declareFunctions; }

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
  static std::string getValueNamePrefix(Value value);

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  ValueMapper valueMapper;

  std::stack<int64_t> valueInScopeCount;
  bool declareStructTypes;
  bool declareFunctions;
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

bool isSelfInitializedDelay(spechls::DelayOp delayOp) {
  return delayOp.getInit() && delayOp.getInit().getDefiningOp() == delayOp.getOperation();
}

std::string getDelayBufferName(CppEmitter &emitter, spechls::DelayOp delayOp) {
  return emitter.getOrCreateName(delayOp).str() + "_buffer";
}

std::string getRewindBufferName(CppEmitter &emitter, spechls::RewindOp rewindOp) {
  return emitter.getOrCreateName(rewindOp).str() + "_buffer";
}

std::string getRollbackBufferName(CppEmitter &emitter, spechls::RollbackOp rollbackOp) {
  return emitter.getOrCreateName(rollbackOp).str() + "_buffer";
}

std::string getCancelBufferName(CppEmitter &emitter, spechls::CancelOp cancelOp) {
  return emitter.getOrCreateName(cancelOp).str() + "_buffer";
}

std::string getFifoBufferName(CppEmitter &emitter, spechls::FIFOOp fifoOp) {
  return emitter.getOrCreateName(fifoOp).str() + "_buffer";
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c, UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c, raw_ostream &os, UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

LogicalResult printFunctionArgs(CppEmitter &emitter, Operation *taskLikeOp, Region::BlockArgListType arguments,
                                bool useReferences) {
  raw_indented_ostream &os = emitter.ostream();

  return interleaveCommaWithError(arguments, os, [&](BlockArgument arg) -> LogicalResult {
    return emitter.emitVariableDeclaration(taskLikeOp->getLoc(), arg.getType(), emitter.getOrCreateName(arg), false,
                                           useReferences && !isa<spechls::ArrayType>(arg.getType()));
  });
}

LogicalResult printFunctionArgTypes(CppEmitter &emitter, Operation *taskLikeOp, Region::BlockArgListType arguments,
                                    bool useReferences) {
  raw_indented_ostream &os = emitter.ostream();

  return interleaveCommaWithError(arguments, os, [&](BlockArgument arg) -> LogicalResult {
    return (useReferences && !isa<spechls::ArrayType>(arg.getType()))
               ? emitter.emitReference(taskLikeOp->getLoc(), arg.getType())
               : emitter.emitType(taskLikeOp->getLoc(), arg.getType());
  });
}

LogicalResult printAllVariables(CppEmitter &emitter, spechls::KernelOp &kernelOp) {
  raw_ostream &os = emitter.ostream();

  WalkResult result = kernelOp->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    // Skip inlined operations.
    if (isa<circt::hw::ConstantOp>(op) || isa<spechls::FieldOp>(op))
      return WalkResult::advance();

    for (OpResult result : op->getResults()) {
      if (failed(emitter.emitVariableDeclaration(result, false)))
        return WalkResult(op->emitError("unable to declare result variable for op"));
      os << "{};\n";
    }

    if (op != kernelOp.getOperation()) {
      for (auto &&region : op->getRegions()) {
        for (auto &&arg : region.getArguments()) {
          if (failed(
                  emitter.emitVariableDeclaration(op->getLoc(), arg.getType(), emitter.getOrCreateName(arg), false))) {
            return WalkResult(op->emitError("unable to declare region argument variable for op"));
          }
          os << "{};\n";
        }
      }
    }

    // Declare buffers.
    if (auto delayOp = dyn_cast<spechls::DelayOp>(op)) {
      if (failed(emitter.emitType(op->getLoc(), delayOp.getType())))
        return failure();
      os << " " << getDelayBufferName(emitter, delayOp) << "[" << delayOp.getDepth() << "]{};\n";
    } else if (auto rewindOp = dyn_cast<spechls::RewindOp>(op)) {
      if (failed(emitter.emitType(op->getLoc(), rewindOp.getType())))
        return failure();
      os << " " << getRewindBufferName(emitter, rewindOp) << "["
         << *std::max_element(rewindOp.getDepths().begin(), rewindOp.getDepths().end()) + 1 << "]{};\n";
    } else if (auto rollbackOp = dyn_cast<spechls::RollbackOp>(op)) {
      if (failed(emitter.emitType(op->getLoc(), rollbackOp.getType())))
        return failure();
      os << " " << getRollbackBufferName(emitter, rollbackOp) << "["
         << *std::max_element(rollbackOp.getDepths().begin(), rollbackOp.getDepths().end()) + 1 << "]{};\n";
    } else if (auto cancelOp = dyn_cast<spechls::CancelOp>(op)) {
      if (failed(emitter.emitType(op->getLoc(), cancelOp.getType())))
        return failure();
      os << " " << getCancelBufferName(emitter, cancelOp) << "[1]{};\n";
    } else if (auto fifoOp = dyn_cast<spechls::FIFOOp>(op)) {
      os << "FifoType<";
      if (failed(emitter.emitType(op->getLoc(), fifoOp.getType())))
        return failure();
      os << "> " << getFifoBufferName(emitter, fifoOp) << "{};\n";
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  return success();
}

LogicalResult emitStruct(CppEmitter &emitter, Location loc, Type type, DenseSet<StringRef> &generatedStructs) {
  if (auto sType = dyn_cast<spechls::StructType>(type)) {
    for (auto &&field : sType.getFieldTypes()) {
      if (failed(emitStruct(emitter, loc, field, generatedStructs)))
        return failure();
    }
    if (!generatedStructs.contains(sType.getName())) {
      if (failed(emitter.emitStructDefinition(loc, sType)))
        return failure();
      generatedStructs.insert(sType.getName());
    }
  }
  return success();
}

LogicalResult printAllStructTypes(CppEmitter &emitter, ModuleOp moduleOp) {
  DenseSet<StringRef> generatedStructs{};

  WalkResult result = moduleOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    for (Type type : op->getResultTypes()) {
      LogicalResult result = emitStruct(emitter, op->getLoc(), type, generatedStructs);
      if (failed(result))
        return WalkResult(result);
    }
    for (Type type : op->getOperandTypes()) {
      LogicalResult result = emitStruct(emitter, op->getLoc(), type, generatedStructs);
      if (failed(result))
        return WalkResult(result);
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

LogicalResult printAllFunctionPrototypes(CppEmitter &emitter, ModuleOp moduleOp) {
  // NOTE: We assume that we don't rely on function overloading, otherwise we would need to differentiate functions by
  // argument types as well as by name.
  StringSet<> declaredFunctions{};

  WalkResult result = moduleOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (auto callOp = dyn_cast<spechls::CallOp>(op)) {
      if (!declaredFunctions.contains(callOp.getCallee())) {
        SmallVector<Type> operandTypes;
        for (auto &&type : callOp.getOperandTypes())
          operandTypes.push_back(type);
        LogicalResult result = emitter.emitFunctionPrototype(op->getLoc(), callOp.getCallee(), operandTypes,
                                                             callOp.getNumResults() != 0 ? callOp.getType(0) : Type{});
        if (failed(result))
          return WalkResult(result);
        declaredFunctions.insert(callOp.getCallee());
      }
    } else if (auto fsmCommandOp = dyn_cast<spechls::FSMCommandOp>(op)) {
      std::string callee = ("fsm_" + fsmCommandOp.getName() + "_command").str();
      LogicalResult result = emitter.emitFunctionPrototype(op->getLoc(), callee, {fsmCommandOp.getState().getType()},
                                                           fsmCommandOp.getType());
      if (failed(result))
        return WalkResult(result);
      declaredFunctions.insert(callee);
    } else if (auto fsmOp = dyn_cast<spechls::FSMOp>(op)) {
      std::string callee = ("fsm_" + fsmOp.getName() + "_next").str();
      LogicalResult result = emitter.emitFunctionPrototype(
          op->getLoc(), callee, {fsmOp.getMispec().getType(), fsmOp.getState().getType()}, fsmOp.getType());
      if (failed(result))
        return WalkResult(result);
      declaredFunctions.insert(callee);
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

LogicalResult printDelayInitialization(CppEmitter &emitter, Region::BlockListType &blocks) {
  raw_ostream &os = emitter.ostream();

  for (auto &&block : blocks) {
    for (auto &&op : block) {
      if (auto delayOp = dyn_cast<spechls::DelayOp>(op)) {
        if (!delayOp.getInit() || isSelfInitializedDelay(delayOp))
          continue;
        os << "delay_init<";
        if (failed(emitter.emitType(delayOp.getLoc(), delayOp.getType())))
          return failure();
        os << ", " << delayOp.getDepth() << ">(" << getDelayBufferName(emitter, delayOp) << ", ";
        if (failed(emitter.emitOperand(delayOp.getInit())))
          return failure();
        os << ");\n";
      }
    }
  }

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, ModuleOp moduleOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "#include <ap_int.h>\n";
  os << "#include <io_printf.h>\n";
  os << "#include <spechls_support.h>\n";
  os << "\n";

  if (emitter.shouldDeclareStructTypes()) {
    if (failed(printAllStructTypes(emitter, moduleOp)))
      return failure();
    os << "\n";
  }

  if (emitter.shouldDeclareFunctions()) {
    if (failed(printAllFunctionPrototypes(emitter, moduleOp)))
      return failure();
    os << "\n";
  }

  for (auto &&op : moduleOp) {
    if (auto kernelOp = dyn_cast<spechls::KernelOp>(op)) {
      os << "void " << kernelOp.getName() << "(";
      if (failed(printFunctionArgTypes(emitter, kernelOp.getOperation(), kernelOp.getArguments(), true)))
        return failure();
      os << ");\n";
    }
  }
  os << "\n";

  for (auto &&op : moduleOp) {
    if (failed(emitter.emitOperation(op, false)))
      return failure();
  }
  return success();
}

bool topoSortCriteria(Value, Operation *op) { return isa<spechls::MuOp>(op) || isa<spechls::DelayOp>(op); }

LogicalResult printOperation(CppEmitter &emitter, spechls::KernelOp kernelOp) {
  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  Operation *op = kernelOp.getOperation();

  os << "void " << kernelOp.getName() << "(";
  if (failed(printFunctionArgs(emitter, op, kernelOp.getArguments(), true)))
    return failure();
  os << ") {\n#pragma HLS inline recursive\n";
  os.indent();

  if (failed(printAllVariables(emitter, kernelOp)))
    return failure();

  if (failed(printDelayInitialization(emitter, kernelOp.getBody().getBlocks())))
    return failure();

  // Generate the exit variable.
  auto exit = cast<spechls::ExitOp>(kernelOp.getBody().front().getTerminator());
  if (failed(emitter.emitVariableDeclaration(exit.getLoc(), exit.getGuard().getType(), emitter.getExitVariableName(),
                                             false))) {
    return failure();
  }
  os << " = false;\n";

  // Generate the init variable.
  if (failed(emitter.emitVariableDeclaration(kernelOp.getLoc(), IntegerType::get(kernelOp.getContext(), 1),
                                             emitter.getInitVariableName(), false))) {
    return failure();
  }
  os << " = true;\n\n";

  os << "while (!" << emitter.getExitVariableName() << ") {\n";
  os.indent();

  mlir::sortTopologically(&kernelOp.getBody().front(), topoSortCriteria);
  for (auto &&op : kernelOp.getBody().front()) {
    if (failed(emitter.emitOperation(op, true)))
      return failure();
  }

  os << emitter.getInitVariableName() << " = false;\n";

  os.unindent();
  os << "}\n";
  os.unindent();
  os << "}";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::TaskOp taskOp) {
  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  SmallVector<spechls::FIFOOp> fifoInputs;
  SmallVector<std::pair<Value, spechls::FIFOOp>> fifoOutputs;
  for (auto &&op : taskOp.getArgs()) {
    if (auto fifoOp = dyn_cast_if_present<spechls::FIFOOp>(op.getDefiningOp())) {
      fifoInputs.push_back(fifoOp);
    }
  }
  for (auto &&res : taskOp.getResults()) {
    for (auto &&user : res.getUsers()) {
      if (auto fifoOp = dyn_cast_if_present<spechls::FIFOOp>(user)) {
        fifoOutputs.push_back({res, fifoOp});
      }
    }
  }

  if (!fifoInputs.empty() || !fifoOutputs.empty()) {
    os << "if (";
    llvm::interleave(
        fifoInputs.begin(), fifoInputs.end(),
        [&](auto &fifo) { os << "!" << getFifoBufferName(emitter, fifo) << ".empty"; }, [&]() { os << " && "; });
    if (!fifoInputs.empty() && !fifoOutputs.empty())
      os << " && ";
    llvm::interleave(
        fifoOutputs.begin(), fifoOutputs.end(),
        [&](auto &fifo) { os << "!" << getFifoBufferName(emitter, fifo.second) << ".full"; }, [&]() { os << " && "; });
    os << ") {\n";
    os.indent();
  }

  // Copy variables to the local scope.
  for (auto arg : llvm::zip_equal(taskOp.getBody().getArguments(), taskOp.getArgs())) {
    if (failed(emitter.emitOperand(std::get<0>(arg))))
      return failure();
    os << " = ";
    if (failed(emitter.emitOperand(std::get<1>(arg))))
      return failure();
    os << ";\n";
  }

  if (failed(printDelayInitialization(emitter, taskOp.getBody().getBlocks())))
    return failure();

  SmallVector<spechls::DelayOp> delays;
  Value nextInputCmd{};
  mlir::sortTopologically(&taskOp.getBody().front(), topoSortCriteria);
  for (auto &&op : taskOp.getBody().front()) {
    if (isa<spechls::CommitOp>(op)) {
      // Print delay push operations just before the end of the task.
      for (auto &&delay : delays) {
        os << "delay_push<";
        if (failed(emitter.emitType(delay.getLoc(), delay.getType())))
          return failure();
        os << ", " << delay.getDepth() << ">(" << getDelayBufferName(emitter, delay) << ", ";
        if (failed(emitter.emitOperand(delay.getInput())))
          return failure();
        if (delay.getEnable()) {
          os << ", ";
          if (failed(emitter.emitOperand(delay.getEnable())))
            return failure();
        }
        os << ");\n";
      }
    } else if (auto delayOp = dyn_cast<spechls::DelayOp>(op)) {
      delays.push_back(delayOp);
    } else if (auto fieldOp = dyn_cast<spechls::FieldOp>(op)) {
      // Retrieve the nextInput FSM command signal. This is a hack that is mirrored from the Java implementation.
      if (fieldOp.getName() == "nextInput") {
        nextInputCmd = fieldOp;
      }
    }
    if (failed(emitter.emitOperation(op, true))) {
      return failure();
    }
  }

  if (!fifoInputs.empty() || !fifoOutputs.empty()) {
    for (auto &&out : fifoOutputs) {
      os << "fifo_write(" << getFifoBufferName(emitter, out.second) << ", " << emitter.getOrCreateName(out.first)
         << ");\n";
    }
    if (nextInputCmd && !fifoInputs.empty()) {
      os << "if (";
      if (failed(emitter.emitOperand(nextInputCmd)))
        return failure();
      os << ") {\n";
      os.indent();
    }
    for (auto &&in : fifoInputs) {
      os << "fifo_read(" << getFifoBufferName(emitter, in) << ");\n";
    }
    if (nextInputCmd && !fifoInputs.empty()) {
      os.unindent();
      os << "}\n";
    }
    os.unindent();
    os << "}";
  }
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

LogicalResult printOperation(CppEmitter &emitter, spechls::DelayOp delayOp) {
  Operation *operation = delayOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "delay_pop<";
  if (failed(emitter.emitType(delayOp.getLoc(), delayOp.getType())))
    return failure();
  os << ", " << delayOp.getDepth() << ">(" << getDelayBufferName(emitter, delayOp) << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::ExitOp exitOp) {
  raw_ostream &os = emitter.ostream();
  os << emitter.getExitVariableName() << " = ";
  if (failed(emitter.emitOperand(exitOp.getGuard())))
    return failure();
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::MuOp muOp) {
  static int id = 0;

  Operation *operation = muOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "mu<";
  if (failed(emitter.emitType(muOp.getLoc(), muOp.getType())))
    return failure();
  os << ", " << id++ << ">(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::PackOp packOp) {
  Operation *operation = packOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitVariableAssignment(operation->getResult(0))))
    return failure();
  if (failed(emitter.emitType(packOp.getLoc(), packOp.getType())))
    return failure();
  os << "{";
  if (failed(interleaveCommaWithError(llvm::zip(packOp.getType().getFieldTypes(), packOp.getOperands()), os,
                                      [&](auto pair) {
                                        auto [type, value] = pair;
                                        os << "static_cast<";
                                        if (failed(emitter.emitType(packOp.getLoc(), type)))
                                          return failure();
                                        os << ">(";
                                        if (failed(emitter.emitOperand(value)))
                                          return failure();
                                        os << ")";
                                        return success();
                                      }))) {
    return failure();
  }
  os << "}";

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::CallOp callOp) {
  Operation *operation = callOp.getOperation();
  StringRef callee = callOp.getCallee();
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::GammaOp gammaOp) {
  Operation *operation = gammaOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "gamma<";
  if (failed(emitter.emitType(gammaOp.getLoc(), gammaOp.getType())))
    return failure();
  os << ">(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
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
  if (failed(interleaveCommaWithError(printOp.getArgs(), os, [&](Value operand) {
        if (isa<IntegerType>(operand.getType()))
          os << "(int)";
        if (failed(emitter.emitOperand(operand)))
          return failure();
        return success();
      })))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::CommitOp commitOp) {
  raw_indented_ostream &os = emitter.ostream();

  if (!commitOp.getValue())
    return success();

  spechls::TaskOp task = commitOp.getParentOp();
  if (failed(emitter.emitOperand(task.getResult())))
    return failure();
  os << " = ";

  bool alwaysEnabled = false;
  Value enable = commitOp.getEnable();
  if (auto constantOp = dyn_cast<circt::hw::ConstantOp>(enable.getDefiningOp())) {
    // Special case for supernodes with always-enabled outputs (i.e., non-speculative supernodes).
    alwaysEnabled = constantOp.getValue().getBoolValue();
  }

  if (alwaysEnabled) {
    if (failed(emitter.emitOperand(commitOp.getValue())))
      return failure();
    return success();
  }

  if (failed(emitter.emitOperand(enable)))
    return failure();
  os << " ? ";
  if (failed(emitter.emitOperand(commitOp.getValue())))
    return failure();
  os << " : ";
  if (failed(emitter.emitType(commitOp.getLoc(), commitOp.getValue().getType())))
    return failure();
  os << "{}";

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::FSMCommandOp fsmCommandOp) {
  Operation *operation = fsmCommandOp.getOperation();
  std::string callee = ("fsm_" + fsmCommandOp.getName() + "_command").str();
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::FSMOp fsmOp) {
  Operation *operation = fsmOp.getOperation();
  std::string callee = ("fsm_" + fsmOp.getName() + "_next").str();
  return printCallOp(emitter, operation, callee);
}

LogicalResult printOperation(CppEmitter &emitter, spechls::RewindOp rewindOp) {
  Operation *operation = rewindOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "rewind<";
  if (failed(emitter.emitType(rewindOp.getLoc(), rewindOp.getType())))
    return failure();
  os << ", ";
  if (failed(interleaveCommaWithError(rewindOp.getDepths(), os, [&](uint64_t depth) {
        os << depth;
        return success();
      }))) {
    return failure();
  }
  os << ">(" << getRewindBufferName(emitter, rewindOp) << ", ";
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
  if (failed(emitter.emitType(rollbackOp.getLoc(), rollbackOp.getType())))
    return failure();
  os << ", " << rollbackOp.getOffset() << ", ";
  if (failed(interleaveCommaWithError(rollbackOp.getDepths(), os, [&](uint64_t depth) {
        os << depth;
        return success();
      }))) {
    return failure();
  }
  os << ">(" << getRollbackBufferName(emitter, rollbackOp) << ", ";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::CancelOp cancelOp) {
  Operation *operation = cancelOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "cancel<";
  os << cancelOp.getOffset();
  os << ">(" << getCancelBufferName(emitter, cancelOp) << ", ";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::FIFOOp fifoOp) {
  Operation *operation = fifoOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << getFifoBufferName(emitter, fifoOp) << ".data";
  return success();
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
  os << "[0 <= ";
  if (failed(emitter.emitOperand(loadOp.getIndex())))
    return failure();
  os << " && ";
  if (failed(emitter.emitOperand(loadOp.getIndex())))
    return failure();
  os << " < " << loadOp.getArray().getType().getSize() << " ? (int)";
  if (failed(emitter.emitOperand(loadOp.getIndex())))
    return failure();
  os << " : 0]";

  return success();
}

LogicalResult printOperation(CppEmitter &emitter, spechls::LUTOp lutOp) {
  Operation *operation = lutOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "lut<";
  if (failed(emitter.emitType(lutOp.getLoc(), lutOp.getType())))
    return failure();
  os << ", ";
  if (failed(emitter.emitType(lutOp.getLoc(), lutOp.getIndex().getType())))
    return failure();
  os << ", ";
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

Type makeSigned(Type t) {
  if (auto iType = dyn_cast<IntegerType>(t)) {
    if (iType.isSigned())
      return t;
    return IntegerType::get(t.getContext(), t.getIntOrFloatBitWidth(), mlir::IntegerType::Signed);
  }
  return t;
}

LogicalResult printBinaryOperation(CppEmitter &emitter, Operation *operation, StringRef binaryOperator,
                                   bool isSigned = false) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  if (isSigned) {
    os << "(";
    if (failed(emitter.emitType(operation->getLoc(), makeSigned(operation->getOperand(0).getType()))))
      return failure();
    os << ")";
  }
  if (failed(emitter.emitOperand(operation->getOperand(0))))
    return failure();
  os << ' ' << binaryOperator << ' ';
  if (isSigned) {
    os << "(";
    if (failed(emitter.emitType(operation->getLoc(), makeSigned(operation->getOperand(0).getType()))))
      return failure();
    os << ")";
  }
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

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ModSOp modsOp) {
  Operation *operation = modsOp.getOperation();
  return printBinaryOperation(emitter, operation, "%", true);
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ModUOp moduOp) {
  Operation *operation = moduOp.getOperation();
  return printBinaryOperation(emitter, operation, "%", false);
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::MulOp mulOp) {
  Operation *operation = mulOp.getOperation();
  return printBinaryOperation(emitter, operation, "*");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::OrOp orOp) {
  Operation *operation = orOp.getOperation();
  return printBinaryOperation(emitter, operation, "|");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ShlOp shlOp) {
  Operation *operation = shlOp.getOperation();
  return printBinaryOperation(emitter, operation, "<<");
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ShrSOp shrsOp) {
  Operation *operation = shrsOp.getOperation();
  return printBinaryOperation(emitter, operation, ">>", true);
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ShrUOp shruOp) {
  Operation *operation = shruOp.getOperation();
  return printBinaryOperation(emitter, operation, ">>", false);
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
  bool isSigned = false;
  switch (icmpOp.getPredicate()) {
  case circt::comb::ICmpPredicate::eq:
    comparisonOperator = "==";
    break;
  case circt::comb::ICmpPredicate::ne:
    comparisonOperator = "!=";
    break;
  case circt::comb::ICmpPredicate::slt:
    isSigned = true;
    [[fallthrough]];
  case circt::comb::ICmpPredicate::ult:
    comparisonOperator = "<";
    break;
  case circt::comb::ICmpPredicate::sle:
    isSigned = true;
    [[fallthrough]];
  case circt::comb::ICmpPredicate::ule:
    comparisonOperator = "<=";
    break;
  case circt::comb::ICmpPredicate::sgt:
    isSigned = true;
    [[fallthrough]];
  case circt::comb::ICmpPredicate::ugt:
    comparisonOperator = ">";
    break;
  case circt::comb::ICmpPredicate::sge:
    isSigned = true;
    [[fallthrough]];
  case circt::comb::ICmpPredicate::uge:
    comparisonOperator = ">=";
    break;
  default:
    return operation->emitOpError("Unexpected icmp predicate ")
           << circt::comb::stringifyICmpPredicate(icmpOp.getPredicate());
  }
  return printBinaryOperation(emitter, operation, comparisonOperator, isSigned);
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
  if (failed(emitter.emitType(muxOp.getLoc(), muxOp.getType())))
    return failure();
  os << "{";
  if (failed(emitter.emitOperand(muxOp.getTrueValue())))
    return failure();
  os << "} : ";
  if (failed(emitter.emitType(muxOp.getLoc(), muxOp.getType())))
    return failure();
  os << "{";
  if (failed(emitter.emitOperand(muxOp.getFalseValue())))
    return failure();
  os << "}";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ExtractOp extractOp) {
  Operation *operation = extractOp.getOperation();
  raw_ostream &os = emitter.ostream();

  uint32_t lowBit = extractOp.getLowBit();
  uint32_t highBit = lowBit + extractOp.getType().getIntOrFloatBitWidth();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "extract<";
  if (failed(emitter.emitType(extractOp.getLoc(), extractOp.getInput().getType())))
    return failure();
  os << ", " << lowBit << ", " << highBit << ">(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, circt::comb::ConcatOp concatOp) {
  Operation *operation = concatOp.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "concat<";
  interleaveComma(concatOp.getOperands(), os, [&](Value operand) { os << operand.getType().getIntOrFloatBitWidth(); });
  os << ">(";
  if (failed(emitter.emitOperands(*operation)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, circt::hw::BitcastOp bitcastOp) {
  Operation *operation = bitcastOp.getOperation();
  raw_ostream &os = emitter.ostream();

  // FIXME: This isn't a bitcast in most cases, but should cover our needs for now.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << "(";
  if (failed(emitter.emitType(bitcastOp.getLoc(), bitcastOp.getType())))
    return failure();
  os << ")";
  if (failed(emitter.emitOperand(bitcastOp.getInput())))
    return failure();
  return success();
}
} // namespace

CppEmitter::CppEmitter(raw_ostream &os, bool declareStructTypes, bool declareFunctions)
    : os(os), declareStructTypes(declareStructTypes), declareFunctions(declareFunctions) {
  valueInScopeCount.push(0);
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  bool skipLineEnding = false;

  LogicalResult status =
      TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // Comb ops.
          .Case<circt::comb::AddOp, circt::comb::AndOp, circt::comb::ConcatOp, circt::comb::DivSOp, circt::comb::DivUOp,
                circt::comb::ExtractOp, circt::comb::ICmpOp, circt::comb::ModSOp, circt::comb::ModUOp,
                circt::comb::MulOp, circt::comb::MuxOp, circt::comb::OrOp, circt::comb::ShlOp, circt::comb::ShrSOp,
                circt::comb::ShrUOp, circt::comb::SubOp, circt::comb::ReplicateOp, circt::comb::XorOp>(
              [&](auto op) { return printOperation(*this, op); })
          // HW ops.
          .Case<circt::hw::BitcastOp>([&](auto op) { return printOperation(*this, op); })
          // SpecHLS ops.
          .Case<spechls::AlphaOp, spechls::CallOp, spechls::CancelOp, spechls::CommitOp, spechls::DelayOp,
                spechls::ExitOp, spechls::FIFOOp, spechls::FSMCommandOp, spechls::FSMOp, spechls::GammaOp,
                spechls::KernelOp, spechls::TaskOp, spechls::LoadOp, spechls::LUTOp, spechls::MuOp, spechls::PackOp,
                spechls::PrintOp, spechls::RewindOp, spechls::RollbackOp, spechls::UnpackOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Inlined operations.
          .Case<circt::hw::ConstantOp, spechls::FieldOp, spechls::SyncOp>([&](auto op) {
            skipLineEnding = true;
            return success();
          })
          .Default([&](Operation *) { return op.emitOpError("unable to find printer for op"); });

  if (failed(status))
    return failure();

  if (!skipLineEnding)
    os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitOperand(Value value) {
  Operation *op = value.getDefiningOp();
  if (op) {
    if (auto constantOp = dyn_cast<circt::hw::ConstantOp>(op))
      return emitAttribute(op->getLoc(), constantOp.getValueAttr());
    if (auto fieldOp = dyn_cast<spechls::FieldOp>(op)) {
      if (failed(emitOperand(fieldOp.getInput())))
        return failure();
      os << "." << fieldOp.getName();
      return success();
    }
    if (auto syncOp = dyn_cast<spechls::SyncOp>(op)) {
      return emitOperand(syncOp.getOperand(0));
    }
  }
  os << getOrCreateName(value);
  return success();
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  return interleaveCommaWithError(op.getOperands(), os, [&](Value operand) { return emitOperand(operand); });
}

LogicalResult CppEmitter::emitType(Location loc, Type type) {
  if (!type) {
    os << "void";
    return success();
  }
  if (auto iType = dyn_cast<IntegerType>(type)) {
    if (shouldMapToUnsigned(iType.getSignedness())) {
      if (iType.getWidth() == 1)
        os << "bool";
      else if (iType.getWidth() == 8)
        os << "unsigned char";
      else if (iType.getWidth() == 16)
        os << "unsigned short";
      else if (iType.getWidth() == 32)
        os << "unsigned int";
      else if (iType.getWidth() == 64)
        os << "unsigned long long";
      else
        os << "ap_uint<" << iType.getWidth() << ">";
    } else {
      // Technically, char is not guaranteed to be a signed type, but it should not matter for our use case.
      if (iType.getWidth() == 1)
        os << "bool";
      else if (iType.getWidth() == 8)
        os << "char";
      else if (iType.getWidth() == 16)
        os << "short";
      else if (iType.getWidth() == 32)
        os << "int";
      else if (iType.getWidth() == 64)
        os << "long long";
      else
        os << "ap_int<" << iType.getWidth() << ">";
    }
    return success();
  }
  if (isa<Float32Type>(type)) {
    os << "float";
    return success();
  }
  if (isa<Float64Type>(type)) {
    os << "double";
    return success();
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

LogicalResult CppEmitter::emitReference(Location loc, Type type) {
  if (failed(emitType(loc, type)))
    return failure();
  os << " &";
  return success();
}

LogicalResult CppEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return failure();
  }
}

LogicalResult CppEmitter::emitStructDefinition(Location loc, spechls::StructType structType) {
  os << "#ifndef DEFINED_" << structType.getName() << "\n";
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
  os << "};\n#endif\n";
  return success();
}

LogicalResult CppEmitter::emitFunctionPrototype(Location loc, StringRef callee, ArrayRef<Type> argumentTypes,
                                                Type returnType) {
  if (failed(emitType(loc, returnType)))
    return failure();
  os << " " << callee << "(";
  if (failed(interleaveCommaWithError(argumentTypes, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ");\n";
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
  case IntegerType::Signed:
    return false;
  case IntegerType::Signless:
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

std::string CppEmitter::getValueNamePrefix(Value value) {
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
  // SpecHLS ops.
  if (isa<spechls::AlphaOp>(op))
    return "alpha";
  if (auto call = dyn_cast<spechls::CallOp>(op))
    return call.getCallee().str();
  if (isa<spechls::DelayOp>(op))
    return "delay";
  if (auto field = dyn_cast<spechls::FieldOp>(op))
    return field.getName().str();
  if (isa<spechls::FIFOOp>(op))
    return "fifo";
  if (auto fsmOp = dyn_cast<spechls::FSMOp>(op))
    return ("fsm_" + fsmOp.getName() + "_next").str();
  if (auto fsmCommandOp = dyn_cast<spechls::FSMCommandOp>(op))
    return ("fsm_" + fsmCommandOp.getName() + "_command").str();
  if (auto gammaOp = dyn_cast<spechls::GammaOp>(op))
    return gammaOp.getSymName().str();
  if (isa<spechls::LoadOp>(op))
    return "load";
  if (isa<spechls::LUTOp>(op))
    return "lut";
  if (auto muOp = dyn_cast<spechls::MuOp>(op))
    return muOp.getSymName().str();
  if (isa<spechls::PrintOp>(op))
    return "print_state";
  if (isa<spechls::RewindOp>(op))
    return "rewind";
  if (isa<spechls::RollbackOp>(op))
    return "rollback";
  if (isa<spechls::CancelOp>(op))
    return "cancel";
  return "v";
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result, bool trailingSemicolon) {
  if (hasValueInScope(result))
    return result.getDefiningOp()->emitError("result variable for the operation already declared");

  return emitVariableDeclaration(result.getOwner()->getLoc(), result.getType(), getOrCreateName(result),
                                 trailingSemicolon);
}

LogicalResult CppEmitter::emitVariableDeclaration(Location loc, Type type, StringRef name, bool trailingSemicolon,
                                                  bool isReference) {
  if (failed(emitType(loc, type)))
    return failure();
  os << " ";
  if (isReference)
    os << "&";
  os << name;
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
    return op.emitOpError("unexpected operation with multiple results");
  }
  return success();
}

StringRef CppEmitter::getOrCreateName(Value value) {
  if (!valueMapper.count(value)) {
    valueMapper.insert(value, llvm::formatv("{0}_{1}", getValueNamePrefix(value), ++valueInScopeCount.top()));
  }
  return *valueMapper.begin(value);
}

LogicalResult spechls::translateToCpp(Operation *op, raw_ostream &os, bool declareStructTypes, bool declareFunctions) {
  CppEmitter emitter(os, declareStructTypes, declareFunctions);
  return emitter.emitOperation(*op, false);
}
