//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Target/Cpp/Export.h"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/IndentedOstream.h>

#include <stack>

using namespace mlir;

namespace {
class CppEmitter {
  raw_indented_ostream os;

public:
  explicit CppEmitter(raw_ostream &os);

  raw_indented_ostream &ostream() { return os; }

  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  LogicalResult emitType(Location loc, Type type);
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  LogicalResult emitVariableDeclaration(Location loc, Type type, StringRef name);

  StringRef getOrCreateName(Value value);

  class Scope {
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    CppEmitter &emitter;

  public:
    explicit Scope(CppEmitter &emitter) : valueMapperScope(emitter.valueMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
    }
    ~Scope() { emitter.valueInScopeCount.pop(); }
  };

private:
  static bool shouldMapToUnsigned(IntegerType::SignednessSemantics semantics);

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  ValueMapper valueMapper;

  std::stack<int64_t> valueInScopeCount;
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
    return emitter.emitVariableDeclaration(taskLikeOp->getLoc(), arg.getType(), emitter.getOrCreateName(arg));
  });
}

LogicalResult printFunctionBody(CppEmitter &emitter, Operation *taskLikeOp, Region::BlockListType &blocks) {
  raw_indented_ostream &os = emitter.ostream();
  os.indent();

  // TODO
  os << "TODO\n";

  os.unindent();
  return success();
}

LogicalResult printOperation(CppEmitter &emitter, ModuleOp moduleOp) {
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
  if (failed(printFunctionBody(emitter, operation, hkernelOp.getBlocks())))
    return failure();
  os << "}";
  return success();
}
} // namespace

CppEmitter::CppEmitter(raw_ostream &os) : os(os) { valueInScopeCount.push(0); }

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  LogicalResult status = TypeSwitch<Operation *, LogicalResult>(&op)
                             // Builtin ops.
                             .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
                             // SpecHLS ops.
                             .Case<spechls::HKernelOp>([&](auto op) { return printOperation(*this, op); })
                             .Default([&](Operation *) { return op.emitOpError("unable to find printer for op"); });

  if (failed(status))
    return failure();

  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      os << "bool";
      return success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        os << "uint" << iType.getWidth() << "_t";
      else
        os << "int" << iType.getWidth() << "_t";
      return success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto tType = dyn_cast<TupleType>(type)) {
    return emitTupleType(loc, tType.getTypes());
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

LogicalResult CppEmitter::emitVariableDeclaration(Location loc, Type type, StringRef name) {
  if (failed(emitType(loc, type)))
    return failure();
  os << ' ' << name;
  return success();
}

StringRef CppEmitter::getOrCreateName(Value value) {
  if (!valueMapper.count(value))
    valueMapper.insert(value, llvm::formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(value);
}

LogicalResult spechls::translateToCpp(Operation *op, raw_ostream &os) {
  CppEmitter emitter(os);
  return emitter.emitOperation(*op, false);
}
