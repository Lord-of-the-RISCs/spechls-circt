//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Analysis/Analysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include <queue>

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
namespace SpecHLS {

// Define a pass to find the corresponding hwOp from a CallOp
circt::hw::HWModuleOp *findHWModuleForInstanceOp(Operation op) {
  circt::hw::HWModuleOp *res;
  op.walk([&res](circt::hw::InstanceOp callOp) {
    // Get the callee symbol reference
    FlatSymbolRefAttr calleeSymbolRef = callOp.getModuleNameAttr();

    // Find the operation with the given symbol reference in the parent
    // operations
    Operation *symbolDefOp =
        SymbolTable::lookupNearestSymbolFrom(callOp, calleeSymbolRef);

    // Check if the found operation is a hwOp
    if (circt::hw::HWModuleOp hwOp =
            dyn_cast<circt::hw::HWModuleOp>(symbolDefOp)) {
      // Do something with the corresponding hwOp
      llvm::errs() << "Found corresponding HWModuleOp: " << hwOp.getName()
                   << "\n";
      res = &hwOp;
    } else {
      llvm::errs() << "Did not find corresponding HWModuleOp for CallOp.\n";
      res = NULL;
    }
  });
}

std::string getPragma(Operation *op) {
  auto attr = op->getAttr(StringRef("#pragma"));
  if (attr != NULL) {
    if (auto strAttr = attr.dyn_cast<mlir::StringAttr>()) {
      return strAttr.getValue().str();
    }
  }
  return NULL;
}

bool isControlLogicOperation(Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<circt::comb::AddOp>([&](auto op) { return true; })
      .Case<circt::comb::SubOp>([&](auto op) { return true; })
      .Case<circt::comb::AndOp>([&](auto op) { return true; })
      .Case<circt::comb::ShlOp>([&](auto op) { return true; })
      .Case<circt::comb::ShrUOp>([&](auto op) { return true; })
      .Case<circt::comb::ReplicateOp>([&](auto op) { return true; })
      .Case<circt::comb::OrOp>([&](auto op) { return true; })
      .Case<circt::comb::TruthTableOp>([&](auto op) { return true; })
      .Case<circt::comb::ExtractOp>([&](auto op) { return true; })
      .Case<circt::hw::OutputOp>([&](auto op) { return true; })
      .Case<circt::hw::ConstantOp>([&](auto op) { return true; })
      .Case<circt::comb::MuxOp>([&](auto op) { return true; })
      .Case<circt::comb::XorOp>([&](auto op) { return true; })
      .Case<circt::comb::ConcatOp>([&](auto op) { return true; })
      .Case<circt::comb::ICmpOp>([&](auto op) { return true; })
      .Default([&](auto op) {
        llvm::errs() << "Operation " << *op << "is not synthesizable\n";
        return false;
      });
}

bool hasControlNodePragma(Operation *op) {

  auto attr = op->getAttr(StringRef("#pragma"));
  if (attr != NULL) {
    if (auto strAttr = attr.dyn_cast<mlir::StringAttr>()) {
      // Compare the attribute value with an existing string
      llvm::StringRef existingString = "CONTROL_NODE";
      if (strAttr.getValue().contains(existingString)) {
        return true;
      }
    }
  }
  return false;
}

bool hasPragmaContaining(Operation *op, llvm::StringRef keyword) {

  auto attr = op->getAttr(StringRef("#pragma"));
  if (attr != NULL) {
    if (auto strAttr = attr.dyn_cast<mlir::StringAttr>()) {
      // Compare the attribute value with an existing string
      if (strAttr.getValue().contains(keyword)) {
        return true;
      }
    }
  }
  return false;
}

bool hasConstantOutputs(circt::hw::HWModuleOp op) {
  auto block = op.getBodyBlock();
  if (block) {

    if (block->getOperations().size() <= 2)
      return true;

    for (auto &_innerop : block->getOperations()) {
      bool ok = TypeSwitch<Operation *, bool>(&_innerop)
                    .Case<circt::hw::ConstantOp>([&](auto op) { return true; })
                    .Case<circt::hw::OutputOp>([&](auto op) { return true; })
                    .Default([&](auto op) {
                      // llvm::outs() << "Operation " << _innerop << "is not
                      // constant\n";
                      return false;
                    });

      if (!ok)
        return false;
    }
    return true;
  }
}

// Helper function to compute the transitive closure of def/use relationships.
llvm::DenseMap<Operation *, llvm::BitVector>
computeTransitiveClosure(Operation *funcOp) {

  llvm::DenseMap<Operation *, llvm::BitVector> defUseMap;
  llvm::DenseMap<Operation *, u_int64_t> opMap;
  u_int64_t cnt = 0;
  funcOp->walk([&](Operation *op) { defUseMap[op].set(cnt++); });

  // Populate the initial def/use relationships.
  funcOp->walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        u_int64_t id = opMap[user];
        defUseMap[op].set(id);
      }
    }
  });

  // Perform a breadth-first search to compute the transitive closure.
  std::queue<Operation *> worklist;
  for (auto &entry : defUseMap) {
    worklist.push(entry.first);
  }

  while (!worklist.empty()) {
    Operation *current = worklist.front();
    worklist.pop();
    u_int64_t currentId = opMap[current];

    for (auto &entry : defUseMap) {
      u_int64_t entryId = opMap[entry.first];
      if (entry.second.test(currentId)) {
        // If there is a connection from the current operation to entry.first,
        // update the transitive closure information.
        if (!defUseMap[current].test(entryId)) {
          defUseMap[current].set(entryId);
          worklist.push(current);
        }
      }
    }
  }

  return defUseMap;
}

// Helper function to compute the transitive closure of def/use relationships.
llvm::DenseMap<Operation *, llvm::BitVector>
computeReverseTransitiveClosure(Operation *funcOp) {

  llvm::DenseMap<Operation *, llvm::BitVector> useDefMap;
  llvm::DenseMap<Operation *, u_int64_t> opMap;
  u_int64_t cnt = 0;
  funcOp->walk([&](Operation *op) { useDefMap[op].set(cnt++); });

  // Populate the initial def/use relationships.
  funcOp->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *user = operand.getDefiningOp();
      u_int64_t id = opMap[user];
      useDefMap[op].set(id);
    }
  });

  // Perform a breadth-first search to compute the transitive closure.
  std::queue<Operation *> worklist;
  for (auto &entry : useDefMap) {
    worklist.push(entry.first);
  }

  while (!worklist.empty()) {
    Operation *current = worklist.front();
    worklist.pop();
    u_int64_t currentId = opMap[current];

    for (auto &entry : useDefMap) {
      u_int64_t entryId = opMap[entry.first];
      if (entry.second.test(currentId)) {
        // If there is a connection from the current operation to entry.first,
        // update the transitive closure information.
        if (!useDefMap[current].test(entryId)) {
          useDefMap[current].set(entryId);
          worklist.push(current);
        }
      }
    }
  }

  return useDefMap;
}

void setPragmaAttr(Operation *op, StringAttr value) {
  op->setAttr("#pragma", value);
}

void removePragmaAttr(Operation *op, llvm::StringRef value) {
  auto attr = op->getAttr(llvm::StringRef("#pragma"));
  if (attr == NULL)
    return;

  auto strAttr = attr.dyn_cast<StringAttr>();
  if (!strAttr)
    return;

  if (strAttr.getValue().contains(value)) {
    op->removeAttr(llvm::StringRef("#pragma"));
    return;
  }
  return;
}
} // namespace SpecHLS
