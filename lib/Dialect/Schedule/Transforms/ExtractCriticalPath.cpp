//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep

#include <circt/Dialect/SSP/SSPAttributes.h>
#include <circt/Dialect/SSP/SSPOps.h>
#include <circt/Dialect/SSP/SSPPasses.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>

using namespace mlir;

namespace schedule {
#define GEN_PASS_DEF_EXTRACTCRITICALPATHPASS
#include "Dialect/Schedule/Transforms/Passes.h.inc"
} // namespace schedule

namespace {

class ExtractCriticalPathPass : public schedule::impl::ExtractCriticalPathPassBase<ExtractCriticalPathPass> {
  int extractII(ModuleOp &moduleOp);

public:
  using ExtractCriticalPathPassBase::ExtractCriticalPathPassBase;
  void runOnOperation() override;
};

} // namespace

int ExtractCriticalPathPass::extractII(ModuleOp &moduleOp) {
  PassManager schedulePM(moduleOp.getContext());
  auto schedulePass = schedule::createSchedulePass();
  schedulePM.addPass(std::move(schedulePass));

  if (failed(schedulePM.run(moduleOp))) {
    return -1;
  }
  auto &instanceOp = moduleOp.getBody()->getOperations().front();
  if (!instanceOp.hasAttrOfType<mlir::IntegerAttr>("spechls.ii")) {
    llvm::errs() << "No II after scheduling.\n";
    instanceOp.dump();
    return -1;
  }
  int ii = instanceOp.getAttrOfType<mlir::IntegerAttr>("spechls.ii").getInt();
  instanceOp.removeAttr("spechls.ii");
  return ii;
}

void ExtractCriticalPathPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto *ctx = moduleOp->getContext();

  // Compute the initial II.
  int initII = extractII(moduleOp);
  if (initII == -1)
    return signalPassFailure();
  if (initII == 1)
    return;

  // Remove every operand that does not modify the II.
  if (auto instanceOp = llvm::dyn_cast<circt::ssp::InstanceOp>(moduleOp.getBody()->getOperations().front())) {
    circt::ssp::DependenceGraphOp graphOp;
    circt::ssp::OperatorLibraryOp libraryOp;
    for (auto &sspOp : instanceOp.getBodyBlock()->getOperations()) {
      if (auto graph = llvm::dyn_cast<circt::ssp::DependenceGraphOp>(sspOp)) {
        graphOp = graph;
      } else if (auto library = llvm::dyn_cast<circt::ssp::OperatorLibraryOp>(sspOp)) {
        libraryOp = library;
      }
    }

    llvm::SmallVector<mlir::Operation *> toRemove;
    size_t numOperations = graphOp.getBodyBlock()->getOperations().size();
    for (size_t i = 0; i < numOperations; ++i) {
      auto it = graphOp.getBodyBlock()->getOperations().begin();
      std::advance(it, i);
      auto operationOp = llvm::dyn_cast<circt::ssp::OperationOp>(*it);

      auto dependencesOpt = operationOp.getDependences();
      llvm::SmallVector<mlir::Attribute> deps;
      if (dependencesOpt) {
        deps = llvm::SmallVector<mlir::Attribute>((*dependencesOpt).getAsRange<mlir::Attribute>());
        operationOp.removeDependencesAttr();
      }

      unsigned int numOperands = operationOp->getNumOperands();
      bool found = false;
      llvm::SmallVector<mlir::Value> originalOperands(operationOp->getOperands());

      operationOp.getOperandsMutable().clear();

      auto cloned = llvm::cast<ModuleOp>(moduleOp->clone());
      int newII = extractII(cloned);
      if (newII == -1)
        return signalPassFailure();
      cloned->erase();
      if (newII == initII) {
        found = true;
        toRemove.push_back(operationOp);
      } else {
        for (unsigned int i = 0; !found && (i < numOperands); ++i) {
          operationOp.getOperandsMutable().clear();
          operationOp.getOperandsMutable().append(originalOperands[i]);
          auto cloned = llvm::cast<ModuleOp>(moduleOp->clone());
          int newII = extractII(cloned);
          if (newII == -1)
            return signalPassFailure();
          cloned->erase();
          if (newII == initII) {
            found = true;
            break;
          }
        }
        if (!found) {
          operationOp.getOperandsMutable().clear();
        }

        unsigned int numDeps = deps.size();
        for (unsigned int i = 0; !found && (i < numDeps); ++i) {
          llvm::SmallVector<mlir::Attribute, 1> newDeps;
          auto dependence = llvm::dyn_cast<circt::ssp::DependenceAttr>(deps[i]);
          newDeps.push_back(
              circt::ssp::DependenceAttr::get(ctx, 0, dependence.getSourceRef(), dependence.getProperties()));
          operationOp.setDependencesAttr(mlir::ArrayAttr::get(ctx, newDeps));
          auto cloned = llvm::cast<ModuleOp>(moduleOp->clone());
          int newII = extractII(cloned);
          if (newII == -1)
            return signalPassFailure();
          cloned->erase();
          if (newII == initII) {
            found = true;
            break;
          }
        }
        if (!found) {
          operationOp.setDependencesAttr(mlir::ArrayAttr::get(ctx, llvm::SmallVector<mlir::Attribute>()));
          toRemove.push_back(operationOp);
        }
      }
    }
    for (auto *op : toRemove) {
      op->erase();
    }

    // gather the remaining the operators and erase the unused ones.
    llvm::DenseSet<circt::ssp::OperatorTypeOp> operators;
    graphOp.getBody().walk([&](circt::ssp::OperationOp op) {
      operators.insert(llvm::dyn_cast<circt::ssp::OperatorTypeOp>(
          SymbolTable::lookupSymbolIn(libraryOp, op.getLinkedOperatorTypeAttr().getValue())));
    });

    for (auto &sspOp : instanceOp.getBodyBlock()->getOperations()) {
      if (auto library = llvm::dyn_cast<circt::ssp::OperatorLibraryOp>(sspOp)) {
        llvm::SmallVector<circt::ssp::OperatorTypeOp> toRemove;
        library.getBodyBlock()->walk([&](circt::ssp::OperatorTypeOp op) {
          if (!operators.contains(op)) {
            op->erase();
          }
        });
      }
    }
  } else {
    llvm::errs() << "extract schedule pass should be called on a ssp instanceOp.\n";
    return signalPassFailure();
  }
}
