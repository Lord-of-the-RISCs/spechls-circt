//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_COMMON_TRANSITIVE_CLOSURE_H
#define SPECHLS_INCLUDED_COMMON_TRANSITIVE_CLOSURE_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/BitVector.h"

#include <functional>
#include <vector>

using namespace mlir;

class PathAnalyzer {
public:
  PathAnalyzer(Region &region, bool intraIteration = true);
  llvm::SmallVector<mlir::Operation *>
  getTransitiveSuccessors(mlir::Operation *op);
  llvm::SmallVector<mlir::Operation *>
  getTransitivePredecessors(mlir::Operation *op);
  bool isTransitiveSuccessor(mlir::Operation *op, mlir::Operation *succ);
  bool isTransitivePredecessor(mlir::Operation *op, mlir::Operation *succ);

private:
  void mapOperations();

  void buildAdjacencyMatrix(std::function<bool(mlir::Operation *)> filter);

  void computeTransitiveClosure();

private:
  llvm::DenseMap<mlir::Operation *, size_t> opToIndexMap;
  llvm::SmallVector<mlir::Operation *> indexToOp;
  std::vector<llvm::BitVector> *adjMatrix;
  std::vector<llvm::BitVector> *forwardAdjMatrix;
  std::vector<llvm::BitVector> *backwardAdjMatrix;

  Region &region;
};

#endif // SPECHLS_INCLUDED_COMMON_TRANSITIVE_CLOSURE_H
