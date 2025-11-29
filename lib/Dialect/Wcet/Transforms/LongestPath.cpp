//===- UnrollInstr.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the UnrollInstr pass
//
//===----------------------------------------------------------------------===//

#include <climits>
#include <cstdio>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <stack>

#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_LONGESTPATHPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace {

void topologicalSort(Operation *op, std::stack<Value> &stack, DenseMap<Operation *, bool> &visited);

void longestPath(wcet::DummyOp startingPoint, PatternRewriter &rewriter, StringAttr distName);

void visitedOps(Operation *op, SmallVector<Operation *> &visited);
} // namespace

namespace wcet {

struct LongestPathPass : public impl::LongestPathPassBase<LongestPathPass> {

public:
  void runOnOperation() override {
    auto top = getOperation();
    wcet::CoreOp core = nullptr;
    top->walk([&](wcet::CoreOp c) {
      if (c->hasAttr("wcet.analysis")) {
        core = c;
        return;
      }
    });

    wcet::DummyOp startPoint = nullptr;
    wcet::DummyOp endPoint = nullptr;
    core->walk([&](wcet::DummyOp dum) {
      if (dum->hasAttr("wcet.current")) {
        startPoint = dum;
        return;
      }
      if (dum->hasAttr("wcet.next")) {
        endPoint = dum;
        return;
      }
    });
    if (!startPoint || !endPoint) {
      signalPassFailure();
      return;
    }

    auto *ctx = &getContext();
    PatternRewriter rewriter(ctx);

    longestPath(startPoint, rewriter, rewriter.getStringAttr("wcet.dists"));

    startPoint->removeAttr("wcet.current");
    endPoint->setAttr("wcet.penalties", endPoint->getAttr("wcet.dists"));
  }
};

} // namespace wcet

namespace {
void topologicalSort(Operation *op, std::stack<Value> &stack, DenseMap<Operation *, bool> &visited) {
  visited[op] = true;

  for (Operation *use : op->getUsers()) {
    if (!visited[use]) {
      topologicalSort(use, stack, visited);
    }
  }
  for (size_t i = 0; i < op->getNumResults(); i++) {
    stack.push(op->getResult(i));
  }
}

void longestPath(wcet::DummyOp startingPoint, PatternRewriter &rewriter, StringAttr distName) {
  DenseMap<Operation *, int> dists;
  DenseMap<Operation *, bool> visited;
  std::stack<Value> stack;
  SmallVector<Operation *> ops;
  visitedOps(startingPoint, ops);
  for (Operation *op : ops) {
    dists[op] = INT_MIN;
    visited[op] = false;
  }
  for (Operation *op : ops) {
    if (!visited[op]) {
      ::topologicalSort(op, stack, visited);
    }
  }
  dists[startingPoint] = 0;
  while (!stack.empty()) {
    Value v = stack.top();
    stack.pop();

    // Get the delay of the Operation
    int delay = 0;
    wcet::PenaltyOp delta = v.getDefiningOp<wcet::PenaltyOp>();
    if (delta) {
      delay = delta.getDepth();
    } else {
      auto *unknown = v.getDefiningOp();
      if (unknown->hasAttr("wcet.delay")) {
        delay = cast<IntegerAttr>(unknown->getAttr("wcet.delay")).getInt();
      }
    }

    // Get the attribute
    Operation *op = v.getDefiningOp();
    auto dist = dists[op];

    // Check attribute's Value
    if (dist == INT_MIN) {
      continue;
    }

    // Save all uses & update all uses' value
    for (Operation *use : v.getUsers()) {
      auto useD = dists[use];
      if (useD < (dist + delay)) {
        dists[use] = dist + delay;
      }
    }
  }
  int maxDist = 0;
  for (Operation *op : ops) {
    if (dists[op] > maxDist)
      maxDist = dists[op];
    op->setAttr(distName, rewriter.getI32IntegerAttr(dists[op]));
  }
}

void visitedOps(Operation *op, SmallVector<Operation *> &visited) {
  if (std::find(visited.begin(), visited.end(), op) != visited.end()) {
    return;
  }

  visited.push_back(op);

  if (op->getName().getStringRef() == wcet::DummyOp::getOperationName().str()) {
    if (op->hasAttr("wcet.next"))
      return;
  }

  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      ::visitedOps(user, visited);
    }
  }
}
} // namespace
