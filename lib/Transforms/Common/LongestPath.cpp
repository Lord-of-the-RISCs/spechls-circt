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
#include <map>
#include <stack>

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

void topologicalSort(Operation *op, std::stack<Value> &stack,
                     DenseMap<Operation *, bool> &visited);

void longestPath(DummyOp starting_point, PatternRewriter &rewriter,
                 StringAttr dist_name);

void visitedOps(Operation *op, SmallVector<Operation *> &visited);

namespace SpecHLS {

struct LongestPathPattern : OpRewritePattern<hw::HWModuleOp> {

  LongestPathPattern(MLIRContext *ctx)
      : OpRewritePattern<hw::HWModuleOp>(ctx) {}

  LogicalResult matchAndRewrite(hw::HWModuleOp top,
                                PatternRewriter &rewriter) const override {
    if (!top->hasAttr("TOP")) {
      return failure();
    }

    Block *body = top.getBodyBlock();
    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockBegin(top.getLoc(), body);
    auto inits = top.getOps<InitOp>();
    SmallVector<Value> init_values;
    SmallVector<Type> init_types;
    for (InitOp init : inits) {
      init_values.push_back(init.getResult());
      init_types.push_back(init.getType());
    }
    if (init_values.size() == 0) {
      return failure();
    }
    DummyOp starting_point = builder.create<DummyOp>(init_types, init_values);

    auto dummy_results = starting_point.getResults();
    for (size_t i = 0; i < init_values.size(); i++) {
      Value oldResult = init_values[i];
      Value newResult = dummy_results[i];
      oldResult.replaceAllUsesExcept(newResult, starting_point);
    }
    std::map<int, SmallVector<DelayOp>> instrs_delay;
    top.walk([&](DelayOp delay) {
      if (delay->hasAttr("instrs")) {
        int instr_num = delay->getAttr("instrs")
                            .cast<IntegerAttr>()
                            .getValue()
                            .getSExtValue();
        if (auto vec = instrs_delay.find(instr_num);
            vec != instrs_delay.end()) {
          instrs_delay[instr_num].push_back(delay);
        } else {
          SmallVector<DelayOp> new_vec = SmallVector<DelayOp>();
          new_vec.push_back(delay);
          instrs_delay[instr_num] = new_vec;
        }
      }
    });
    for (int i = 0; i < instrs_delay.size(); i++) {
      SmallVector<DelayOp> vec = instrs_delay[i];
      SmallVector<Value> del_val;
      SmallVector<Type> del_types;
      for (size_t j = 0; j < vec.size(); j++) {
        del_val.push_back(vec[j].getResult());
        del_types.push_back(vec[j].getType());
      }
      DummyOp inter_instr = builder.create<DummyOp>(del_types, del_val);
      auto results = inter_instr.getResults();
      for (size_t j = 0; j < del_val.size(); j++) {
        Value oldResult = del_val[j];
        Value newResult = results[j];
        oldResult.replaceAllUsesExcept(newResult, inter_instr);
      }
    }

    StringAttr dist_name = rewriter.getStringAttr("dist");
    longestPath(starting_point, rewriter, dist_name);
    top->removeAttr("TOP");
    return success();
  }
};

struct LongestPathPass : public impl::LongestPathPassBase<LongestPathPass> {

public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LongestPathPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "failed\n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLongestPathPass() {
  return std::make_unique<LongestPathPass>();
}

} // namespace SpecHLS

void topologicalSort(Operation *op, std::stack<Value> &stack,
                     DenseMap<Operation *, bool> &visited) {
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

void longestPath(DummyOp starting_point, PatternRewriter &rewriter,
                 StringAttr dist_name) {
  DenseMap<Operation *, int> dists;
  DenseMap<Operation *, bool> visited;
  std::stack<Value> stack;
  SmallVector<Operation *> ops;
  visitedOps(starting_point, ops);
  for (Operation *op : ops) {
    dists[op] = INT_MIN;
    visited[op] = false;
  }
  for (Operation *op : ops) {
    if (!visited[op]) {
      topologicalSort(op, stack, visited);
    }
  }
  dists[starting_point] = 0;
  while (!stack.empty()) {
    Value v = stack.top();
    stack.pop();

    // Get the delay of the Operation
    int delay = 0;
    DelayOp delta = v.getDefiningOp<DelayOp>();
    if (delta) {
      delay = delta.getDepth();
    } else {
      auto unknown = v.getDefiningOp();
      if (unknown->hasAttr("delay")) {
        delay = unknown->getAttr("delay").cast<IntegerAttr>().getInt();
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
      auto use_d = dists[use];
      if (use_d < (dist + delay)) {
        dists[use] = dist + delay;
      }
    }
  }
  for (Operation *op : ops) {
    op->setAttr(dist_name, rewriter.getI32IntegerAttr(dists[op]));
  }
}

void visitedOps(Operation *op, SmallVector<Operation *> &visited) {
  if (std::find(visited.begin(), visited.end(), op) != visited.end()) {
    return;
  }

  visited.push_back(op);

  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      visitedOps(user, visited);
    }
  }
}
