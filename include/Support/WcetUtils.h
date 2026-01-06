//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef WCET_ANALYSIS_UTILS_H
#define WCET_ANALYSIS_UTILS_H
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace wcet {
typedef struct StateStruct {
  StateStruct(const StateStruct &) = default;
  StateStruct(StateStruct &&) = default;
  StateStruct &operator=(const StateStruct &) = default;
  StateStruct &operator=(StateStruct &&) = default;
  StateStruct(int64_t pen, size_t layers, mlir::SmallVector<std::optional<mlir::IntegerAttr>> st)
      : pen(pen), layers(layers), st(std::move(st)) {}

  int64_t pen;
  size_t layers;
  mlir::SmallVector<std::optional<mlir::IntegerAttr>> st;

  bool operator==(const StateStruct &rhs) const {
    bool result = rhs.pen == this->pen && rhs.layers == this->layers;
    for (auto it : llvm::enumerate(this->st)) {
      if (!result)
        break;
      result = result && it.value() == rhs.st[it.index()];
    }
    return result;
  }

} state;

} // namespace wcet
wcet::CoreOp createAnalyseCore(mlir::IRRewriter &rewriter, mlir::ModuleOp &top, wcet::CoreOp &analyzedCore,
                               mlir::SmallVector<std::optional<mlir::IntegerAttr>> &state,
                               mlir::SmallVector<mlir::Type> &types);

mlir::SmallVector<int64_t> retrieveMultWcet(spechls::FSMOp &fsm);

void dumpGraph(DenseMap<wcet::state, SmallVector<wcet::state>> outs, SmallVector<size_t> instrs,
               DenseMap<wcet::state, int64_t> &dists);

SmallVector<std::optional<IntegerAttr>> generateNextState(IRRewriter &rewriter, wcet::CoreOp &analyzedCore,
                                                          SmallVector<Type> &stTypes,
                                                          SmallVector<std::optional<IntegerAttr>> &dumResult,
                                                          int64_t pen);
#endif
