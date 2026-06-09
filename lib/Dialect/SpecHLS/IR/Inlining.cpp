//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#ifndef SPECHLS_DIALECT_TRANSFORMS_INLINING_H
#define SPECHLS_DIALECT_TRANSFORMS_INLINING_H

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"

#include <llvm/ADT/UniqueVector.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

namespace {

mlir::LogicalResult inlineBlock(spechls::OptimizedFuncOp *op, mlir::Block *block, mlir::PatternRewriter &rewriter) {

  mlir::IRMapping mapper;
  for (auto [op, arg] : llvm::zip(op->getArgs(), block->getArguments())) {
    mapper.map(arg, op);
  }
  rewriter.setInsertionPoint(*op);
  for (auto &op : block->without_terminator()) {
    rewriter.clone(op, mapper);
  }

  op->getResult().replaceAllUsesWith(mapper.lookup(llvm::cast<spechls::YieldOp>(block->getTerminator()).getValue()));
  rewriter.eraseOp(*op);
  return mlir::failure();
}

} // namespace

mlir::LogicalResult spechls::OptimizedFuncOp::inlineBody(mlir::PatternRewriter &rewriter) {
  return inlineBlock(this, getBodyBlock(), rewriter);
}

::mlir::LogicalResult spechls::OptimizedFuncOp::inlineOptBody(mlir::PatternRewriter &rewriter) {
  return inlineBlock(this, getOptBodyBlock(), rewriter);
}

#endif // SPECHLS_DIALECT_TRANSFORMS_INLINING_H