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
wcet::CoreOp createAnalyseCore(mlir::IRRewriter &rewriter, mlir::ModuleOp &top, wcet::CoreOp &analyzedCore,
                               mlir::SmallVector<std::optional<mlir::IntegerAttr>> &state,
                               mlir::SmallVector<mlir::Type> &types, size_t instrs);

mlir::SmallVector<int64_t> retrieveMultWcet(spechls::FSMOp &fsm);

#endif
