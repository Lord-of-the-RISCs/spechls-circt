//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef WCET_ANALYSIS_PENALTY_GRAPH_ANALYSIS_H
#define WCET_ANALYSIS_PENALTY_GRAPH_ANALYSIS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
namespace wcet {
struct PenaltyGraphAnalysis {
  int64_t wcet;

  PenaltyGraphAnalysis(mlir::ModuleOp mod, mlir::SmallVector<size_t> instrs);
};
} // namespace wcet

#endif
