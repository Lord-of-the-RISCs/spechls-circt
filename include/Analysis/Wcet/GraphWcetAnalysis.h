//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef WCET_ANALYSIS_GRAPH_WCET_ANALYSIS
#define WCET_ANALYSIS_GRAPH_WCET_ANALYSIS

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace wcet {
struct GraphAnalysis {

  int64_t wcet;

  GraphAnalysis(mlir::ModuleOp mod, mlir::SmallVector<size_t> instrs);
};
} // namespace wcet

#endif
