//===- InitAllDialects.h - CIRCT Dialects Registration ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef SPECHLS_INITALLDIALECTS_H_
#define SPECHLS_INITALLDIALECTS_H_

#include "Dialect/ScheduleDialect/ScheduleDialectDialect.h"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "mlir/IR/Dialect.h"

namespace SpecHLS {

// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<SpecHLS::SpecHLSDialect>();
  registry.insert<SpecHLS::ScheduleDialectDialect>();
  // clang-format off
  // clang-format on
}

} // namespace SpecHLS

#endif // SPECHLS_INITALLDIALECTS_H_
