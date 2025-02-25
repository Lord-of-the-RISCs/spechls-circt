//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_INIT_ALL_DIALECTS_H
#define SPECHLS_INCLUDED_INIT_ALL_DIALECTS_H

#include "Dialect/ScheduleDialect/ScheduleDialectDialect.h"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "mlir/IR/Dialect.h"

namespace SpecHLS {

// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<SpecHLS::SpecHLSDialect>();
  registry.insert<SpecHLS::ScheduleDialectDialect>();
}

} // namespace SpecHLS

#endif // SPECHLS_INCLUDED_INIT_ALL_DIALECTS_H
