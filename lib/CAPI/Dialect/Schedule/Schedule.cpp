//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "CAPI/Dialect/Schedule.h" // IWYU pragma: keep
#include "Dialect/Schedule/IR/Schedule.h"
#include "Dialect/Schedule/Transforms/Passes.h" // IWYU pragma: keep

#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>

using namespace schedule;

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Schedule, schedule, schedule::ScheduleDialect);

#include "Dialect/Schedule/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
