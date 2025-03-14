//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/IR/Schedule.h"

#include <mlir/CAPI/Registration.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Schedule, schedule, schedule::ScheduleDialect);
