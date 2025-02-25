//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/ScheduleDialect.h"
#include "Dialect/Schedule/ScheduleOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace SpecHLS;

void ScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Schedule/ScheduleOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Schedule/ScheduleOpsTypes.cpp.inc"
      >();
}
