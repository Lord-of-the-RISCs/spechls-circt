//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SCHEDULE_IR_SCHEDULE_OPS_H
#define SPECHLS_INCLUDED_DIALECT_SCHEDULE_IR_SCHEDULE_OPS_H

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/RegionKindInterface.h>

#include "Dialect/Schedule/IR/Schedule.h"      // IWYU pragma: export
#include "Dialect/Schedule/IR/ScheduleTypes.h" // IWYU pragma: keep

#define GET_OP_CLASSES
#include "Dialect/Schedule/IR/Schedule.h.inc"

#endif // SPECHLS_INCLUDED_DIALECT_SCHEDULE_IR_SCHEDULE_OPS_H
