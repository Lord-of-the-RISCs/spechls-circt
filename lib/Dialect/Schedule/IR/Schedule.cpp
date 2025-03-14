//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/Schedule/IR/ScheduleOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

#include "Dialect/Schedule/IR/ScheduleDialect.cpp.inc"

using namespace mlir;

//===--------------------------------------------------------------------------------------------------------------===//
// Schedule dialect
//===--------------------------------------------------------------------------------------------------------------===//

void schedule::ScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Schedule/IR/Schedule.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Schedule/IR/ScheduleTypes.cpp.inc"
      >();
}

//===--------------------------------------------------------------------------------------------------------------===//
// TableGen'd types and op method definitions
//===--------------------------------------------------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Dialect/Schedule/IR/ScheduleTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Schedule/IR/Schedule.cpp.inc"
