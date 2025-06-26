//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_WCET_IR_WCET_OPS_H
#define SPECHLS_INCLUDED_DIALECT_WCET_IR_WCET_OPS_H

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/RegionKindInterface.h>

#include "Dialect/Wcet/IR/Wcet.h"      // IWYU pragma: export
#include "Dialect/Wcet/IR/WcetTypes.h" // IWYU pragma: keep

#define GET_OP_CLASSES
#include "Dialect/Wcet/IR/Wcet.h.inc"


#endif // SPECHLS_INCLUDED_DIALECT_WCET_IR_WCET_OPS_H
