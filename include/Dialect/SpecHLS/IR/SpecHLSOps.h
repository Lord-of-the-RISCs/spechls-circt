//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_OPS_H
#define SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_OPS_H

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/RegionKindInterface.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Dialect/SpecHLS/IR/SpecHLS.h"      // IWYU pragma: export
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h" // IWYU pragma: keep

#define GET_OP_CLASSES
#include "Dialect/SpecHLS/IR/SpecHLS.h.inc"

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_OPS_H
