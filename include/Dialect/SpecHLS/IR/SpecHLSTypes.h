//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_TYPES_H
#define SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_TYPES_H

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h> // IWYU pragma: keep
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_TYPEDEF_CLASSES
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h.inc"

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_TYPES_H
