//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_H
#define SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_H

#include <mlir/Pass/Pass.h>

namespace wcet {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

#endif // SPECHLS_INCLUDED_DIALECT_WCET_TRANSFORMS_PASSES_H
