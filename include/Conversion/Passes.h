//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_CONVERSION_PASSES_H
#define SPECHLS_CONVERSION_PASSES_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace spechls {

std::unique_ptr<mlir::Pass> createScheduleToSSP();

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

}; // namespace spechls

#endif // SPECHLS_CONVERSION_PASSES_H
