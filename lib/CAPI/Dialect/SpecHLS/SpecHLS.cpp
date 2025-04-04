//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "CAPI/Dialect/SpecHLS.h" // IWYU pragma: keep
#include "Dialect/SpecHLS/IR/SpecHLS.h"

#include <mlir/CAPI/Registration.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SpecHLS, spechls, spechls::SpecHLSDialect);
