//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_CAPI_SPECHLS_H
#define SPECHLS_INCLUDED_CAPI_SPECHLS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SpecHLS, spechls);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(ScheduleDialect, scheduledialect);

#ifdef __cplusplus
}
#endif

#endif // SPECHLS_INCLUDED_CAPI_SPECHLS_H
