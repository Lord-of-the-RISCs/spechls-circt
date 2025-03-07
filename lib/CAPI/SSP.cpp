//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "CAPI/SSP.h"
#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Dialect/SSP/SSPPasses.h"

#include "mlir/CAPI/Registration.h"

extern "C" {
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Scheduling, ssp, circt::ssp::SSPDialect)
void registerSSPPasses() { circt::ssp::registerPasses(); }
}
