//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SCHEDULE_DIALECT_SCHEDULE_DIALECT_TD
#define SPECHLS_INCLUDED_DIALECT_SCHEDULE_DIALECT_SCHEDULE_DIALECT_TD

include "mlir/IR/OpBase.td"

def ScheduleDialect : Dialect {
  let name = "Schedule";
  let summary = "A dialect for schedule passes.";
  let description = [{
        A dialect for schedule passes.
    }];
  let cppNamespace = "SpecHLS";
  let useDefaultTypePrinterParser = 1;
}

// Base class for the operation in this dialect.
class ScheduleDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<ScheduleDialect, mnemonic, traits>;

#endif // SPECHLS_INCLUDED_DIALECT_SCHEDULE_DIALECT_SCHEDULE_DIALECT_TD
