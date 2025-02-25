//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SPECHLS_SPECHLS_DIALECT_TD
#define SPECHLS_INCLUDED_DIALECT_SPECHLS_SPECHLS_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SpecHLS dialect definition.
//===----------------------------------------------------------------------===//

def SpecHLS_Dialect : Dialect {
  let name = "SpecHLS";
  let summary = "A Dialect for Speculative HLS.";
  let description = [{
        This dialect is tu support speculative pipelining in CIRCT .
    }];
  let cppNamespace = "SpecHLS";
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base SpecHLS operation definition.
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class SpecHLSOp<string mnemonic, list<Trait> traits = []>
    : Op<SpecHLS_Dialect, mnemonic, traits>;

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_SPECHLS_DIALECT_TD
