//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_WCET_IR_WCET_TD
#define SPECHLS_INCLUDED_DIALECT_WCET_IR_WCET_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

def WcetDialect : Dialect {
    let name = "wcet";

    let summary = "A dialect for wcet analysis in speculative High-Level Synthesis.";
    let description = [{
        TODO
    }];

    let cppNamespace = "wcet";

    let useDefaultTypePrinterParser = 1;
}

// Base class for the Schedule dialect operations.
class WcetOp<string mnemonic, list<Trait> traits = []>
    : Op<WcetDialect, mnemonic, traits>;

//===--------------------------------------------------------------------------------------------------------------===//
// Types
//===--------------------------------------------------------------------------------------------------------------===//

def Wcet_OperationType : TypeDef<WcetDialect, "WcetOperation", []> {
  let mnemonic = "WcetOperation";
}

//===--------------------------------------------------------------------------------------------------------------===//
// Operations
//===--------------------------------------------------------------------------------------------------------------===//

def Wcet_DummyOp : WcetOp<"dummy", []>  {
  let summary = "Dummy.";
  let description = [{Dummy operation for longestPath analysis.}];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $inputs `:` type($inputs) `->` qualified(type($outputs)) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

def Wcet_InitOp : WcetOp<"init", []> {
    let summary = "Init.";
    let description = [{}];

   let arguments = (ins FlatSymbolRefAttr:$name);
   let results = (outs AnyType:$result );

  // let hasCustomAssemblyFormat =1;
     let assemblyFormat = [{
         $name `:` type($result)      attr-dict
     }];
}

def Wcet_PenaltyOp : WcetOp<"penalty",
        [AttrSizedOperandSegments,
         TypesMatchWith<"Delay input and output types must match", "input", "result", "$_self">,
         OptionalTypesMatchWith<"Delay input and initialization value must have the same type",
                                "input", "init", "$_self">]> {
    let arguments = (ins
        AnyType:$input,
        UI32Attr:$depth,
        Optional<I1>:$enable,
        Optional<AnyType>:$init
    );
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;

    let hasCanonicalizeMethod = 1;
}


#endif // SPECHLS_INCLUDED_DIALECT_WCET_IR_WCET_TD
