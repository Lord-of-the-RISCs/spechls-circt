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
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

def WcetDialect : Dialect {
    let name = "wcet";

    let summary = "A dialect for wcet analysis in speculative High-Level Synthesis.";
    let description = [{
        TODO
    }];

    let cppNamespace = "wcet";

    //let useDefaultTypePrinterParser = 1;
}

// Base class for the Wcet dialect operations.
class WcetOp<string mnemonic, list<Trait> traits = []>
    : Op<WcetDialect, mnemonic, traits>;

// Base class for the Wcet dialect types.
class WcetType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<WcetDialect, name, traits> {
    let mnemonic = typeMnemonic;
}
//===--------------------------------------------------------------------------------------------------------------===//
// Types
//===--------------------------------------------------------------------------------------------------------------===//

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
}

def Wcet_CastOp : WcetOp<"cast",
  [DeclareOpInterfaceMethods<CastOpInterface>,
  Pure]> {
      let arguments = (ins AnyType:$input);
      let results = (outs AnyType:$output);

      let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
    }

//----
// Wcet Callable 
//---- 

def Wcet_CoreOp : WcetOp<"core", [
  FunctionOpInterface, IsolatedFromAbove,
  CallableOpInterface
]> {
  let summary = "TODO";
  let description = [{ TODO }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs
      );

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
      "::mlir::StringRef":$name, "::mlir::FunctionType":$type,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ::mlir::ArrayRef<::mlir::Type> getResultTypes() {return getFunctionType().getResults(); }
    ::mlir::Region *getCallableRegion() {return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def Wcet_CommitOp : WcetOp<"commit", [Pure, HasParent<"CoreOp">, Terminator]> {
  let summary = "Commit the ouput of the core";
  let description = [{}];

  let arguments = (ins AnyType:$input);

  let assemblyFormat = "$input `:` type($input) attr-dict";

}

def Wcet_CoreInstanceOp : WcetOp<"core_instance",
[DeclareOpInterfaceMethods<CallOpInterface>]> {
      let summary = "instanciate a core";
      let description = [{}];

      let arguments = (ins
          FlatSymbolRefAttr:$callee,
          Variadic<AnyType>:$inputs,
          OptionalAttr<DictArrayAttr>:$arg_attrs,
          OptionalAttr<DictArrayAttr>:$res_attrs
          );

      let results = (outs Variadic<AnyType>);

      let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
      }];

      let builders = [
        OpBuilder<(ins "CoreOp":$callee, "::mlir::ArrayRef<::mlir::Value>":$inputs), [{
            $_state.addOperands(inputs);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getFunctionType().getResults());
          }]>
      ];
    }



#endif // SPECHLS_INCLUDED_DIALECT_WCET_IR_WCET_TD
