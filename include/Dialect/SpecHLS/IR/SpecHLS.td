//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SPECHLS_SPECHLS_TD
#define SPECHLS_INCLUDED_DIALECT_SPECHLS_SPECHLS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def SpecHLSDialect : Dialect {
    let name = "spechls";
    let summary = "A dialect for speculative High-Level Synthesis.";
    let description = [{
        TODO
    }];
    let cppNamespace = "spechls";
}

// Base class for the SpecHLS dialect operations.
class SpecHLSOp<string mnemonic, list<Trait> traits = []>
    : Op<SpecHLSDialect, mnemonic, traits>;

//===--------------------------------------------------------------------------------------------------------------===//
// Operations
//===--------------------------------------------------------------------------------------------------------------===//

class SpecHLS_TaskLikeOp<string mnemonic, list<Trait> traits = []> :
        SpecHLSOp<mnemonic,
            !listconcat(traits, [AutomaticAllocationScope, IsolatedFromAbove, FunctionOpInterface, SingleBlock])> {
    let arguments = (ins StrAttr:$sym_name,
                         TypeAttrOf<FunctionType>:$function_type,
                         OptionalAttr<DictArrayAttr>:$arg_attrs,
                         OptionalAttr<DictArrayAttr>:$res_attrs);
    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ::mlir::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

        // CallableOpInterface
        ::mlir::Region *getCallableRegion() { return &getBody(); }
    }];
}

def SpecHLS_HKernelOp : SpecHLS_TaskLikeOp<"hkernel"> {}

def SpecHLS_ExitOp : SpecHLSOp<"exit", [HasParent<"HKernelOp">, Terminator, ReturnLike]> {
    let arguments = (ins Variadic<AnyType>:$values);

    let assemblyFormat = [{
        ($values^ `:` type($values))? attr-dict
    }];
    let hasVerifier = 1;
}

def SpecHLS_HTaskOp : SpecHLS_TaskLikeOp<"htask"> {}

def SpecHLS_CommitOp : SpecHLSOp<"commit", [HasParent<"HTaskOp">, Terminator, ReturnLike]> {
    let arguments = (ins Variadic<AnyType>:$values);

    let assemblyFormat = [{
        ($values^ `:` type($values))? attr-dict
    }];
    let hasVerifier = 1;
}

def SpecHLS_LaunchOp : SpecHLSOp<"launch", [DeclareOpInterfaceMethods<CallOpInterface>, HasParent<"HKernelOp">]> {
    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$arguments,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $results)
    }];
}

def SpecHLS_GammaOp : SpecHLSOp<"gamma", [Pure, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let arguments = (ins StrAttr:$sym_name, AnyInteger:$select, Variadic<AnyType>:$inputs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        `<` $sym_name `>` `(` $select `,` $inputs `)` attr-dict `:` type($select) `,` functional-type($inputs, $result)
    }];

    let extraClassDefinition = [{
        void $cppClass::getAsmResultNames(llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
            setNameFn(getResult(), "gamma");
        }
    }];
}

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_SPECHLS_TD

// kate: syntax Normal;
