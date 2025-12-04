//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_TD
#define SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

def SpecHLSDialect : Dialect {
    let name = "spechls";
    let summary = "A dialect for speculative High-Level Synthesis.";
    let description = [{
        TODO
    }];

    let cppNamespace = "spechls";

    let useDefaultTypePrinterParser = 1;
}

// Base class for the SpecHLS dialect operations.
class SpecHLSOp<string mnemonic, list<Trait> traits = []>
    : Op<SpecHLSDialect, mnemonic, traits>;

// Base class for the SpecHLS dialect types.
class SpecHLSType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<SpecHLSDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

//===--------------------------------------------------------------------------------------------------------------===//
// Attributes
//===--------------------------------------------------------------------------------------------------------------===//

def I64ArrayArrayAttr : TypedArrayAttrBase<I64ArrayAttr, "Array of 64-bit integer dense arrays attribute">;

//===--------------------------------------------------------------------------------------------------------------===//
// Types
//===--------------------------------------------------------------------------------------------------------------===//

def SpecHLS_ArrayType : SpecHLSType<"Array", "array"> {
    let summary = "array type";

    let parameters = (ins "unsigned":$size, "::mlir::Type":$element_type);

    let assemblyFormat = [{
        `<` $element_type `,` $size `>`
    }];
}

def SpecHLS_StructType : SpecHLSType<"Struct", "struct"> {
    let summary = "structure type";

    let parameters = (ins
        StringRefParameter<"struct name", [{""}]>:$name,
        ArrayRefParameter<"::std::string">:$fieldNames,
        ArrayRefParameter<"::mlir::Type">:$fieldTypes
    );

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

//===--------------------------------------------------------------------------------------------------------------===//
// Operations
//===--------------------------------------------------------------------------------------------------------------===//

def SpecHLS_KernelOp : SpecHLSOp<"kernel", [IsolatedFromAbove, FunctionOpInterface, SingleBlock, RegionKindInterface]> {
    let arguments = (ins StrAttr:$sym_name,
                         TypeAttrOf<FunctionType>:$function_type,
                         OptionalAttr<DictArrayAttr>:$arg_attrs,
                         OptionalAttr<DictArrayAttr>:$res_attrs);
    let regions = (region AnyRegion:$body);

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

        //===------------------------------------------------------------------===//
        // RegionKindInterface Methods
        //===------------------------------------------------------------------===//

        static ::mlir::RegionKind getRegionKind(unsigned) {
            return ::mlir::RegionKind::Graph;
        }
    }];
}

def SpecHLS_ExitOp : SpecHLSOp<"exit", [HasParent<"KernelOp">, Terminator, ReturnLike]> {
    let arguments = (ins I1:$guard, Variadic<AnyType>:$values);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def SpecHLS_TaskOp : SpecHLSOp<"task", [SingleBlock, RegionKindInterface, IsolatedFromAbove]> {
    let arguments = (ins StrAttr:$sym_name, Variadic<AnyType>:$args);
    let results = (outs SpecHLS_StructType:$result);
    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Type":$result, "::mlir::StringRef":$name, "::mlir::ValueRange":$inputs), [{
            $_state.addOperands(inputs);
            $_state.addAttribute("sym_name", $_builder.getStringAttr(name));
            ::mlir::Region *body = $_state.addRegion();
            body->emplaceBlock();
            for(auto&& input : inputs) {
              body->addArgument(input.getType(), $_builder.getUnknownLoc());
            }
            if (result)
                $_state.addTypes(result);
        }]>
    ];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // RegionKindInterface Methods
        //===------------------------------------------------------------------===//

        static ::mlir::RegionKind getRegionKind(unsigned) {
            return ::mlir::RegionKind::Graph;
        }

        ::mlir::Block *getBodyBlock() {
            return &getBody().front();
        }
    }];
}

def SpecHLS_CommitOp : SpecHLSOp<"commit", [HasParent<"TaskOp">, Terminator]> {
    let arguments = (ins
        Variadic<AnyType>:$value
    );

    let assemblyFormat = [{
        $value `:` type($value) attr-dict
    }];
    let hasVerifier = 1;
}

def SpecHLS_GammaOp : SpecHLSOp<"gamma",
        [Pure, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let arguments = (ins StrAttr:$sym_name, AnyInteger:$select, Variadic<AnyType>:$inputs);
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDefinition = [{
        void $cppClass::getAsmResultNames(llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
            setNameFn(getResult(), "gamma");
        }
    }];
}

def SpecHLS_MuOp : SpecHLSOp<"mu",
        [Pure, SameOperandsAndResultType, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let arguments = (ins StrAttr:$sym_name, AnyType:$initValue, AnyType:$loopValue);
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;

    let extraClassDefinition = [{
        void $cppClass::getAsmResultNames(llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
            setNameFn(getResult(), "mu");
        }
    }];
}

def SpecHLS_PrintOp : SpecHLSOp<"print", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let arguments = (ins I32:$state, I1:$enable, StrAttr:$format, Variadic<AnyType>:$args);
    let results = (outs I32:$new_state);

    let assemblyFormat = [{
        $state `,` $enable `,` $format (`,` $args^ `:` type($args))? attr-dict
    }];

    let extraClassDefinition = [{
        void $cppClass::getAsmResultNames(llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
            setNameFn(getResult(), "io_state");
        }
    }];
}

def SpecHLS_CallOp : SpecHLSOp<"call",
        [DeclareOpInterfaceMethods<CallOpInterface>,
         DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$arguments,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let results = (outs Optional<AnyType>:$result);

    let assemblyFormat = [{
        $callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $result)
    }];

    let extraClassDefinition = [{
        void $cppClass::getAsmResultNames(llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
            if(getNumResults() != 0)
                setNameFn(getResult(), getCallee());
        }
    }];
}

def SpecHLS_AlphaOp : SpecHLSOp<"alpha"> {
    let arguments = (ins
        Arg<SpecHLS_ArrayType, "The array to write to", [MemWriteAt<0, FullEffect>]>:$array,
        AnyInteger:$index,
        AnyType:$value,
        I1:$we
    );
    let results = (outs SpecHLS_ArrayType:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def SpecHLS_LoadOp : SpecHLSOp<"load", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins
        Arg<SpecHLS_ArrayType, "The array to read from", [MemReadAt<0, FullEffect>]>:$array,
        AnyInteger:$index
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `[` $index `:` type($index) `]` attr-dict `:` type($array)
    }];
}

def SpecHLS_LUTOp : SpecHLSOp<"lut", [Pure, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let arguments = (ins
        AnyInteger:$index,
        DenseI64ArrayAttr:$contents
    );
    let results = (outs AnyInteger:$result);

    let assemblyFormat = [{
        $index $contents attr-dict `:` functional-type($index, $result)
    }];

    let hasVerifier = 1;

    let extraClassDefinition = [{
        void $cppClass::getAsmResultNames(llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
            setNameFn(getResult(), "lut");
        }
    }];
}

def SpecHLS_DelayOp : SpecHLSOp<"delay",
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

def SpecHLS_CancellableDelayOp : SpecHLSOp<"cancellableDelay",
        [AttrSizedOperandSegments]> {
    let arguments = (ins
        I1:$input,
        UI32Attr:$depth,
        AnyInteger:$cancel,
        UI32Attr:$offset,
        I1:$cancelWe,
        Optional<I1>:$enable,
        Optional<I1>:$init
    );
    let results = (outs I1:$result);

    let hasCustomAssemblyFormat = 1;
}

def SpecHLS_RollbackableDelayOp : SpecHLSOp<"rollbackableDelay",
        [AttrSizedOperandSegments,
         TypesMatchWith<"Delay input and output types must match", "input", "result", "$_self">,
         OptionalTypesMatchWith<"Delay input and initialization value must have the same type",
                                "input", "init", "$_self">]> {
    let arguments = (ins
        AnyType:$input,
        UI32Attr:$depth,
        AnyInteger:$rollback,
        UI32Attr:$offset,
        I1:$rbWe,
        DenseI64ArrayAttr:$rollbackDepths,
        Optional<I1>:$enable,
        Optional<AnyType>:$init
    );
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;
}

def SpecHLS_FIFOOp : SpecHLSOp<"fifo", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>, AllTypesMatch<["input", "result"]>]> {
    let arguments = (ins SpecHLS_StructType:$input, UI32Attr:$depth);
    let results = (outs SpecHLS_StructType:$result);

    let assemblyFormat = [{
        `<` $depth `>` $input attr-dict `:` type($input)
    }];

    let hasVerifier = 1;

    let extraClassDefinition = [{
        void $cppClass::getAsmResultNames(llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
            setNameFn(getResult(), "fifo");
        }
    }];
}

def SpecHLS_PackOp : SpecHLSOp<"pack", [Pure]> {
    let arguments = (ins Variadic<AnyType>:$inputs);
    let results = (outs SpecHLS_StructType:$result);

    let assemblyFormat = [{
        $inputs attr-dict `:` functional-type($inputs, $result)
    }];
}

def SpecHLS_UnpackOp : SpecHLSOp<"unpack", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins SpecHLS_StructType:$input);
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def SpecHLS_SyncOp : SpecHLSOp<"sync", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<AnyType>:$inputs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $inputs attr-dict `:` type($inputs)
    }];
}

def SpecHLS_FieldOp : SpecHLSOp<"field", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins
        StrAttr:$name,
        SpecHLS_StructType:$input
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        `<` $name `>` $input attr-dict `:` type($input)
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
}

def SpecHLS_FSMOp : SpecHLSOp<"fsm",
        [TypesMatchWith<"State input and output types must match", "state", "result", "$_self">,
         SameVariadicOperandSize]> {
    let arguments = (ins
        StrAttr:$name,
        StrArrayAttr:$gammaNames,
        DenseI64ArrayAttr:$condDelays,
        DenseI64ArrayAttr:$fastIndices,
        I64ArrayArrayAttr:$inputDelays,
        SpecHLS_StructType:$mispec,
        SpecHLS_StructType:$state
    );
    let results = (outs SpecHLS_StructType:$result);

    let assemblyFormat = [{
        $name `<` $gammaNames `>` $condDelays $fastIndices $inputDelays `,` $mispec `,` $state attr-dict
            `:` type($mispec) `,` type($state)
    }];
}

def SpecHLS_FSMCommandOp : SpecHLSOp<"fsm_command"> {
    let arguments = (ins
        StrAttr:$name,
        SpecHLS_StructType:$state
    );
    let results = (outs SpecHLS_StructType:$result);

    let assemblyFormat = [{
        $name `,` $state attr-dict `:` functional-type($state, $result)
    }];
}

def SpecHLS_RewindOp : SpecHLSOp<"rewind",
        [TypesMatchWith<"Data input and output types must match", "input", "result", "$_self">]> {
    let arguments = (ins
        DenseI64ArrayAttr:$depths,
        AnyType:$input,
        AnyInteger:$rewind,
        I1:$writeCommand
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        `<` $depths `>` $input `,` $rewind `,` $writeCommand attr-dict `:` type($input) `,` type($rewind)
    }];
}

def SpecHLS_RollbackOp : SpecHLSOp<"rollback",
        [TypesMatchWith<"Data input and output types must match", "input", "result", "$_self">]> {
    let arguments = (ins
        DenseI64ArrayAttr:$depths,
        I64Attr:$offset,
        AnyType:$input,
        AnyInteger:$rollback,
        I1:$writeCommand
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        `<` $depths `,` $offset `>` $input `,` $rollback `,` $writeCommand attr-dict `:` type($input) `,` type($rollback)
    }];
}

def SpecHLS_CancelOp : SpecHLSOp<"cancel",
        [TypesMatchWith<"Data input and output types must match", "input", "result", "$_self">]> {
    let arguments = (ins
        I64Attr:$offset,
        I1:$input,
        AnyInteger:$rollback,
        I1:$writeCommand
    );
    let results = (outs I1:$result);

    let assemblyFormat = [{
        `<` $offset `>` $input `,` $rollback `,` $writeCommand attr-dict `:` type($rollback)
    }];
}

#endif // SPECHLS_INCLUDED_DIALECT_SPECHLS_IR_SPECHLS_TD
