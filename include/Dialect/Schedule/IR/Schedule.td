//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef SPECHLS_INCLUDED_DIALECT_SCHEDULE_SCHEDULE_TD
#define SPECHLS_INCLUDED_DIALECT_SCHEDULE_SCHEDULE_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

def ScheduleDialect : Dialect {
    let name = "schedule";

    let summary = "A dialect for scheduling in speculative High-Level Synthesis.";
    let description = [{
        TODO
    }];

    let cppNamespace = "schedule";

    let useDefaultTypePrinterParser = 1;
}

// Base class for the Schedule dialect operations.
class ScheduleOp<string mnemonic, list<Trait> traits = []>
    : Op<ScheduleDialect, mnemonic, traits>;

//===--------------------------------------------------------------------------------------------------------------===//
// Types
//===--------------------------------------------------------------------------------------------------------------===//

def Schedule_OperationType : TypeDef<ScheduleDialect, "ScheduleOperation", []> {
  let mnemonic = "ScheduleOperation";
}

//===--------------------------------------------------------------------------------------------------------------===//
// Operations
//===--------------------------------------------------------------------------------------------------------------===//

def Schedule_CircuitOp : ScheduleOp<"circuit", [RegionKindInterface, NoTerminator, IsolatedFromAbove]> {
    let summary = "Represents a schedulable circuit.";

    let arguments = (ins F32Attr:$targetClock);
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $targetClock attr-dict-with-keyword $body
    }];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // RegionKindInterface Methods
        //===------------------------------------------------------------------===//

        static ::mlir::RegionKind getRegionKind(unsigned) {
            return ::mlir::RegionKind::Graph;
        }
    }];
}

def Schedule_OperationOp : ScheduleOp<"operation", [HasParent<"CircuitOp">]> {
    let summary = "An operation to schedule.";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        Variadic<Schedule_OperationType>:$dependences,
        ArrayAttr:$distances,
        F32Attr:$inDelay,
        I32Attr:$latency,
        F32Attr:$outDelay);
    let results = (outs Schedule_OperationType:$res);

    let assemblyFormat = [{
        $sym_name `(` $dependences `:` $distances `,` $inDelay `,` $latency `,` $outDelay `)` attr-dict
    }];
}

#endif // SPECHLS_INCLUDED_DIALECT_SCHEDULE_SCHEDULE_TD
