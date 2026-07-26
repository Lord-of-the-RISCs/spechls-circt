"""Compatibility parsing for the native SpecHLS MLIR emitted by CIRCT."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects import arith, builtin, comb
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i1
from xdsl.ir import Dialect
from xdsl.irdl import AnyAttr, IRDLOperation, attr_def, irdl_op_definition, result_def
from xdsl.parser import Parser
from xdsl.printer import Printer

from spechls.dialect import SpecHLS


@irdl_op_definition
class NativeBoolConstantOp(IRDLOperation):
    """Native MLIR prints boolean arith constants without an explicit type."""

    name = "arith.constant"
    result = result_def(i1)
    value = attr_def(IntegerAttr)

    @classmethod
    def parse(cls, parser: Parser) -> "NativeBoolConstantOp":
        value = 1 if parser.parse_optional_keyword("true") is not None else 0
        if value == 0:
            parser.parse_keyword("false")
        return cls.create(result_types=[i1], attributes={"value": IntegerAttr(value, i1)})

    def print(self, printer: Printer) -> None:
        printer.print_string(" true")


NativeArith = Dialect("arith", [NativeBoolConstantOp, arith.CmpiOp], [])


@irdl_op_definition
class NativeHWConstantOp(IRDLOperation):
    """Parse CIRCT's boolean and attributed ``hw.constant`` spellings."""

    name = "hw.constant"
    result = result_def(AnyAttr())
    value = attr_def(IntegerAttr)

    @classmethod
    def parse(cls, parser: Parser) -> "NativeHWConstantOp":
        boolean = parser.parse_optional_keyword("true")
        if boolean is not None:
            result_type = i1
            value = 1
        elif parser.parse_optional_keyword("false") is not None:
            result_type = i1
            value = 0
        else:
            value = parser.parse_integer()
            parser.parse_punctuation(":")
            result_type = parser.parse_type()
        attributes = parser.parse_optional_attr_dict()
        operation = cls.create(
            result_types=[result_type],
            attributes={"value": IntegerAttr(value, result_type)},
        )
        operation.attributes |= attributes
        return operation


NativeHW = Dialect("hw", [NativeHWConstantOp], [])


def native_mlir_context() -> Context:
    """Create a context for the native transformed SpecHLS test representation."""
    context = Context()
    for dialect in (builtin.Builtin, SpecHLS, NativeArith, NativeHW, comb.Comb):
        context.load_dialect(dialect)
    return context


def parse_native_mlir(source: str) -> ModuleOp:
    """Parse native MLIR using the compatibility operations above."""
    return Parser(native_mlir_context(), source).parse_module()


__all__ = [
    "NativeBoolConstantOp",
    "NativeHWConstantOp",
    "native_mlir_context",
    "parse_native_mlir",
]
