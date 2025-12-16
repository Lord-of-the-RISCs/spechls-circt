# hw_ext.py
from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir import Dialect, Operation, Attribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.interfaces import (
    ConstantLikeInterface,
    HasFolderInterface,
    HasCanonicalizationPatternsInterface,
)
from xdsl.traits import Pure
from xdsl.parser import Parser
from xdsl.printer import Printer

from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, op_type_rewrite_pattern
from xdsl.dialect_interfaces.constant_materialization import ConstantMaterializationInterface

# -------------------------
# hw.constant op
# -------------------------

@irdl_op_definition
class HWConstantOp(
    IRDLOperation,
    ConstantLikeInterface,
    HasFolderInterface,
    HasCanonicalizationPatternsInterface,
):
    name = "hw.constant"

    # CIRCT: value is an IntegerAttr (with its integer type).
    value = prop_def(IntegerAttr)

    # Result type is the integer type of the attribute (i1/iN)
    res = result_def(IntegerType)

    traits = traits_def(Pure())

    def __init__(self, value: IntegerAttr, *, attributes: dict[str, Attribute] | None = None):
        super().__init__(
            result_types=[value.type],
            properties={"value": value},
            attributes=attributes or {},
        )

    # ConstantLikeInterface
    def get_constant_value(self) -> Attribute:
        return self.value

    # HasFolderInterface: returning an Attribute requires dialect constant materialization.
    def fold(self):
        return (self.value,)

    # Custom parse: support CIRCT sugar:
    #   hw.constant true
    #   hw.constant false
    #   hw.constant 42 : i32
    @classmethod
    def parse(cls, parser: Parser) -> "HWConstantOp":
        if parser.parse_optional_keyword("true") is not None:
            ty = IntegerType(1)
            value = IntegerAttr(1, ty)
        elif parser.parse_optional_keyword("false") is not None:
            ty = IntegerType(1)
            value = IntegerAttr(0, ty)
        else:
            v = parser.parse_integer()
            parser.parse_punctuation(":")
            ty = parser.parse_type()
            if not isinstance(ty, IntegerType):
                parser.raise_error("hw.constant expects an integer type (iN)")
            value = IntegerAttr(v, ty)

        extra_attrs = parser.parse_optional_attr_dict()
        return cls(value, attributes=extra_attrs)

    # Custom print: print `true/false` for i1, otherwise `N : iW`
    def print(self, printer: Printer) -> None:
        ty = self.res.type
        assert isinstance(ty, IntegerType)

        printer.print_string(" ")
        if ty.width.data == 1:
            printer.print_string("true" if int(self.value.value.data) != 0 else "false")
        else:
            printer.print_string(str(self.value.value.data))
            printer.print_string(" : ")
            printer.print_attribute(ty)

        # print any extra attributes (properties like `value` are NOT in this dict)
        printer.print_op_attributes(self.attributes)

    # Canonicalization hook: return patterns used by the canonicalizer.
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return (NormalizeI1Constant(),)


# Example canonicalization:
# Ensure i1 constants are normalized to 0/1 (in case someone builds -1 : i1, etc.)
class NormalizeI1Constant(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HWConstantOp, rewriter: PatternRewriter):
        ty = op.res.type
        if not isinstance(ty, IntegerType) or ty.width.data != 1:
            return
        v = int(op.value.value.data) & 1
        if v != int(op.value.value.data):
            rewriter.replace_op(
                op,
                HWConstantOp(IntegerAttr(v, ty), attributes=dict(op.attributes)),
            )


# -------------------------
# Dialect constant materialization (needed for fold() returning Attributes)
# -------------------------

class HWConstantMaterialization(ConstantMaterializationInterface):
    def materialize_constant(self, value: Attribute, type: Attribute) -> Operation | None:
        if not isinstance(type, IntegerType):
            return None
        if isinstance(value, IntegerAttr):
            # Re-wrap if needed
            if value.type != type:
                value = IntegerAttr(value.value.data, type)
            return HWConstantOp(value)
        return None


# -------------------------
# Extend xDSL's stub CIRCT hw dialect with hw.constant
# -------------------------

from xdsl.dialects.hw import HW as HWStub

HW = Dialect(
    "hw",
    [*HWStub.operations, HWConstantOp],
    [*HWStub.attributes],
    [HWConstantMaterialization()],
)
