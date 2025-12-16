# Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# -----------------------------------------------------------------------------
# AUTO-GENERATED xDSL bindings for the SpecHLS dialect from SpecHLS.td.
#
# This file focuses on *structural* IRDL definitions (ops/attrs/types) so that
# xDSL can load the dialect and parse/print it in the *generic* MLIR format.
#
# Notes:
# - SpecHLS has a number of operations with custom assembly formats in MLIR
#   (kernel/exit/task/gamma/mu/alpha/delay). This generated binding does NOT
#   implement custom parse/print hooks for them. If you need to parse MLIR
#   produced with the non-generic printer, you will need to add those parse/print
#   methods (or print your MLIR with -mlir-print-op-generic).
# - Dialect types (!spechls.array/!spechls.struct) and dialect attributes
#   (#spechls.gamma_spec/#spechls.gamma_config) DO implement parameter parsing
#   compatible with the MLIR syntax in SpecHLS.cpp/SpecHLS.td.
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
from typing import Sequence

from xdsl.dialects import builtin
from xdsl.dialects.utils.format import (
    parse_assignment,
    parse_func_op_like,
    print_assignment,
    print_func_op_like,
)
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    attr_def,
    opt_attr_def,
    operand_def,
    opt_operand_def,
    region_def,
    result_def,
    opt_result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError
from xdsl.irdl import AttrSizedOperandSegments


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$.]*$")


def _is_valid_identifier(s: str) -> bool:
    """Conservative MLIR-ish bare identifier check."""
    return bool(_IDENT_RE.match(s))


def _parse_bare_identifier(parser: AttrParser) -> str:
    """
    Parse a bare identifier (keyword-ish token). The exact helper name changed
    across xDSL versions, so we probe a few likely method names.
    """
    for meth in ("parse_identifier", "parse_bare_id", "parse_bare_ident"):
        fn = getattr(parser, meth, None)
        if fn is not None:
            return fn()
    _parser_error(parser, 
        "AttrParser does not expose parse_identifier/parse_bare_id; "
        "update _parse_bare_identifier for your xDSL version."
    )
    raise AssertionError("unreachable")


def _parse_keyword_or_string(parser: AttrParser) -> str:
    """
    SpecHLS uses MLIR's parseKeywordOrString for struct/type names.
    We accept either a bare identifier or a string literal.
    """
    try:
        return _parse_bare_identifier(parser)
    except ParseError:
        return parser.parse_str_literal()




# -----------------------------------------------------------------------------
# Parser/Printer helpers for op custom formats
# -----------------------------------------------------------------------------

def _parse_uint(parser: Parser) -> int:
    """Parse an unsigned integer literal in custom op formats.

    xDSL's Parser API has evolved a bit over time, so we probe a few method
    names/signatures.
    """
    parse_int = getattr(parser, "parse_integer", None)
    if parse_int is not None:
        try:
            return int(parse_int(allow_negative=False, allow_boolean=False))
        except TypeError:
            # Older signatures didn't take keyword args.
            return int(parse_int())

    # Fallbacks (older naming variants).
    for meth in ("parse_int", "parse_number"):
        fn = getattr(parser, meth, None)
        if fn is not None:
            return int(fn())

    # Last resort: ask the parser to raise a helpful error if available.
    raise ParseError("Parser does not support parsing integer literals")


def _i32_attr(value: int) -> builtin.IntegerAttr:
    """Create a 32-bit integer attribute for things printed as bare integers."""
    ctor = getattr(builtin.IntegerAttr, "from_int_and_width", None)
    if ctor is not None:
        try:
            return ctor(value, 32)
        except TypeError:
            pass

    # Most xDSL versions support IntegerAttr(value, type)
    try:
        return builtin.IntegerAttr(value, builtin.IntegerType(32))
    except TypeError:
        # Some versions store the integer payload in an IntAttr parameter.
        return builtin.IntegerAttr(builtin.IntAttr(value), builtin.IntegerType(32))


def _integer_attr_value(attr: builtin.IntegerAttr) -> int:
    """Extract the Python int value from a builtin.IntegerAttr."""
    # Common layout: IntegerAttr(value=IntAttr(...), type=...)
    if hasattr(attr, "value"):
        v = getattr(attr, "value")
        if hasattr(v, "data"):
            return int(getattr(v, "data"))
        try:
            return int(v)
        except Exception:
            pass

    # Some older variants: IntegerAttr(data=...)
    if hasattr(attr, "data"):
        try:
            return int(getattr(attr, "data"))
        except Exception:
            pass

    # Last resort: try to peel parameters.
    if hasattr(attr, "parameters"):
        params = getattr(attr, "parameters")
        if params:
            p0 = params[0]
            if hasattr(p0, "data"):
                return int(getattr(p0, "data"))

    raise ValueError(f"Cannot extract integer value from {attr!r}")


def _parser_error(parser: Parser, message: str) -> None:
    """Raise a parse error, using parser location if available."""
    raise_fn = getattr(parser, "raise_error", None)
    if callable(raise_fn):
        raise_fn(message)
    raise ParseError(message)

@irdl_attr_definition
class GammaSpecAttr(ParametrizedAttribute):
    """#spechls.gamma_spec< { ... } >"""

    name = "spechls.gamma_spec"

    spec: builtin.DictionaryAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            spec = parser.parse_attribute()
            return (spec,)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.spec)


@irdl_attr_definition
class GammaConfigAttr(ParametrizedAttribute):
    """#spechls.gamma_config< { ... } >"""

    name = "spechls.gamma_config"

    config: builtin.DictionaryAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            cfg = parser.parse_attribute()
            return (cfg,)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.config)


@irdl_attr_definition
class ArrayType(ParametrizedAttribute, TypeAttribute):
    """!spechls.array<element_type, size>"""

    name = "spechls.array"

    element_type: TypeAttribute
    size: builtin.IntAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        parser.parse_punctuation("<")
        elem_ty = parser.parse_type()
        parser.parse_punctuation(",")
        # UnsignedParameter in TableGen.
        sz = parser.parse_integer(allow_negative=False, allow_boolean=False)
        parser.parse_punctuation(">")
        return (elem_ty, builtin.IntAttr(sz))

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_attribute(self.element_type)
        printer.print_string(", ")
        printer.print_int(self.size.data)
        printer.print_string(">")


@irdl_attr_definition
class StructType(ParametrizedAttribute, TypeAttribute):
    """
    !spechls.struct<name, field0:type0, field1:type1, ...>
    !spechls.struct<>  (empty)
    """

    name = "spechls.struct"

    struct_name: builtin.StringAttr
    field_names: builtin.ArrayAttr[builtin.StringAttr]
    field_types: builtin.ArrayAttr[TypeAttribute]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        # NOTE: In xDSL, `parse_parameters` is expected to parse *inside* the angle
        # brackets (don't consume '<' / '>').
        with parser.in_angle_brackets():
            # C++ uses parseString: a quoted string literal.
            name = parser.parse_str_literal()
            name_attr = builtin.StringAttr(name)

            f_names: list[builtin.StringAttr] = []
            f_types: list[TypeAttribute] = []

            def parse_field() -> None:
                fname = parser.parse_str_literal()
                parser.parse_punctuation(":")
                fty = parser.parse_type()
                f_names.append(builtin.StringAttr(fname))
                f_types.append(fty)

            # `{ ... }` list may be empty; elements are comma-separated.
            parser.parse_comma_separated_list(parser.Delimiter.BRACES, parse_field)

            return (
                name_attr,
                builtin.ArrayAttr(tuple(f_names)),
                builtin.ArrayAttr(tuple(f_types)),
            )

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string('<"')
        printer.print_string(self.struct_name.data)
        printer.print_string('" { ')

        for i, (fn, ft) in enumerate(zip(self.field_names.data, self.field_types.data)):
            if i:
                printer.print_string(", ")
            printer.print_string('"')
            printer.print_string(fn.data)
            printer.print_string('" : ')
            printer.print_attribute(ft)
    # Convenience helpers (non-essential)
    def get_field_types(self) -> tuple[TypeAttribute, ...]:
        return tuple(self.field_types.data)

    def get_field_names(self) -> tuple[str, ...]:
        return tuple(sa.data for sa in self.field_names.data)

# -----------------------------------------------------------------------------
# Operations
# -----------------------------------------------------------------------------


# Helper types (avoid relying on builtin.i1 / builtin.index constants existing)
_i1 = builtin.IntegerType(1)
_i32 = builtin.IntegerType(32)
_index = builtin.IndexType()


@irdl_op_definition
class KernelOp(IRDLOperation):
    name = "spechls.kernel"

    sym_name = attr_def(builtin.StringAttr)
    function_type = attr_def(builtin.FunctionType)
    arg_attrs = opt_attr_def(builtin.ArrayAttr)
    res_attrs = opt_attr_def(builtin.ArrayAttr)
    body = region_def()

    @classmethod
    def parse(cls, parser: Parser) -> "KernelOp":
        # Function-like parsing (symbol name, signature, optional attributes, and body)
        name, input_types, return_types, region, extra_attrs, arg_attrs, res_attrs = parse_func_op_like(
            parser,
            reserved_attr_names=(
                "sym_name",
                "function_type",
                "arg_attrs",
                "res_attrs",
                "sym_visibility",
            ),
        )

        # Normalize extra_attrs to a dict if parse helper returned None
        if extra_attrs is None:
            extra_attrs = {}

        attrs: dict[str, Attribute] = {
            "sym_name": builtin.StringAttr(name),
            "function_type": builtin.FunctionType.from_lists(input_types, return_types),
        }
        if arg_attrs is not None:
            attrs["arg_attrs"] = arg_attrs
        if res_attrs is not None:
            attrs["res_attrs"] = res_attrs
        attrs.update(extra_attrs)

        return cls.build(attributes=attrs, regions=[region])

    def print(self, printer: Printer) -> None:
        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            arg_attrs=self.arg_attrs,
            res_attrs=self.res_attrs,
            reserved_attr_names=(
                "sym_name",
                "function_type",
                "arg_attrs",
                "res_attrs",
                "sym_visibility",
            ),
        )


@irdl_op_definition
class ExitOp(IRDLOperation):
    name = "spechls.exit"

    guard = operand_def(_i1)
    values = var_operand_def()

    @classmethod
    def parse(cls, parser: Parser) -> "ExitOp":
        # spechls.exit if %guard (with %v0, %v1 : i32, i32)?
        parser.parse_keyword("if")
        guard_u = parser.parse_unresolved_operand()

        values: list = []
        if parser.parse_optional_keyword("with"):
            # Parse a comma-separated operand list up to ':'
            values_u = [parser.parse_unresolved_operand()]
            while parser.parse_optional_punctuation(","):
                values_u.append(parser.parse_unresolved_operand())

            parser.parse_punctuation(":")
            types = [parser.parse_type()]
            while parser.parse_optional_punctuation(","):
                types.append(parser.parse_type())

            if len(values_u) != len(types):
                _parser_error(parser, 
                    f"expected {len(values_u)} types for {len(values_u)} operands, got {len(types)}"
                )

            values = [parser.resolve_operand(op, ty) for op, ty in zip(values_u, types)]

        # Optional attribute dictionary (ignored by the C++ printer, but accepted by the parser).
        extra_attrs = parser.parse_optional_attr_dict() or {}

        guard = parser.resolve_operand(guard_u, _i1)
        return cls.build(operands=[guard, values], attributes=extra_attrs)

    def print(self, printer: Printer) -> None:
        printer.print_string(" if ")
        printer.print_operand(self.guard)
        if len(self.values) != 0:
            printer.print_string(" with ")
            printer.print_list(self.values, printer.print_operand)
            printer.print_string(" : ")
            printer.print_list([v.type for v in self.values], printer.print_attribute)


@irdl_op_definition
class TaskOp(IRDLOperation):
    name = "spechls.task"

    sym_name = attr_def(builtin.StringAttr)

    args = var_operand_def()
    result = opt_result_def()

    body = region_def()

    @classmethod
    def parse(cls, parser: Parser) -> "TaskOp":
        # spechls.task "name"(%arg0 = %v0, %arg1 = %v1) : (t0, t1) -> tR attributes {..} { ... }
        sym_name_attr = parser.parse_attribute()
        if not isinstance(sym_name_attr, builtin.StringAttr):
            _parser_error(parser, "expected string attribute for task symbol name")

        # Parse assignment list mapping entry block args to operands.
        assignments = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parse_assignment(parser)
        )

        parser.parse_punctuation(":")
        func_type = parser.parse_function_type()

        input_types = list(func_type.inputs.data)
        output_types = list(func_type.outputs.data)

        if len(assignments) != len(input_types):
            _parser_error(parser, 
                f"expected {len(input_types)} assignments, got {len(assignments)}"
            )

        # Typed block arguments for the region.
        region_args = [
            uarg.resolve(ty) for (uarg, _), ty in zip(assignments, input_types)
        ]

        # Optional attribute dictionary with keyword `attributes`.
        extra_attrs = parser.parse_optional_attr_dict_with_keyword(
            reserved_attr_names=("sym_name",)
        ) or {}

        body = parser.parse_region(region_args)

        # Resolve operands from the assignment list.
        resolved_operands = [
            parser.resolve_operand(uop, ty) for (_, uop), ty in zip(assignments, input_types)
        ]

        # Task has at most one result.
        res_ty = output_types[0] if len(output_types) else None

        attrs: dict[str, Attribute] = {"sym_name": sym_name_attr}
        attrs.update(extra_attrs)

        return cls.build(
            operands=[resolved_operands],
            result_types=[res_ty],
            attributes=attrs,
            regions=[body],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.sym_name)

        entry_args = []
        if len(self.body.blocks):
            entry_args = list(self.body.blocks[0].args)

        pairs = list(zip(entry_args, self.args))
        with printer.in_parens():
            printer.print_list(pairs, lambda p: print_assignment(printer, p[0], p[1]))

        printer.print_string(" : (")
        printer.print_list([v.type for v in self.args], printer.print_attribute)
        printer.print_string(") -> ")

        if self.result is None:
            printer.print_string("()")
        else:
            printer.print_attribute(self.result.type)

        # Match the C++ printer: always a separating space, then an optional
        # `attributes {...}` dictionary (excluding sym_name), then the region.
        printer.print_string(" ")

        extra_attrs = dict(self.attributes)
        extra_attrs.pop("sym_name", None)
        if extra_attrs:
            printer.print_string("attributes ")
            printer.print_attr_dict(extra_attrs)
            printer.print_string(" ")

        printer.print_region(self.body, print_entry_block_args=False)


@irdl_op_definition
class CommitOp(IRDLOperation):
    name = "spechls.commit"

    enable = operand_def(_i1)
    value = opt_operand_def()

    @classmethod
    def parse(cls, parser: Parser) -> "CommitOp":
        # spechls.commit %en
        # spechls.commit %en, %v : i32 {attrs?}
        enable_u = parser.parse_unresolved_operand()

        value_u = None
        value_ty = None
        if parser.parse_optional_punctuation(","):
            value_u = parser.parse_unresolved_operand()
            parser.parse_punctuation(":")
            value_ty = parser.parse_type()

        attrs = parser.parse_optional_attr_dict() or {}

        enable_v = parser.resolve_operand(enable_u, _i1)
        value_v = (
            parser.resolve_operand(value_u, value_ty) if value_u is not None else None
        )

        return cls.build(operands=[enable_v, value_v], result_types=[], attributes=attrs)

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.enable)
        if self.value is not None:
            printer.print_string(", ")
            printer.print_operand(self.value)
            printer.print_string(" : ")
            printer.print_attribute(self.value.type)

        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(dict(self.attributes))


@irdl_op_definition
class GammaOp(IRDLOperation):
    name = "spechls.gamma"

    sym_name = attr_def(builtin.StringAttr)
    select = operand_def(builtin.IntegerType)
    inputs = var_operand_def()
    result = result_def()

    @classmethod
    def parse(cls, parser: Parser) -> "GammaOp":
        # <"sym">(%select, %in0, %in1, ...) {attrs}? : selectType, argType
        parser.parse_punctuation("<")
        name_attr = parser.parse_attribute()
        if not isinstance(name_attr, builtin.StringAttr):
            raise ParseError(f"expected string attribute for sym_name, got {type(name_attr)}")
        parser.parse_punctuation(">")

        operands_u = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        if len(operands_u) < 2:
            raise ParseError("expected at least 2 operands in gamma operand list")
        select_u, *inputs_u = operands_u

        extra_attrs = parser.parse_optional_attr_dict() or {}

        parser.parse_punctuation(":")
        select_ty = parser.parse_type()
        parser.parse_punctuation(",")
        arg_ty = parser.parse_type()

        select = parser.resolve_operand(select_u, select_ty)
        inputs = [parser.resolve_operand(op, arg_ty) for op in inputs_u]

        attrs: dict[str, Attribute] = {"sym_name": name_attr}
        attrs.update(extra_attrs)

        return cls.build(
            operands=[select, inputs],
            result_types=[arg_ty],
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" <")
        printer.print_attribute(self.sym_name)
        printer.print_string(">(")
        printer.print_operand(self.select)
        if len(self.inputs) > 0:
            printer.print_string(", ")
            printer.print_list(self.inputs, printer.print_operand)
        printer.print_string(") ")

        extra_attrs = dict(self.attributes)
        extra_attrs.pop("sym_name", None)
        if extra_attrs:
            printer.print_attr_dict(extra_attrs)
            printer.print_string(" ")

        printer.print_string(": ")
        printer.print_attribute(self.select.type)
        printer.print_string(", ")
        if len(self.inputs) > 0:
            printer.print_attribute(self.inputs[0].type)
        else:
            # Fallback: use result type
            printer.print_attribute(self.result.type)


@irdl_op_definition
class MuOp(IRDLOperation):
    name = "spechls.mu"

    sym_name = attr_def(builtin.StringAttr)

    init_value = operand_def()
    loop_value = operand_def()
    result = result_def()

    @classmethod
    def parse(cls, parser: Parser) -> "MuOp":
        # <"name">(%init, %loop) {optional attrs} : type
        parser.parse_punctuation("<")
        name_attr = parser.parse_attribute()
        if not isinstance(name_attr, builtin.StringAttr):
            _parser_error(parser, "expected string attribute for sym_name")

        parser.parse_punctuation(">")
        operands_u = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        if len(operands_u) != 2:
            _parser_error(parser, "mu expects exactly 2 operands")

        attrs = parser.parse_optional_attr_dict() or {}
        parser.parse_punctuation(":")
        ty = parser.parse_type()

        init = parser.resolve_operand(operands_u[0], ty)
        loop = parser.resolve_operand(operands_u[1], ty)

        all_attrs: dict[str, Attribute] = {"sym_name": name_attr, **attrs}
        return cls.build(operands=[init, loop], result_types=[ty], attributes=all_attrs)

    def print(self, printer: Printer) -> None:
        printer.print_string(" <")
        printer.print_attribute(self.sym_name)
        printer.print_string(">")
        with printer.in_parens():
            printer.print_operand(self.init_value)
            printer.print_string(", ")
            printer.print_operand(self.loop_value)

        printer.print_string(" ")
        extra_attrs = dict(self.attributes)
        extra_attrs.pop("sym_name", None)
        if extra_attrs:
            printer.print_attr_dict(extra_attrs)
            printer.print_string(" ")

        printer.print_string(": ")
        printer.print_attribute(self.init_value.type)


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "spechls.print"

    state = operand_def(_i32)
    enable = operand_def(_i1)
    format = attr_def(builtin.StringAttr)
    args = var_operand_def()

    new_state = result_def(_i32)

    @classmethod
    def parse(cls, parser: Parser) -> "PrintOp":
        # spechls.print %state, %enable, "fmt" (, %a, %b : i32, i1)? {attrs?}
        state_u = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        enable_u = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        fmt_attr = parser.parse_attribute()
        if not isinstance(fmt_attr, builtin.StringAttr):
            _parser_error(parser, "expected a string attribute for print format")

        args_u = []
        args_types = []

        if parser.parse_optional_punctuation(",") is not None:
            args_u.append(parser.parse_unresolved_operand())
            while parser.parse_optional_punctuation(",") is not None:
                args_u.append(parser.parse_unresolved_operand())

            parser.parse_punctuation(":")
            args_types.append(parser.parse_type())
            while parser.parse_optional_punctuation(",") is not None:
                args_types.append(parser.parse_type())

            if len(args_u) != len(args_types):
                _parser_error(parser, f"expected {len(args_u)} argument types, got {len(args_types)}")

        attrs = parser.parse_optional_attr_dict() or {}
        attrs = {"format": fmt_attr, **attrs}

        state_v = parser.resolve_operand(state_u, _i32)
        enable_v = parser.resolve_operand(enable_u, _i1)
        resolved_args = [parser.resolve_operand(u, t) for u, t in zip(args_u, args_types)]

        return cls.build(
            operands=[state_v, enable_v, resolved_args],
            result_types=[_i32],
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.state)
        printer.print_string(", ")
        printer.print_operand(self.enable)
        printer.print_string(", ")
        printer.print_attribute(self.format)

        if len(self.args):
            printer.print_string(", ")
            printer.print_list(self.args, printer.print_operand)
            printer.print_string(" : ")
            printer.print_list([v.type for v in self.args], printer.print_attribute)

        extra_attrs = dict(self.attributes)
        extra_attrs.pop("format", None)
        if extra_attrs:
            printer.print_string(" ")
            printer.print_attr_dict(extra_attrs)


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "spechls.call"

    callee = attr_def(builtin.SymbolRefAttr)
    arguments = var_operand_def()
    result = opt_result_def()

    @classmethod
    def parse(cls, parser: Parser) -> "CallOp":
        # spechls.call @foo(%a, %b) : (i32, i32) -> i32
        callee_attr = parser.parse_attribute()
        if not isinstance(callee_attr, builtin.SymbolRefAttr):
            _parser_error(parser, "expected a symbol reference attribute for callee")

        args_u = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )

        attrs = parser.parse_optional_attr_dict() or {}

        parser.parse_punctuation(":")
        func_type = parser.parse_function_type()
        input_types = list(func_type.inputs.data)
        output_types = list(func_type.outputs.data)

        if len(args_u) != len(input_types):
            _parser_error(parser, f"expected {len(input_types)} arguments, got {len(args_u)}")

        if len(output_types) > 1:
            _parser_error(parser, "spechls.call supports at most one result")

        resolved_args = [
            parser.resolve_operand(uop, ty) for uop, ty in zip(args_u, input_types)
        ]
        res_ty = output_types[0] if len(output_types) else None

        all_attrs: dict[str, Attribute] = {"callee": callee_attr, **attrs}
        return cls.build(
            operands=[resolved_args],
            result_types=[res_ty],
            attributes=all_attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.callee)

        with printer.in_parens():
            printer.print_list(self.arguments, printer.print_operand)

        extra_attrs = dict(self.attributes)
        extra_attrs.pop("callee", None)
        if extra_attrs:
            printer.print_string(" ")
            printer.print_attr_dict(extra_attrs)

        printer.print_string(" : (")
        printer.print_list([v.type for v in self.arguments], printer.print_attribute)
        printer.print_string(") -> ")
        if self.result is None:
            printer.print_string("()")
        else:
            printer.print_attribute(self.result.type)


@irdl_op_definition
class AlphaOp(IRDLOperation):
    name = "spechls.alpha"

    array = operand_def(ArrayType)
    index = operand_def(_index)
    value = operand_def()
    we = operand_def(_i1)

    out = result_def(ArrayType)

    @classmethod
    def parse(cls, parser: Parser) -> "AlphaOp":
        # spechls.alpha %array[%index: <index-type>], %value if %we {attrs}? : <array-type>
        array_u = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        index_u = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        index_ty = parser.parse_type()
        parser.parse_punctuation("]")
        parser.parse_punctuation(",")
        value_u = parser.parse_unresolved_operand()
        parser.parse_keyword("if")
        we_u = parser.parse_unresolved_operand()

        attrs = parser.parse_optional_attr_dict() or {}

        parser.parse_punctuation(":")
        array_ty = parser.parse_type()
        if not isinstance(array_ty, ArrayType):
            _parser_error(parser, "alpha expects an ArrayType after ':'")

        array = parser.resolve_operand(array_u, array_ty)
        index = parser.resolve_operand(index_u, index_ty)
        value = parser.resolve_operand(value_u, array_ty.element_type)
        we = parser.resolve_operand(we_u, _i1)

        return cls.build(
            operands=[array, index, value, we],
            result_types=[array_ty],
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.array)
        printer.print_string("[")
        printer.print_operand(self.index)
        printer.print_string(": ")
        printer.print_attribute(self.index.type)
        printer.print_string("], ")
        printer.print_operand(self.value)
        printer.print_string(" if ")
        printer.print_operand(self.we)

        # Optional attributes
        printer.print_op_attributes(self.attributes, print_keyword=False)

        printer.print_string(" : ")
        printer.print_attribute(self.array.type)

@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "spechls.load"

    array = operand_def(ArrayType)
    index = operand_def()
    result = result_def()

    @classmethod
    def parse(cls, parser: Parser) -> "LoadOp":
        # spechls.load %a[%i : i32] : !spechls.array<i32, 8>
        array_u = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        index_u = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        index_ty = parser.parse_type()
        parser.parse_punctuation("]")

        attrs = parser.parse_optional_attr_dict() or {}

        parser.parse_punctuation(":")
        array_ty = parser.parse_type()
        if not isinstance(array_ty, ArrayType):
            _parser_error(parser, "expected !spechls.array type after ':'")

        array_v = parser.resolve_operand(array_u, array_ty)
        index_v = parser.resolve_operand(index_u, index_ty)

        # result type = element type of the array
        res_ty = array_ty.element_type

        return cls.build(
            operands=[array_v, index_v],
            result_types=[res_ty],
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.array)
        printer.print_string("[")
        printer.print_operand(self.index)
        printer.print_string(": ")
        printer.print_attribute(self.index.type)
        printer.print_string("]")

        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(dict(self.attributes))

        printer.print_string(" : ")
        printer.print_attribute(self.array.type)

@irdl_op_definition
class LUTOp(IRDLOperation):
    name = "spechls.lut"

    index = operand_def(builtin.IntegerType)
    contents = attr_def(builtin.DenseArrayBase[builtin.I64])
    result = result_def(builtin.IntegerType)

    @classmethod
    def parse(cls, parser: Parser) -> "LUTOp":
        # spechls.lut %idx <contents> {attrs?} : (i32) -> i8
        index_u = parser.parse_unresolved_operand()
        contents_attr = parser.parse_attribute()
        print(contents_attr)
        if not isinstance(contents_attr, builtin.ArrayAttr):
            _parser_error(parser, "expected a DenseArray attribute for LUT contents ")

        attrs = parser.parse_optional_attr_dict() or {}
        attrs = {"contents": contents_attr, **attrs}

        parser.parse_punctuation(":")
        fnty = parser.parse_function_type()
        in_tys = list(fnty.inputs.data)
        out_tys = list(fnty.outputs.data)

        if len(in_tys) != 1 or len(out_tys) != 1:
            _parser_error(parser, "spechls.lut expects a 1->1 function type")

        index_v = parser.resolve_operand(index_u, in_tys[0])

        return cls.build(
            operands=[index_v],
            result_types=[out_tys[0]],
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.index)
        printer.print_string(" ")
        printer.print_attribute(self.contents)

        extra_attrs = dict(self.attributes)
        extra_attrs.pop("contents", None)
        if extra_attrs:
            printer.print_string(" ")
            printer.print_attr_dict(extra_attrs)

        printer.print_string(" : (")
        printer.print_attribute(self.index.type)
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type)

@irdl_op_definition
class DelayOp(IRDLOperation):
    name = "spechls.delay"

    irdl_options = [AttrSizedOperandSegments(as_property=False)]

    input = operand_def()
    depth = attr_def(builtin.IntegerAttr)
    enable = opt_operand_def(_i1)
    init = opt_operand_def()
    result = result_def()

    @classmethod
    def parse(cls, parser: Parser) -> "DelayOp":
        # spechls.delay %in by 3 if %en init %iv : i32
        in_u = parser.parse_unresolved_operand()
        parser.parse_keyword("by")
        delay = _parse_uint(parser)

        enable_u = None
        init_u = None
        if parser.parse_optional_keyword("if"):
            enable_u = parser.parse_unresolved_operand()
        if parser.parse_optional_keyword("init"):
            init_u = parser.parse_unresolved_operand()

        extra_attrs = parser.parse_optional_attr_dict() or {}
        parser.parse_punctuation(":")
        ty = parser.parse_type()

        in_v = parser.resolve_operand(in_u, ty)
        enable_v = parser.resolve_operand(enable_u, _i1) if enable_u is not None else None
        init_v = parser.resolve_operand(init_u, ty) if init_u is not None else None

       # seg = builtin.DenseArrayBase.create_dense_int_or_index(
        seg = builtin.DenseArrayBase.create_dense_int(
            builtin.i32,
            [1, 1 if enable_v is not None else 0, 1 if init_v is not None else 0],
        )

        attrs = {"operandSegmentSizes": seg,"depth": _i32_attr(delay), **extra_attrs}
        return cls.build(
            operands=[in_v, enable_v, init_v],
            result_types=[ty],
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        # spechls.delay %in by 3 if %en init %iv : i32
        printer.print_string(" ")
        printer.print_operand(self.input)
        printer.print_string(" by ")
        printer.print_string(str(_integer_attr_value(self.depth)))
        if self.enable is not None:
            printer.print_string(" if ")
            printer.print_operand(self.enable)
        if self.init is not None:
            printer.print_string(" init ")
            printer.print_operand(self.init)
        printer.print_string(" : ")
        printer.print_attribute(self.input.type)


@irdl_op_definition
class FIFOOp(IRDLOperation):
    name = "spechls.fifo"

    input = operand_def(StructType)
    depth = attr_def(builtin.IntegerAttr)
    result = result_def(StructType)

    @classmethod
    def parse(cls, parser: Parser) -> "FIFOOp":
        # spechls.fifo <4> %in {attrs?} : (!spechls.struct<...>) -> !spechls.struct<...>
        parser.parse_punctuation("<")
        depth_int = _parse_uint(parser)
        parser.parse_punctuation(">")

        input_u = parser.parse_unresolved_operand()
        attrs = parser.parse_optional_attr_dict() or {}

        attrs = {"depth": _i32_attr(depth_int), **attrs}

        parser.parse_punctuation(":")
        fnty = parser.parse_function_type()
        in_tys = list(fnty.inputs.data)
        out_tys = list(fnty.outputs.data)

        if len(in_tys) != 1 or len(out_tys) != 1:
            _parser_error(parser, "spechls.fifo expects a 1->1 function type")

        input_v = parser.resolve_operand(input_u, in_tys[0])

        return cls.build(
            operands=[input_v],
            result_types=[out_tys[0]],
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" <")
        printer.print_string(str(_integer_attr_value(self.depth)))
        printer.print_string("> ")
        printer.print_operand(self.input)

        extra_attrs = dict(self.attributes)
        extra_attrs.pop("depth", None)
        if extra_attrs:
            printer.print_string(" ")
            printer.print_attr_dict(extra_attrs)

        printer.print_string(" : (")
        printer.print_attribute(self.input.type)
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class PackOp(IRDLOperation):
    name = "spechls.pack"

    inputs = var_operand_def()
    result = result_def(StructType)


@irdl_op_definition
class UnpackOp(IRDLOperation):
    name = "spechls.unpack"

    input = operand_def(StructType)
    results = var_result_def()


@irdl_op_definition
class SyncOp(IRDLOperation):
    name = "spechls.sync"

    inputs = var_operand_def()
    result = result_def()

    @classmethod
    def parse(cls, parser: Parser) -> "SyncOp":
        # spechls.sync %a, %b {attrs?} : i32, i32
        inputs_u = [parser.parse_unresolved_operand()]
        while parser.parse_optional_punctuation(",") is not None:
            inputs_u.append(parser.parse_unresolved_operand())

        attrs = parser.parse_optional_attr_dict() or {}

        parser.parse_punctuation(":")
        inputs_types = [parser.parse_type()]
        while parser.parse_optional_punctuation(",") is not None:
            inputs_types.append(parser.parse_type())

        if len(inputs_u) != len(inputs_types):
            _parser_error(parser, f"expected {len(inputs_u)} input types, got {len(inputs_types)}")

        resolved_inputs = [parser.resolve_operand(u, t) for u, t in zip(inputs_u, inputs_types)]
        if not resolved_inputs:
            _parser_error(parser, "spechls.sync requires at least one input")

        # SpecHLS.cpp inferReturnTypes: result type is the first input type.
        res_ty = resolved_inputs[0].type

        return cls.build(
            operands=[resolved_inputs],
            result_types=[res_ty],
            attributes=attrs,
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_list(self.inputs, printer.print_operand)

        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(dict(self.attributes))

        printer.print_string(" : ")
        printer.print_list([v.type for v in self.inputs], printer.print_attribute)


@irdl_op_definition
class FieldOp(IRDLOperation):
    name = "spechls.field"

    name_attr = attr_def(builtin.StringAttr)
    input = operand_def(StructType)
    result = result_def()


@irdl_op_definition
class FSMOp(IRDLOperation):
    name = "spechls.fsm"

    name_attr = attr_def(builtin.StringAttr)
    gamma_names = attr_def(builtin.ArrayAttr[builtin.StringAttr])
    cond_delays = attr_def(builtin.DenseArrayBase[builtin.I64])
    input_delays = attr_def(builtin.ArrayAttr[builtin.DenseArrayBase[builtin.I64]])

    mispec = operand_def(StructType)
    state = operand_def(StructType)

    result = result_def(StructType)


@irdl_op_definition
class FSMCommandOp(IRDLOperation):
    name = "spechls.fsm_command"

    name_attr = attr_def(builtin.StringAttr)
    state = operand_def(StructType)
    result = result_def(StructType)


@irdl_op_definition
class RewindOp(IRDLOperation):
    name = "spechls.rewind"

    depths = attr_def(builtin.DenseArrayBase[builtin.I64])

    input = operand_def()
    rewind = operand_def(builtin.IntegerType)
    write_command = operand_def(_i1)

    result = result_def()


@irdl_op_definition
class RollbackOp(IRDLOperation):
    name = "spechls.rollback"

    depths = attr_def(builtin.DenseArrayBase[builtin.I64])
    offset = attr_def(builtin.IntegerAttr)

    input = operand_def()
    rollback = operand_def(builtin.IntegerType)
    write_command = operand_def(_i1)

    result = result_def()


@irdl_op_definition
class CancelOp(IRDLOperation):
    name = "spechls.cancel"

    offset = attr_def(builtin.IntegerAttr)

    input = operand_def(_i1)
    rollback = operand_def(builtin.IntegerType)
    write_command = operand_def(_i1)

    result = result_def(_i1)


SpecHLS = Dialect(
    "spechls",
    [
        KernelOp,
        ExitOp,
        TaskOp,
        CommitOp,
        GammaOp,
        MuOp,
        PrintOp,
        CallOp,
        AlphaOp,
        LoadOp,
        LUTOp,
        DelayOp,
        FIFOOp,
        PackOp,
        UnpackOp,
        SyncOp,
        FieldOp,
        FSMOp,
        FSMCommandOp,
        RewindOp,
        RollbackOp,
        CancelOp,
    ],
    [
        GammaSpecAttr,
        GammaConfigAttr,
        ArrayType,
        StructType,
    ],
)

__all__ = [
    "SpecHLS",
    # Types
    "ArrayType",
    "StructType",
    # Attrs
    "GammaSpecAttr",
    "GammaConfigAttr",
    # Ops
    "KernelOp",
    "ExitOp",
    "TaskOp",
    "CommitOp",
    "GammaOp",
    "MuOp",
    "PrintOp",
    "CallOp",
    "AlphaOp",
    "LoadOp",
    "LUTOp",
    "DelayOp",
    "FIFOOp",
    "PackOp",
    "UnpackOp",
    "SyncOp",
    "FieldOp",
    "FSMOp",
    "FSMCommandOp",
    "RewindOp",
    "RollbackOp",
    "CancelOp",
]
