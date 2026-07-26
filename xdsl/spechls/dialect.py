"""The SpecHLS dialect, structurally equivalent to CIRCT's SpecHLS dialect."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import (
    ArrayAttr, DenseArrayBase, DictionaryAttr, FlatSymbolRefAttr, FunctionType,
    IntAttr, IntegerAttr, IntegerType, StringAttr, SymbolNameConstraint, i1, i32,
    i64,
)
from xdsl.dialects.utils import parse_func_op_like, print_func_op_like
from xdsl.ir import Attribute, Block, Dialect, Operation, ParametrizedAttribute, Region, SSAValue, TypeAttribute
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.irdl import (
    AnyAttr, AttrSizedOperandSegments, IRDLOperation, VarConstraint, attr_def,
    irdl_attr_definition, irdl_op_definition, operand_def, opt_attr_def,
    opt_operand_def, region_def, result_def, traits_def, var_operand_def,
    var_result_def,
)
from xdsl.traits import HasParent, IsolatedFromAbove, IsTerminator, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException


def _integer(value: SSAValue) -> bool:
    return isinstance(value.type, IntegerType)


@irdl_attr_definition
class ArrayType(ParametrizedAttribute, TypeAttribute):
    name = "spechls.array"
    element_type: Attribute
    size: IntAttr

    def __init__(self, element_type: Attribute, size: int | IntAttr):
        super().__init__(element_type, IntAttr(size) if isinstance(size, int) else size)

    def print_parameters(self, printer):
        with printer.in_angle_brackets():
            printer.print_attribute(self.element_type)
            printer.print_string(", ")
            printer.print_int(self.size.data)

    @classmethod
    def parse_parameters(cls, parser):
        parser.parse_characters("<")
        element_type = parser.parse_type()
        parser.parse_characters(",")
        size = parser.parse_integer()
        parser.parse_characters(">")
        return (element_type, IntAttr(size))


@irdl_attr_definition
class StructType(ParametrizedAttribute, TypeAttribute):
    name = "spechls.struct"
    struct_name: StringAttr
    field_names: ArrayAttr[StringAttr]
    field_types: ArrayAttr[Attribute]

    def __init__(self, name: str | StringAttr, field_names: Sequence[str | StringAttr], field_types: Sequence[Attribute]):
        if len(field_names) != len(field_types):
            raise ValueError("field name and field type count mismatch")
        super().__init__(
            StringAttr(name) if isinstance(name, str) else name,
            ArrayAttr(StringAttr(n) if isinstance(n, str) else n for n in field_names),
            ArrayAttr(field_types),
        )

    def print_parameters(self, printer):
        with printer.in_angle_brackets():
            printer.print_string_literal(self.struct_name.data)
            printer.print_string(" { ")
            for index, (name, typ) in enumerate(zip(self.field_names, self.field_types)):
                if index:
                    printer.print_string(", ")
                printer.print_string_literal(name.data)
                printer.print_string(" : ")
                printer.print_attribute(typ)
            printer.print_string(" }")

    @classmethod
    def parse_parameters(cls, parser):
        parser.parse_characters("<")
        name = StringAttr(parser.parse_str_literal())
        fields = parser.parse_comma_separated_list(
            parser.Delimiter.BRACES,
            lambda: (StringAttr(parser.parse_str_literal()), _parse_field_type(parser)),
        )
        parser.parse_characters(">")
        return (name, ArrayAttr(x[0] for x in fields), ArrayAttr(x[1] for x in fields))


def _parse_field_type(parser):
    parser.parse_punctuation(":")
    return parser.parse_type()


@irdl_attr_definition
class SpeculationSlowPathAttr(ParametrizedAttribute):
    """Configuration and recovery commands for one non-fast path."""

    name = "spechls.speculation_slow_path"
    selector: IntAttr
    latency: IntAttr
    rewind: IntAttr
    rbwe: IntAttr
    rollback_mu_ids: ArrayAttr[StringAttr]
    rollback_array_ids: ArrayAttr[StringAttr]
    rollback_gamma_ids: ArrayAttr[StringAttr]
    rollback_depth: IntAttr

    def __init__(self, selector: int | IntAttr, latency: int | IntAttr, rewind: bool | int | IntAttr, rbwe: bool | int | IntAttr, rollback_mu_ids: Sequence[str | StringAttr], rollback_array_ids: Sequence[str | StringAttr], rollback_gamma_ids: Sequence[str | StringAttr], rollback_depth: int | IntAttr):
        def flag(value: bool | int | IntAttr) -> IntAttr:
            return IntAttr(int(value)) if isinstance(value, (bool, int)) else value
        names = lambda values: ArrayAttr(StringAttr(value) if isinstance(value, str) else value for value in values)
        super().__init__(IntAttr(selector) if isinstance(selector, int) else selector, IntAttr(latency) if isinstance(latency, int) else latency, flag(rewind), flag(rbwe), names(rollback_mu_ids), names(rollback_array_ids), names(rollback_gamma_ids), IntAttr(rollback_depth) if isinstance(rollback_depth, int) else rollback_depth)

    def print_parameters(self, printer):
        with printer.in_angle_brackets():
            printer.print_string("selector = "); printer.print_int(self.selector.data)
            printer.print_string(", latency = "); printer.print_int(self.latency.data)
            printer.print_string(", rewind = "); printer.print_int(self.rewind.data)
            printer.print_string(", rbwe = "); printer.print_int(self.rbwe.data)
            for name, values in (("rollback_mu_ids", self.rollback_mu_ids), ("rollback_array_ids", self.rollback_array_ids), ("rollback_gamma_ids", self.rollback_gamma_ids)):
                printer.print_string(f", {name} = [")
                for index, target_id in enumerate(values):
                    if index: printer.print_string(", ")
                    printer.print_string_literal(target_id.data)
                printer.print_string("]")
            printer.print_string(", rollback_depth = "); printer.print_int(self.rollback_depth.data)

    @classmethod
    def parse_parameters(cls, parser):
        parser.parse_punctuation("<")
        parser.parse_keyword("selector"); parser.parse_punctuation("="); selector = IntAttr(parser.parse_integer())
        parser.parse_punctuation(","); parser.parse_keyword("latency"); parser.parse_punctuation("="); latency = IntAttr(parser.parse_integer())
        parser.parse_punctuation(","); parser.parse_keyword("rewind"); parser.parse_punctuation("="); rewind = IntAttr(parser.parse_integer())
        parser.parse_punctuation(","); parser.parse_keyword("rbwe"); parser.parse_punctuation("="); rbwe = IntAttr(parser.parse_integer())
        target_sets = []
        for name in ("rollback_mu_ids", "rollback_array_ids", "rollback_gamma_ids"):
            parser.parse_punctuation(","); parser.parse_keyword(name); parser.parse_punctuation("=")
            target_sets.append(ArrayAttr(StringAttr(value) for value in parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_str_literal)))
        parser.parse_punctuation(","); parser.parse_keyword("rollback_depth"); parser.parse_punctuation("="); rollback_depth = IntAttr(parser.parse_integer())
        parser.parse_punctuation(">")
        return (selector, latency, rewind, rbwe, *target_sets, rollback_depth)


@irdl_attr_definition
class SpeculationEntryAttr(ParametrizedAttribute):
    """Configuration for one speculation input and all of its slow paths."""

    name = "spechls.speculation_entry"
    cond_latency: IntAttr
    resolve_stage: IntAttr
    gamma_id: StringAttr
    fast_selector: IntAttr
    poison_speculation_ids: ArrayAttr[IntAttr]
    slow_paths: ArrayAttr[SpeculationSlowPathAttr]

    def __init__(self, cond_latency: int | IntAttr, resolve_stage: int | IntAttr, gamma_id: str | StringAttr, fast_selector: int | IntAttr, slow_paths: Sequence[SpeculationSlowPathAttr], poison_speculation_ids: Sequence[int | IntAttr] = ()):
        super().__init__(IntAttr(cond_latency) if isinstance(cond_latency, int) else cond_latency, IntAttr(resolve_stage) if isinstance(resolve_stage, int) else resolve_stage, StringAttr(gamma_id) if isinstance(gamma_id, str) else gamma_id, IntAttr(fast_selector) if isinstance(fast_selector, int) else fast_selector, ArrayAttr(IntAttr(value) if isinstance(value, int) else value for value in poison_speculation_ids), ArrayAttr(slow_paths))

    def print_parameters(self, printer):
        with printer.in_angle_brackets():
            printer.print_string("cond_latency = "); printer.print_int(self.cond_latency.data)
            printer.print_string(", resolve_stage = "); printer.print_int(self.resolve_stage.data)
            printer.print_string(", gamma_id = "); printer.print_string_literal(self.gamma_id.data)
            printer.print_string(", fast_selector = "); printer.print_int(self.fast_selector.data)
            printer.print_string(", poison_speculation_ids = [")
            for index, speculation_id in enumerate(self.poison_speculation_ids):
                if index: printer.print_string(", ")
                printer.print_int(speculation_id.data)
            printer.print_string("]")
            printer.print_string(", slow_paths = [")
            for index, slow_path in enumerate(self.slow_paths):
                if index: printer.print_string(", ")
                printer.print_attribute(slow_path)
            printer.print_string("]")

    @classmethod
    def parse_parameters(cls, parser):
        parser.parse_punctuation("<")
        parser.parse_keyword("cond_latency"); parser.parse_punctuation("="); cond_latency = IntAttr(parser.parse_integer())
        parser.parse_punctuation(","); parser.parse_keyword("resolve_stage"); parser.parse_punctuation("="); resolve_stage = IntAttr(parser.parse_integer())
        parser.parse_punctuation(","); parser.parse_keyword("gamma_id"); parser.parse_punctuation("="); gamma_id = StringAttr(parser.parse_str_literal())
        parser.parse_punctuation(","); parser.parse_keyword("fast_selector"); parser.parse_punctuation("="); fast_selector = IntAttr(parser.parse_integer())
        parser.parse_punctuation(","); parser.parse_keyword("poison_speculation_ids"); parser.parse_punctuation("=")
        poison_speculation_ids = ArrayAttr(IntAttr(value) for value in parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_integer))
        parser.parse_punctuation(","); parser.parse_keyword("slow_paths"); parser.parse_punctuation("=")
        slow_paths = parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_attribute)
        parser.parse_punctuation(">")
        if not all(isinstance(path, SpeculationSlowPathAttr) for path in slow_paths):
            parser.raise_error("slow paths must be #spechls.speculation_slow_path attributes")
        return (cond_latency, resolve_stage, gamma_id, fast_selector, poison_speculation_ids, ArrayAttr(slow_paths))


@irdl_attr_definition
class SpeculationConfigAttr(ParametrizedAttribute):
    """The explicit speculation entries consumed by FSM inference."""

    name = "spechls.speculation_config"
    entries: ArrayAttr[SpeculationEntryAttr]

    def __init__(self, entries: Sequence[SpeculationEntryAttr]):
        super().__init__(ArrayAttr(entries))

    @classmethod
    def from_mlir_dictionary(cls, value: ArrayAttr[DictionaryAttr]) -> "SpeculationConfigAttr":
        """Read the portable builtin-attribute form emitted by native MLIR."""
        def integer(attribute: Attribute, field: str) -> int:
            if not isinstance(attribute, IntegerAttr):
                raise VerifyException(f"speculation configuration field '{field}' must be an integer")
            return attribute.value.data

        entries = []
        for entry in value:
            if not isinstance(entry, DictionaryAttr):
                raise VerifyException("speculation configuration entries must be dictionaries")
            data = entry.data
            gamma_id = data.get("gamma_id")
            slow_paths = data.get("slow_paths")
            poisoned = data.get("poison_speculation_ids")
            if not isinstance(gamma_id, StringAttr) or not isinstance(slow_paths, ArrayAttr) or not isinstance(poisoned, ArrayAttr):
                raise VerifyException("speculation configuration entry is missing required fields")
            paths = []
            for path in slow_paths:
                if not isinstance(path, DictionaryAttr):
                    raise VerifyException("slow path must be a dictionary")
                path_data = path.data
                target_names = lambda field: [item for item in path_data.get(field, ArrayAttr(())) if isinstance(item, StringAttr)]
                paths.append(SpeculationSlowPathAttr(
                    integer(path_data["selector"], "selector"),
                    integer(path_data["latency"], "latency"),
                    integer(path_data["rewind"], "rewind"),
                    integer(path_data["rbwe"], "rbwe"),
                    target_names("rollback_mu_ids"),
                    target_names("rollback_array_ids"),
                    target_names("rollback_gamma_ids"),
                    integer(path_data["rollback_depth"], "rollback_depth"),
                ))
            entries.append(SpeculationEntryAttr(
                integer(data["cond_latency"], "cond_latency"),
                0,
                gamma_id,
                integer(data["fast_selector"], "fast_selector"),
                paths,
                [integer(item, "poison_speculation_ids") for item in poisoned],
            ))
        return cls(entries)

    def print_parameters(self, printer):
        with printer.in_angle_brackets():
            printer.print_string("[")
            for index, entry in enumerate(self.entries):
                if index: printer.print_string(", ")
                printer.print_attribute(entry)
            printer.print_string("]")

    @classmethod
    def parse_parameters(cls, parser):
        parser.parse_punctuation("<")
        entries = parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_attribute)
        parser.parse_punctuation(">")
        if not all(isinstance(entry, SpeculationEntryAttr) for entry in entries):
            parser.raise_error("speculation configuration entries must be #spechls.speculation_entry attributes")
        return (ArrayAttr(entries),)


@irdl_attr_definition
class SpeculationRecoveryAttr(ParametrizedAttribute):
    """One typed edge or path from Java's symbolic recovery exploration."""

    name = "spechls.speculation_recovery"
    kind: StringAttr
    speculation_ids: ArrayAttr[IntAttr]
    selectors: ArrayAttr[IntAttr]
    release_windows: ArrayAttr[IntAttr]
    poisoned_speculation_ids: ArrayAttr[IntAttr]
    remaining_cycles: IntAttr
    destination_speculation_ids: ArrayAttr[IntAttr]
    destination_selectors: ArrayAttr[IntAttr]

    def __init__(self, kind: str | StringAttr, speculation_ids: Sequence[int | IntAttr], selectors: Sequence[int | IntAttr], release_windows: Sequence[int | IntAttr], poisoned_speculation_ids: Sequence[int | IntAttr] = (), remaining_cycles: int | IntAttr = -1, destination_speculation_ids: Sequence[int | IntAttr] = (), destination_selectors: Sequence[int | IntAttr] = ()):
        if not (len(speculation_ids) == len(selectors) == len(release_windows)):
            raise ValueError("recovery speculation, selector, and release-window counts must match")
        values = lambda items: ArrayAttr(IntAttr(value) if isinstance(value, int) else value for value in items)
        if len(destination_speculation_ids) != len(destination_selectors):
            raise ValueError("recovery destination speculation and selector counts must match")
        super().__init__(StringAttr(kind) if isinstance(kind, str) else kind, values(speculation_ids), values(selectors), values(release_windows), values(poisoned_speculation_ids), IntAttr(remaining_cycles) if isinstance(remaining_cycles, int) else remaining_cycles, values(destination_speculation_ids), values(destination_selectors))

    def print_parameters(self, printer):
        with printer.in_angle_brackets():
            printer.print_string("kind = "); printer.print_string_literal(self.kind.data)
            for name, values in (("speculation_ids", self.speculation_ids), ("selectors", self.selectors), ("release_windows", self.release_windows)):
                printer.print_string(f", {name} = [")
                for index, value in enumerate(values):
                    if index: printer.print_string(", ")
                    printer.print_int(value.data)
                printer.print_string("]")
            printer.print_string(", poisoned_speculation_ids = [")
            for index, value in enumerate(self.poisoned_speculation_ids):
                if index: printer.print_string(", ")
                printer.print_int(value.data)
            printer.print_string("]")
            printer.print_string(", remaining_cycles = "); printer.print_int(self.remaining_cycles.data)
            printer.print_string(", destination_speculation_ids = [")
            for index, value in enumerate(self.destination_speculation_ids):
                if index: printer.print_string(", ")
                printer.print_int(value.data)
            printer.print_string("]")
            printer.print_string(", destination_selectors = [")
            for index, value in enumerate(self.destination_selectors):
                if index: printer.print_string(", ")
                printer.print_int(value.data)
            printer.print_string("]")

    @classmethod
    def parse_parameters(cls, parser):
        parser.parse_punctuation("<")
        parser.parse_keyword("kind"); parser.parse_punctuation("="); kind = StringAttr(parser.parse_str_literal())
        values = []
        for name in ("speculation_ids", "selectors", "release_windows", "poisoned_speculation_ids"):
            parser.parse_punctuation(","); parser.parse_keyword(name); parser.parse_punctuation("=")
            values.append(ArrayAttr(IntAttr(value) for value in parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_integer)))
        parser.parse_punctuation(","); parser.parse_keyword("remaining_cycles"); parser.parse_punctuation("=")
        remaining_cycles = IntAttr(parser.parse_integer())
        parser.parse_punctuation(","); parser.parse_keyword("destination_speculation_ids"); parser.parse_punctuation("=")
        destination_speculation_ids = ArrayAttr(IntAttr(value) for value in parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_integer))
        parser.parse_punctuation(","); parser.parse_keyword("destination_selectors"); parser.parse_punctuation("=")
        destination_selectors = ArrayAttr(IntAttr(value) for value in parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_integer))
        parser.parse_punctuation(">")
        return (kind, *values, remaining_cycles, destination_speculation_ids, destination_selectors)


class _SameTypeOp(IRDLOperation):
    DATA: ClassVar = VarConstraint("data", AnyAttr())


@irdl_op_definition
class KernelOp(IRDLOperation):
    name = "spechls.kernel"
    sym_name = attr_def(SymbolNameConstraint())
    function_type = attr_def(FunctionType)
    arg_attrs = opt_attr_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_attr_def(ArrayAttr[DictionaryAttr])
    body = region_def()
    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface())

    def __init__(self, name: str, function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]], body: Region | None = None):
        if isinstance(function_type, tuple): function_type = FunctionType.from_lists(*function_type)
        super().__init__(attributes={"sym_name": StringAttr(name), "function_type": function_type}, regions=[body or Region(Block(arg_types=function_type.inputs))])

    @classmethod
    def parse(cls, parser):
        name, inputs, outputs, body, extra_attrs, arg_attrs, res_attrs = parse_func_op_like(
            parser, reserved_attr_names=("sym_name", "function_type", "arg_attrs", "res_attrs")
        )
        op = cls(name, (inputs, outputs), body)
        if arg_attrs is not None: op.attributes["arg_attrs"] = arg_attrs
        if res_attrs is not None: op.attributes["res_attrs"] = res_attrs
        if extra_attrs is not None: op.attributes |= extra_attrs.data
        return op

    def print(self, printer):
        print_func_op_like(
            printer, self.sym_name, self.function_type, self.body, self.attributes,
            arg_attrs=self.arg_attrs, res_attrs=self.res_attrs,
            reserved_attr_names=("sym_name", "function_type", "arg_attrs", "res_attrs"),
        )

    def verify_(self):
        _verify_single_block_terminator(self.body, ExitOp, "kernel")


@irdl_op_definition
class ExitOp(IRDLOperation):
    name = "spechls.exit"
    guard = operand_def(i1)
    values = var_operand_def()
    traits = traits_def(IsTerminator(), HasParent(KernelOp))
    assembly_format = "`if` $guard (`with` $values^ `:` type($values))? attr-dict"

    def __init__(self, guard: SSAValue | Operation, values: Sequence[SSAValue | Operation]):
        """Build a kernel terminator from CoreDSL's explicit loop-exit payload.

        Role: preserve the established OpenHLS builder interface while using the
        consolidated SpecHLS xDSL operation definition.
        Goal: allow CoreDSL lowering to construct typed exit operations directly.
        """
        super().__init__(operands=[SSAValue.get(guard), [SSAValue.get(value) for value in values]])

    def verify_(self):
        kernel = self.parent_op()
        assert isinstance(kernel, KernelOp)
        if tuple(v.type for v in self.values) != kernel.function_type.outputs.data:
            raise VerifyException("exit values must match enclosing kernel result types")


@irdl_op_definition
class TaskOp(IRDLOperation):
    name = "spechls.task"
    sym_name = attr_def(StringAttr)
    args = var_operand_def()
    result = result_def(StructType)
    body = region_def()
    traits = traits_def(IsolatedFromAbove())

    def __init__(self, result: StructType, name: str, args: Sequence[SSAValue | Operation], body: Region | None = None):
        values = [SSAValue.get(arg) for arg in args]
        super().__init__(operands=[values], result_types=[result], attributes={"sym_name": StringAttr(name)}, regions=[body or Region(Block(arg_types=[v.type for v in values]))])

    @classmethod
    def parse(cls, parser: Parser) -> "TaskOp":
        name = parser.parse_str_literal()

        def parse_binding():
            argument = parser.parse_argument(expect_type=False)
            parser.parse_punctuation("=")
            return argument, parser.parse_unresolved_operand()

        bindings = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parse_binding)
        parser.parse_punctuation(":")
        input_types = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parser.parse_type)
        parser.parse_punctuation("->")
        result_type = parser.parse_type()
        if not isinstance(result_type, StructType):
            parser.raise_error("spechls.task result must be a !spechls.struct")
        if len(bindings) != len(input_types):
            parser.raise_error("task binding count must match task input types")
        operands = parser.resolve_operands(
            [operand for _, operand in bindings], input_types, parser.pos
        )
        arguments = [argument.resolve(type) for (argument, _), type in zip(bindings, input_types)]
        attributes = parser.parse_optional_attr_dict_with_keyword(("sym_name",))
        body = parser.parse_region(arguments)
        task = cls(result_type, name, operands, body)
        if attributes is not None:
            task.attributes |= attributes.data
        return task

    def verify_(self):
        _verify_single_block_terminator(self.body, CommitOp, "task")


@irdl_op_definition
class CommitOp(IRDLOperation):
    name = "spechls.commit"
    value = var_operand_def()
    traits = traits_def(IsTerminator(), HasParent(TaskOp))
    assembly_format = "$value `:` type($value) attr-dict"

    def __init__(self, values: Sequence[SSAValue | Operation]):
        super().__init__(operands=[[SSAValue.get(value) for value in values]])

    def verify_(self):
        task = self.parent_op(); assert isinstance(task, TaskOp)
        if tuple(v.type for v in self.value) != task.result.type.field_types.data:
            raise VerifyException("commit values must match task result fields")


def _verify_single_block_terminator(region: Region, terminator: type[Operation], owner: str) -> None:
    if len(region.blocks) != 1:
        raise VerifyException(f"{owner} body must have exactly one block")
    if not isinstance(region.block.last_op, terminator):
        raise VerifyException(f"{owner} body must end with {terminator.name}")


@irdl_op_definition
class GammaOp(IRDLOperation):
    name = "spechls.gamma"
    DATA: ClassVar = VarConstraint("data", AnyAttr())
    sym_name = attr_def(StringAttr)
    select = operand_def()
    inputs = var_operand_def(DATA)
    result = result_def(DATA)
    def __init__(self, name: str, select: SSAValue, inputs: Sequence[SSAValue]):
        super().__init__(operands=[select, inputs], result_types=[inputs[0].type], attributes={"sym_name": StringAttr(name)})
    def verify_(self):
        if len(self.inputs) < 2: raise VerifyException("gamma expects at least two data inputs")
        if not _integer(self.select): raise VerifyException("gamma select must be an integer")
        if any(v.type != self.result.type for v in self.inputs): raise VerifyException("gamma input and result types must match")
    assembly_format = "`<` $sym_name `>` `(` $select `,` $inputs `)` attr-dict `:` type($select) `,` type($result)"


@irdl_op_definition
class MuOp(_SameTypeOp):
    name = "spechls.mu"
    sym_name = attr_def(StringAttr)
    init_value = operand_def(_SameTypeOp.DATA)
    loop_value = operand_def(_SameTypeOp.DATA)
    result = result_def(_SameTypeOp.DATA)
    assembly_format = "`<` $sym_name `>` `(` $init_value `,` $loop_value `)` attr-dict `:` type($result)"

    def __init__(self, name: str, init_value: SSAValue | Operation, loop_value: SSAValue | Operation):
        """Build one loop-carried value with CoreDSL's stable positional API.

        Role: make the existing structured-ISS lowering independent of the former
        OpenHLS SpecHLS package.
        Goal: preserve typed recurrence construction for SCC extraction.
        """
        initial = SSAValue.get(init_value)
        super().__init__(
            operands=[initial, SSAValue.get(loop_value)],
            result_types=[initial.type],
            attributes={"sym_name": StringAttr(name)},
        )


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "spechls.print"
    state = operand_def(i32); enable = operand_def(i1); format = attr_def(StringAttr); args = var_operand_def(); new_state = result_def(i32)
    assembly_format = "$state `,` $enable `,` $format (`,` $args^ `:` type($args))? attr-dict"


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "spechls.call"
    callee = attr_def(FlatSymbolRefAttr); arguments = var_operand_def(); arg_attrs = opt_attr_def(ArrayAttr[DictionaryAttr]); res_attrs = opt_attr_def(ArrayAttr[DictionaryAttr]); result = var_result_def()
    assembly_format = "$callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $result)"


@irdl_op_definition
class AlphaOp(IRDLOperation):
    name = "spechls.alpha"
    ARRAY: ClassVar = VarConstraint("array", AnyAttr())
    array = operand_def(ARRAY); index = operand_def(); value = operand_def(); we = operand_def(i1); result = result_def(ARRAY)

    def __init__(self, array: SSAValue | Operation, index: SSAValue | Operation, value: SSAValue | Operation, write_enable: SSAValue | Operation):
        """Build one guarded array update using the active CoreDSL lowering API.

        Role: retain value-semantic architectural-state updates after consolidating
        the OpenHLS dialect into this package.
        Goal: emit an array result whose type matches the updated input state.
        """
        state = SSAValue.get(array)
        super().__init__(
            operands=[state, SSAValue.get(index), SSAValue.get(value), SSAValue.get(write_enable)],
            result_types=[state.type],
        )

    def print(self, printer):
        """Print CIRCT's guarded array-update syntax for native spechls-opt.

        Role: retain the OpenHLS form where the value type is inferred from the
        array element type.
        Goal: keep CoreDSL-generated array updates consumable by the C++ dialect.
        """
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
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)
    def verify_(self):
        if not isinstance(self.array.type, ArrayType) or not _integer(self.index) or self.value.type != self.array.type.element_type:
            raise VerifyException("alpha array, value, and result types are inconsistent")


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "spechls.load"
    array = operand_def(ArrayType); index = operand_def(); result = result_def()

    def __init__(self, array: SSAValue | Operation, index: SSAValue | Operation):
        """Build one architectural array read using the CoreDSL lowering API.

        Role: preserve the previous OpenHLS load builder while centralizing the
        concrete operation class in the LOTR SpecHLS package.
        Goal: infer the scalar result type from the array element type.
        """
        state = SSAValue.get(array)
        if not isinstance(state.type, ArrayType):
            raise VerifyException("spechls.load requires a spechls.array operand")
        super().__init__(operands=[state, SSAValue.get(index)], result_types=[state.type.element_type])

    def print(self, printer):
        """Print CIRCT's inferred-result load syntax for native spechls-opt.

        Role: preserve the native SpecHLS load assembly accepted by the C++ flow.
        Goal: avoid serializing an xDSL-only explicit result type annotation.
        """
        printer.print_string(" ")
        printer.print_operand(self.array)
        printer.print_string("[")
        printer.print_operand(self.index)
        printer.print_string(": ")
        printer.print_attribute(self.index.type)
        printer.print_string("] : ")
        printer.print_attribute(self.array.type)
    def verify_(self):
        if not _integer(self.index) or self.result.type != self.array.type.element_type: raise VerifyException("load result must be the array element type")


@irdl_op_definition
class LUTOp(IRDLOperation):
    name = "spechls.lut"
    index = operand_def(); contents = attr_def(DenseArrayBase.constr(i64)); result = result_def()
    assembly_format = "$index $contents attr-dict `:` functional-type($index, $result)"
    def verify_(self):
        if not _integer(self.index) or not _integer(self.result): raise VerifyException("lut index and result must be integers")
        if len(self.contents) == 0 or len(self.contents) & (len(self.contents) - 1): raise VerifyException("lut contents size must be a power of two")


class _DelayBase(_SameTypeOp):
    depth = attr_def(IntegerAttr)
    input = operand_def(_SameTypeOp.DATA)
    enable = opt_operand_def(i1)
    init = opt_operand_def(_SameTypeOp.DATA)
    result = result_def(_SameTypeOp.DATA)
    irdl_options = (AttrSizedOperandSegments(),)


@irdl_op_definition
class DelayOp(_DelayBase):
    name = "spechls.delay"
    assembly_format = "$input `by` $depth (`if` $enable^)? (`init` $init^)? attr-dict `:` type($result)"


@irdl_op_definition
class CancellableDelayOp(IRDLOperation):
    name = "spechls.cancellableDelay"
    input = operand_def(i1); depth = attr_def(IntegerAttr); cancel = operand_def(); offset = attr_def(IntegerAttr); cancel_we = operand_def(i1); enable = opt_operand_def(i1); init = opt_operand_def(i1); result = result_def(i1)
    irdl_options = (AttrSizedOperandSegments(),)
    assembly_format = "$input `by` $depth `cancel` $cancel `:` type($cancel) `at` $offset $cancel_we (`if` $enable^)? (`init` $init^)? attr-dict `:` type($result)"


@irdl_op_definition
class RollbackableDelayOp(_DelayBase):
    name = "spechls.rollbackableDelay"
    rollback = operand_def(); offset = attr_def(IntegerAttr); rb_we = operand_def(i1); rollback_depths = attr_def(DenseArrayBase.constr(i64))
    assembly_format = "$input `by` $depth `rollback` $rollback `:` type($rollback) `at` $offset $rb_we $rollback_depths (`if` $enable^)? (`init` $init^)? attr-dict `:` type($result)"


@irdl_op_definition
class FIFOOp(IRDLOperation):
    name = "spechls.fifo"
    input = operand_def(StructType); depth = attr_def(IntegerAttr); result = result_def(StructType)
    assembly_format = "`<` $depth `>` $input `:` type($input) attr-dict `->` type($result)"
    def verify_(self):
        if self.input.type != self.result.type or not self.input.type.field_types or self.input.type.field_types.data[0] != i1: raise VerifyException("fifo input must start with i1 and match its result")


@irdl_op_definition
class PackOp(IRDLOperation):
    name = "spechls.pack"
    inputs = var_operand_def(); result = result_def(StructType)
    assembly_format = "$inputs attr-dict `:` functional-type($inputs, $result)"
    def verify_(self):
        if tuple(v.type for v in self.inputs) != self.result.type.field_types.data: raise VerifyException("pack inputs must match struct fields")


@irdl_op_definition
class UnpackOp(IRDLOperation):
    name = "spechls.unpack"
    input = operand_def(StructType); results = var_result_def()
    assembly_format = "$input attr-dict `:` type($input)"
    def verify_(self):
        if self.results.types != self.input.type.field_types.data: raise VerifyException("unpack results must match struct fields")


@irdl_op_definition
class SyncOp(IRDLOperation):
    name = "spechls.sync"
    DATA: ClassVar = VarConstraint("data", AnyAttr())
    inputs = var_operand_def(DATA); result = result_def(DATA)
    assembly_format = "$inputs attr-dict `:` type($inputs)"
    def verify_(self):
        if not self.inputs or self.result.type != self.inputs[0].type: raise VerifyException("sync result must match its first input")


@irdl_op_definition
class FieldOp(IRDLOperation):
    name = "spechls.field"
    field_name = attr_def(StringAttr, attr_name="name"); input = operand_def(StructType); result = result_def()

    def print(self, printer):
        """Print CIRCT's inferred-result struct-field projection syntax.

        Role: serialize task interfaces created by SCC outlining in the native
        SpecHLS form.
        Goal: let native spechls-opt infer the projection type from the struct.
        """
        printer.print_string("<")
        printer.print_string_literal(self.field_name.data)
        printer.print_string("> ")
        printer.print_operand(self.input)
        printer.print_string(" : ")
        printer.print_attribute(self.input.type)

    @classmethod
    def parse(cls, parser: Parser) -> "FieldOp":
        parser.parse_punctuation("<")
        field_name = parser.parse_str_literal()
        parser.parse_punctuation(">")
        input_value = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        if parser.parse_optional_punctuation("<") is not None:
            struct_name = parser.parse_str_literal()

            def parse_field() -> tuple[str, Attribute]:
                name = parser.parse_str_literal()
                parser.parse_punctuation(":")
                return name, parser.parse_type()

            fields = parser.parse_comma_separated_list(parser.Delimiter.BRACES, parse_field)
            parser.parse_punctuation(">")
            input_type = StructType(struct_name, [name for name, _ in fields], [type for _, type in fields])
        else:
            input_type = parser.parse_type()
            if not isinstance(input_type, StructType):
                parser.raise_error("spechls.field input must be a !spechls.struct")
            fields = list(zip((name.data for name in input_type.field_names), input_type.field_types))
        input_value = parser.resolve_operand(input_value, input_type)
        try:
            index = [name for name, _ in fields].index(field_name)
        except ValueError:
            parser.raise_error(f"field '{field_name}' is not present in the input struct")
        return cls.create(
            operands=[input_value],
            result_types=[fields[index][1]],
            attributes={"name": StringAttr(field_name)},
        )

    def verify_(self):
        try: index = [n.data for n in self.input.type.field_names].index(self.field_name.data)
        except ValueError: raise VerifyException(f"invalid field name '{self.field_name.data}'")
        if self.result.type != self.input.type.field_types.data[index]: raise VerifyException("field result has incorrect type")


@irdl_op_definition
class FSMMachineOp(IRDLOperation):
    """Module-level controller declaration emitted by speculation lowering."""

    name = "spechls.fsm.machine"
    sym_name = attr_def(SymbolNameConstraint())
    function_type = attr_def(FunctionType)
    body = region_def()
    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface())
    assembly_format = "$sym_name `:` $function_type $body attr-dict"


@irdl_op_definition
class FSMControllerStateOp(IRDLOperation):
    """One state with constant command outputs and ordered transitions."""

    name = "spechls.fsm.state"
    state_name = attr_def(StringAttr, attr_name="name")
    output = region_def()
    transitions = region_def()
    assembly_format = "$name `output` $output `transitions` $transitions attr-dict"


@irdl_op_definition
class FSMOutputOp(IRDLOperation):
    """Named constant command vector produced by a controller state."""

    name = "spechls.fsm.output"
    names = attr_def(ArrayAttr[StringAttr])
    values = attr_def(DenseArrayBase.constr(i64))
    traits = traits_def(IsTerminator(), HasParent(FSMControllerStateOp))
    assembly_format = "$names $values attr-dict"

    def verify_(self):
        if len(self.names) != len(self.values):
            raise VerifyException("fsm output names and values must have equal lengths")


@irdl_op_definition
class FSMControllerTransitionOp(IRDLOperation):
    """One ordered transition; an empty guard is unconditional."""

    name = "spechls.fsm.transition"
    target = attr_def(StringAttr)
    kind = attr_def(StringAttr)
    guard = region_def()
    assembly_format = "$target $kind (`guard` $guard^)? attr-dict"


@irdl_op_definition
class FSMInputOp(IRDLOperation):
    name = "spechls.fsm.input"
    index = attr_def(IntegerAttr)
    result = result_def(AnyAttr())

    @classmethod
    def parse(cls, parser: Parser) -> "FSMInputOp":
        index = parser.parse_integer()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        return cls.create(
            result_types=[result_type],
            attributes={"index": IntegerAttr(index, i64)},
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(f" {self.index.value.data} : ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class FSMReturnOp(IRDLOperation):
    name = "spechls.fsm.return"
    condition = operand_def(i1)
    traits = traits_def(IsTerminator(), HasParent(FSMControllerTransitionOp))
    assembly_format = "$condition attr-dict"


@irdl_op_definition
class FSMInstanceOp(IRDLOperation):
    """Task-local binding of a controller declaration."""

    name = "spechls.fsm.instance"
    instance_name = attr_def(StringAttr, attr_name="name")
    machine = attr_def(FlatSymbolRefAttr)
    assembly_format = "$name $machine attr-dict"


@irdl_op_definition
class FSMTriggerOp(IRDLOperation):
    """Evaluate one controller step using flattened inputs and outputs."""

    name = "spechls.fsm.trigger"
    instance = attr_def(StringAttr)
    inputs = var_operand_def()
    result = var_result_def()
    assembly_format = "$instance `(` $inputs `)` attr-dict `:` functional-type($inputs, $result)"


@irdl_op_definition
class FSMOp(IRDLOperation):
    name = "spechls.fsm"
    name_attr = attr_def(StringAttr, attr_name="name"); gamma_names = attr_def(ArrayAttr[StringAttr]); cond_delays = attr_def(DenseArrayBase.constr(i64)); fast_indices = attr_def(DenseArrayBase.constr(i64)); input_delays = attr_def(ArrayAttr[DenseArrayBase]); mispec = operand_def(StructType); state = operand_def(StructType); result = result_def(StructType); body = region_def()
    assembly_format = "$name `<` $gamma_names `>` $cond_delays $fast_indices $input_delays `,` $mispec `,` $state attr-dict `:` type($mispec) `,` type($state) `->` type($result) $body"
    def verify_(self):
        if self.state.type != self.result.type: raise VerifyException("fsm state and result types must match")
        if len(self.body.blocks) != 1: raise VerifyException("fsm body must contain exactly one block")
        states = {state.name.data for state in self.body.block.ops if isinstance(state, FSMStateOp)}
        for transition in self.body.block.ops:
            if not isinstance(transition, FSMTransitionOp):
                continue
            if transition.source.data not in states or transition.target.data not in states:
                raise VerifyException("fsm transition endpoints must name states in the body")


@irdl_op_definition
class FSMStateOp(IRDLOperation):
    """One state and its ordered fsm-command output vector."""

    name = "spechls.fsm_state"
    state_name = attr_def(StringAttr, attr_name="name")
    commands = attr_def(DenseArrayBase.constr(i64))
    assembly_format = "$name $commands attr-dict"


@irdl_op_definition
class FSMTransitionOp(IRDLOperation):
    """Exact selector guard from one named FSM state to another."""

    name = "spechls.fsm_transition"
    source = attr_def(StringAttr)
    target = attr_def(StringAttr)
    kind = attr_def(StringAttr)
    input_ids = attr_def(DenseArrayBase.constr(i64))
    selectors = attr_def(DenseArrayBase.constr(i64))
    assembly_format = "$source `->` $target $kind $input_ids $selectors attr-dict"

    def verify_(self):
        if len(self.input_ids) != len(self.selectors):
            raise VerifyException("fsm transition input_ids and selectors must have equal lengths")


@irdl_op_definition
class FSMCommandOp(IRDLOperation):
    name = "spechls.fsm_command"
    command_name = attr_def(StringAttr, attr_name="name"); state = operand_def(StructType); result = result_def(StructType)
    assembly_format = "$name `,` $state attr-dict `:` functional-type($state, $result)"


class _ControlDataOp(_SameTypeOp):
    input = operand_def(_SameTypeOp.DATA); control = operand_def(); write_command = operand_def(i1); result = result_def(_SameTypeOp.DATA)


@irdl_op_definition
class RewindOp(_ControlDataOp):
    name = "spechls.rewind"
    depths = attr_def(DenseArrayBase.constr(i64))
    assembly_format = "`<` $depths `>` $input `,` $control `,` $write_command attr-dict `:` type($input) `,` type($control)"


@irdl_op_definition
class RollbackOp(_ControlDataOp):
    name = "spechls.rollback"
    depths = attr_def(DenseArrayBase.constr(i64)); offset = attr_def(IntegerAttr)
    assembly_format = "`<` $depths `,` $offset `>` $input `,` $control `,` $write_command attr-dict `:` type($input) `,` type($control)"


@irdl_op_definition
class CancelOp(IRDLOperation):
    name = "spechls.cancel"
    offset = attr_def(IntegerAttr); input = operand_def(i1); rollback = operand_def(); write_command = operand_def(i1); result = result_def(i1)
    assembly_format = "`<` $offset `>` $input `,` $rollback `,` $write_command attr-dict `:` type($rollback)"


SpecHLS = Dialect("spechls", [KernelOp, ExitOp, TaskOp, CommitOp, GammaOp, MuOp, PrintOp, CallOp, AlphaOp, LoadOp, LUTOp, DelayOp, CancellableDelayOp, RollbackableDelayOp, FIFOOp, PackOp, UnpackOp, SyncOp, FieldOp, FSMMachineOp, FSMControllerStateOp, FSMOutputOp, FSMControllerTransitionOp, FSMInputOp, FSMReturnOp, FSMInstanceOp, FSMTriggerOp, FSMOp, FSMStateOp, FSMTransitionOp, FSMCommandOp, RewindOp, RollbackOp, CancelOp], [ArrayType, StructType, SpeculationSlowPathAttr, SpeculationEntryAttr, SpeculationConfigAttr, SpeculationRecoveryAttr])
