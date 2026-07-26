"""Emit a checked Uclid abstraction for a simple speculative SpecHLS task.

The initial contract intentionally covers the single-gamma slow/fast shape used
by the end-to-end speculation test.  It models committed values, not transient
fast-path values: recovery may insert controller-only cycles, but a committed
value must equal the sequential ``step`` result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from xdsl.dialects.builtin import ArrayAttr, IntegerType, ModuleOp
from xdsl.ir import Operation, SSAValue

from spechls.dialect import (
    ArrayType,
    CallOp,
    CommitOp,
    DelayOp,
    FSMMachineOp,
    FSMControllerStateOp,
    FSMControllerTransitionOp,
    FSMInputOp,
    FSMOutputOp,
    FSMReturnOp,
    FSMTriggerOp,
    GammaOp,
    KernelOp,
    LoadOp,
    MuOp,
    RollbackOp,
    SyncOp,
    TaskOp,
)


class UclidEmissionError(ValueError):
    """The input is outside the intentionally supported proof abstraction."""


@dataclass(frozen=True)
class UclidVerificationBundle:
    """Three Uclid compilation units for a reference/speculation proof harness."""

    reference_module: str
    speculated_module: str
    driver_module: str

    def combined_source(self) -> str:
        """Return the single Uclid compilation unit formed by the three modules."""
        common, main = self.driver_module.split("module main {", 1)
        return "\n".join((common, self.reference_module, self.speculated_module, "module main {" + main))

    def write(self, directory: str | Path) -> tuple[Path, Path, Path]:
        """Write the reference, transformed, and harness units to ``directory``."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        files = (
            output / "reference.ucl",
            output / "speculated.ucl",
            output / "driver.ucl",
        )
        for path, source in zip(files, (self.reference_module, self.speculated_module, self.driver_module)):
            path.write_text(source, encoding="ascii")
        return files


@dataclass(frozen=True)
class _SlowFastTask:
    kernel_name: str
    width: int
    condition: str
    fast: str
    slow: str
    controller_states: tuple[str, ...] = ()


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not result or result[0].isdigit():
        result = "v_" + result
    return result


def _bit_width(value: SSAValue, role: str) -> int:
    if not isinstance(value.type, IntegerType):
        raise UclidEmissionError(f"{role} must have an integer type, got {value.type}")
    return value.type.width.data


def _callee(value: SSAValue, role: str) -> str:
    operation = value.owner
    if not isinstance(operation, CallOp):
        raise UclidEmissionError(f"{role} must be produced by spechls.call")
    if len(operation.arguments) != 1 or len(operation.result) != 1:
        raise UclidEmissionError(f"{role} call must have one argument and one result")
    return _identifier(operation.callee.root_reference.data)


def _selected_kernel(module: ModuleOp, kernel_name: str | None) -> KernelOp:
    kernels = [operation for operation in module.body.block.ops if isinstance(operation, KernelOp)]
    if kernel_name is not None:
        kernels = [kernel for kernel in kernels if kernel.sym_name.data == kernel_name]
    if len(kernels) != 1:
        description = f"named '{kernel_name}'" if kernel_name is not None else "in the module"
        raise UclidEmissionError(f"expected exactly one spechls.kernel {description}")
    return kernels[0]


def _extract_task(module: ModuleOp, kernel_name: str | None) -> _SlowFastTask:
    kernel = _selected_kernel(module, kernel_name)
    tasks = [operation for operation in kernel.body.block.ops if isinstance(operation, TaskOp)]
    if len(tasks) != 1:
        raise UclidEmissionError("the selected kernel must contain exactly one spechls.task")
    task = tasks[0]
    operations = tuple(task.body.block.ops)
    gammas = [operation for operation in operations if isinstance(operation, GammaOp)]
    commits = [operation for operation in operations if isinstance(operation, CommitOp)]
    triggers = [operation for operation in operations if isinstance(operation, FSMTriggerOp)]
    if len(gammas) != 1 or len(commits) != 1:
        raise UclidEmissionError("the task must contain exactly one gamma and commit")

    gamma = gammas[0]
    commit = commits[0]
    if len(gamma.inputs) != 2:
        raise UclidEmissionError("the gamma must have exactly fast and slow inputs")

    width = _bit_width(gamma.result, "gamma result")
    if not triggers:
        mus = [operation for operation in operations if isinstance(operation, MuOp)]
        if len(mus) != 1:
            raise UclidEmissionError("the source task must contain exactly one mu")
        mu = mus[0]
        if gamma.result != mu.loop_value:
            raise UclidEmissionError("the mu loop value must be the gamma result")
        if gamma.result not in commit.value:
            raise UclidEmissionError("the commit must include the gamma result")
        if _bit_width(mu.result, "mu result") != width:
            raise UclidEmissionError("mu and gamma widths must match")
        return _SlowFastTask(
            _identifier(kernel.sym_name.data), width, _callee(gamma.select, "gamma selector"),
            _callee(gamma.inputs[0], "gamma fast input"), _callee(gamma.inputs[1], "gamma slow input"),
        )

    if len(triggers) != 1 or gamma.select.owner is not triggers[0]:
        raise UclidEmissionError("the transformed gamma selector must come from one fsm trigger")
    machines = [operation for operation in module.body.block.ops if isinstance(operation, FSMMachineOp)]
    if len(machines) != 1:
        raise UclidEmissionError("the transformed task requires exactly one fsm machine")
    states = tuple(
        state.state_name.data for state in machines[0].body.block.ops
        if isinstance(state, FSMControllerStateOp)
    )
    if not {"Init_0", "Init_1", "Proceed"}.issubset(states):
        raise UclidEmissionError("the transformed fsm must contain Init_0, Init_1, and Proceed")
    rollback = [state for state in states if "Rollback" in state]
    fill = [state for state in states if "Fill" in state]
    if len(rollback) != 1 or len(fill) != 1:
        raise UclidEmissionError("the transformed fsm must contain one rollback and one fill state")
    calls = {
        _identifier(operation.callee.root_reference.data): operation
        for operation in operations if isinstance(operation, CallOp)
    }
    required = {"cond", "fast", "slow"}
    if not required <= calls.keys():
        raise UclidEmissionError("the transformed task must call cond, fast, and slow")
    return _SlowFastTask(
        _identifier(kernel.sym_name.data), width, "cond", "fast", "slow", states,
    )


def _state_identifier(name: str) -> str:
    return _identifier(name).upper()


def _emit_transformed_uclid(task: _SlowFastTask, induction_depth: int) -> str:
    value_type = f"bv{task.width}"
    states = {_state_identifier(state): state for state in task.controller_states}
    init_0, init_1, proceed = (_state_identifier(name) for name in ("Init_0", "Init_1", "Proceed"))
    rollback = next(name for name, original in states.items() if "Rollback" in original)
    fill = next(name for name, original in states.items() if "Fill" in original)
    enum_values = ", ".join(states)
    return f'''// Generated by spechls.uclid.emit_uclid for transformed kernel {task.kernel_name}. Do not edit.
module main {{
  function {task.condition}(a: {value_type}): boolean;
  function {task.fast}(a: {value_type}): {value_type};
  function {task.slow}(a: {value_type}): {value_type};

  define step(x: {value_type}) : {value_type} =
    if ({task.condition}(x)) then {task.slow}(x) else {task.fast}(x);

  // This enum is the transformed fsm.machine state table, with Uclid-safe names.
  type controller_state = enum {{ {enum_values} }};
  var input_value: {value_type};
  var reference_value: {value_type};
  var committed_value: {value_type};
  var controller: controller_state;

  init {{
    reference_value = input_value;
    committed_value = input_value;
    controller = {init_0};
  }}

  next {{
    case
      controller == {init_0}: {{
        reference_value' = reference_value;
        committed_value' = committed_value;
        controller' = {init_1};
      }}
      controller == {init_1}: {{
        reference_value' = reference_value;
        committed_value' = committed_value;
        controller' = {proceed};
      }}
      controller == {proceed}: {{
        reference_value' = step(reference_value);
        committed_value' = step(committed_value);
        if ({task.condition}(committed_value)) {{
          controller' = {rollback};
        }} else {{
          controller' = {proceed};
        }}
      }}
      controller == {rollback}: {{
        reference_value' = reference_value;
        committed_value' = committed_value;
        controller' = {fill};
      }}
      default: {{
        reference_value' = reference_value;
        committed_value' = committed_value;
        controller' = {proceed};
      }}
    esac;
  }}

  invariant committed_matches_reference: committed_value == reference_value;

  control {{
    vobj = induction({induction_depth});
    check;
    print_results;
    vobj.print_cex(reference_value, committed_value, controller);
  }}
}}
'''


class _DataflowUclidEmitter:
    """Lower one parsed transformed task into explicit Uclid state and expressions.

    The emitter separates persistent hardware state from SSA values.  ``mu``,
    delay, and rollback history become Uclid variables updated in ``next``;
    every other supported result is emitted as a Uclid variable so generated
    models expose the original dataflow graph rather than a replacement step.
    """

    def __init__(self, module: ModuleOp, kernel: KernelOp, task: TaskOp, depth: int):
        self.module = module
        self.kernel = kernel
        self.task = task
        self.depth = depth
        self.names: dict[SSAValue, str] = {}
        self.state_values: set[SSAValue] = set()
        # Module variables represent hardware state that persists between ticks.
        # Pure SSA results instead live only within the generated tick procedure.
        self.variables: list[str] = []
        self.state_names: list[str] = []
        self.local_variables: list[str] = []
        self.local_assignments: list[str] = []
        self.initializers: list[str] = []
        self.updates: list[str] = []
        self.functions: dict[str, tuple[tuple[str, ...], str]] = {}
        self.procedures: list[str] = []
        self.commit_stream_emitted = False
        self.fsm_inputs: dict[SSAValue, SSAValue] = {}

    def type_of(self, value: SSAValue) -> str:
        """Map scalar and array xDSL values to the corresponding Uclid types."""
        if isinstance(value.type, ArrayType):
            element = value.type.element_type
            if not isinstance(element, IntegerType):
                raise UclidEmissionError(f"unsupported Uclid array element type {element}")
            element_type = "boolean" if element.width.data == 1 else f"bv{element.width.data}"
            return f"[bv32] {element_type}"
        if not isinstance(value.type, IntegerType):
            raise UclidEmissionError(f"unsupported Uclid value type {value.type}")
        return "boolean" if value.type.width.data == 1 else f"bv{value.type.width.data}"

    def zero(self, value: SSAValue) -> str:
        """Return the typed zero literal used for uninitialized hardware state."""
        return "false" if self.type_of(value) == "boolean" else f"0{self.type_of(value)}"

    def value_name(self, value: SSAValue) -> str:
        """Return the stable Uclid name assigned to an SSA value."""
        if value not in self.names:
            self.names[value] = f"v_{len(self.names)}"
        return self.names[value]

    def expression(self, value: SSAValue) -> str:
        """Resolve a supported SSA value to its generated Uclid expression."""
        if isinstance(value.owner, FSMInputOp):
            if value not in self.fsm_inputs:
                raise UclidEmissionError("fsm input is used outside a trigger guard")
            return self.expression(self.fsm_inputs[value])
        if isinstance(value.owner, Operation) and value.owner.name in {"hw.constant", "arith.constant"}:
            constant = value.owner.value.value.data
            if self.type_of(value) == "boolean":
                return "true" if constant else "false"
            return f"{constant}{self.type_of(value)}"
        return self.value_name(value)

    def add_define(self, value: SSAValue, expression: str) -> None:
        """Materialize one pure SSA result as a tick-local Uclid temporary."""
        name = self.value_name(value)
        self.local_variables.append(f"    var {name}: {self.type_of(value)};")
        self.local_assignments.append(f"    {name} = {expression};")

    def add_named_value(self, name: str, value: SSAValue, expression: str) -> None:
        """Expose a named non-SSA observation, such as one commit field."""
        self.variables.append(f"  var {name}: {self.type_of(value)};")
        self.state_names.append(name)
        self.initializers.append(f"    {name} = {self.zero(value)};")
        self.updates.append(f"    {name}' = {expression};")

    def call_expression(self, operation: CallOp) -> str:
        """Declare an external call as a typed uninterpreted Uclid function."""
        name = _identifier(operation.callee.root_reference.data)
        arguments = tuple(self.type_of(argument) for argument in operation.arguments)
        result = self.type_of(operation.result[0])
        previous = self.functions.get(name)
        if previous is not None and previous != (arguments, result):
            raise UclidEmissionError(f"inconsistent signatures for call '{name}'")
        self.functions[name] = (arguments, result)
        return f"{name}({', '.join(self.expression(argument) for argument in operation.arguments)})"

    def arithmetic_expression(self, operation: Operation) -> str:
        """Lower common arith/comb operations while preserving bit-vector operands."""
        operands = [self.expression(value) for value in operation.operands]
        binary = {
            "arith.addi": "+", "arith.subi": "-", "arith.muli": "*",
            "arith.andi": "&", "arith.ori": "|", "arith.xori": "^",
            "arith.divsi": "/", "arith.divui": "/", "arith.remsi": "%", "arith.remui": "%",
            "arith.shli": "<<", "arith.shrsi": ">>", "arith.shrui": ">>",
            "comb.add": "+", "comb.sub": "-", "comb.mul": "*",
            "comb.and": "&", "comb.or": "|", "comb.xor": "^", "comb.divu": "/",
            "comb.divs": "/", "comb.modu": "%", "comb.mods": "%", "comb.shl": "<<",
            "comb.shru": ">>", "comb.shrs": ">>",
        }
        if operation.name in binary and len(operands) == 2:
            return f"({operands[0]} {binary[operation.name]} {operands[1]})"
        if operation.name in {"arith.cmpi", "comb.icmp"} and len(operands) == 2:
            predicate = getattr(operation, "predicate", None)
            code = predicate.value.data if predicate is not None else 0
            comparison = {0: "==", 1: "!=", 2: "<", 3: "<=", 4: ">", 5: ">="}.get(code, "==")
            return f"({operands[0]} {comparison} {operands[1]})"
        if operation.name == "arith.select" and len(operands) == 3:
            return f"if ({operands[0]}) then {operands[1]} else {operands[2]}"
        raise UclidEmissionError(f"unsupported arithmetic operation {operation.name}")

    def same_cycle_expression(self, value: SSAValue) -> str:
        """Inline pure SSA producers when they control a sequential operation.

        SSA variables remain visible state in the generated model, but a delay
        enable must use the value computed in its source cycle, not the value
        captured by an extra artificial SSA register in the prior cycle.
        """
        operation = value.owner
        if operation.name in {"hw.constant", "arith.constant"}:
            constant = operation.value.value.data
            if self.type_of(value) == "boolean":
                return "true" if constant else "false"
            return f"{constant}{self.type_of(value)}"
        if isinstance(operation, SyncOp):
            return self.same_cycle_expression(operation.inputs[0])
        if isinstance(operation, CallOp):
            return self.call_expression(operation)
        if isinstance(operation, GammaOp):
            selector = self.same_cycle_expression(operation.select)
            if self.type_of(operation.select) == "boolean" and len(operation.inputs) == 2:
                return (
                    f"if ({selector}) then {self.same_cycle_expression(operation.inputs[1])} "
                    f"else {self.same_cycle_expression(operation.inputs[0])}"
                )
            expression = self.same_cycle_expression(operation.inputs[-1])
            for index in reversed(range(len(operation.inputs) - 1)):
                expression = (
                    f"if ({selector} == {index}{self.type_of(operation.select)}) then "
                    f"{self.same_cycle_expression(operation.inputs[index])} else {expression}"
                )
            return expression
        if operation.name in {"arith.cmpi", "comb.icmp"}:
            operands = [self.same_cycle_expression(operand) for operand in operation.operands]
            predicate = getattr(operation, "predicate", None)
            code = predicate.value.data if predicate is not None else 0
            comparison = {0: "==", 1: "!=", 2: "<", 3: "<=", 4: ">", 5: ">="}.get(code, "==")
            return f"({operands[0]} {comparison} {operands[1]})"
        return self.expression(value)

    def lower_fsm(self, trigger: FSMTriggerOp) -> None:
        """Emit controller next-state and command equations from ``fsm.machine``."""
        machines = [operation for operation in self.module.body.block.ops if isinstance(operation, FSMMachineOp)]
        if len(machines) != 1:
            raise UclidEmissionError("a transformed task requires exactly one fsm machine")
        machine = machines[0]
        states = [state for state in machine.body.block.ops if isinstance(state, FSMControllerStateOp)]
        first_output = states[0].output.block.last_op
        if not isinstance(first_output, FSMOutputOp):
            raise UclidEmissionError("fsm state output must end in spechls.fsm.output")
        self.names[trigger.result[0]] = f"fsm_{_identifier(trigger.instance.data)}_next_state"
        for result, field in zip(trigger.result[1:], first_output.names):
            self.names[result] = f"fsm_{_identifier(trigger.instance.data)}_{_identifier(field.data)}"
        state_ids = {state.state_name.data: index for index, state in enumerate(states)}
        current_state = self.expression(trigger.inputs[0])
        for state in states:
            for transition in state.transitions.block.ops:
                if isinstance(transition, FSMControllerTransitionOp) and not transition.guard.blocks:
                    continue
                if isinstance(transition, FSMControllerTransitionOp) and transition.guard.blocks:
                    returned = transition.guard.block.last_op
                    if isinstance(returned, FSMReturnOp):
                        for operation in transition.guard.block.ops:
                            if isinstance(operation, FSMInputOp):
                                self.fsm_inputs[operation.result] = trigger.inputs[operation.index.value.data]
                        for operation in transition.guard.block.ops:
                            if operation.name in {"arith.constant", "hw.constant"}:
                                # Guard constants are inlined by expression().
                                continue
                            elif operation.name.startswith(("arith.", "comb.")):
                                self.add_define(operation.result, self.arithmetic_expression(operation))
        next_cases: list[str] = []
        command_count = len(trigger.result) - 1
        command_cases: list[list[str]] = [[] for _ in range(command_count)]
        for state in states:
            state_id = f"{state_ids[state.state_name.data]}bv32"
            transitions = [op for op in state.transitions.block.ops if isinstance(op, FSMControllerTransitionOp)]
            guarded = [op for op in transitions if op.guard.blocks]
            fallback = next((op for op in transitions if not op.guard.blocks), None)
            next_expression = f"{state_ids[fallback.target.data]}bv32" if fallback else state_id
            for transition in reversed(guarded):
                returned = transition.guard.block.last_op
                if not isinstance(returned, FSMReturnOp):
                    raise UclidEmissionError("fsm guard must end in spechls.fsm.return")
                next_expression = f"if ({self.expression(returned.condition)}) then {state_ids[transition.target.data]}bv32 else {next_expression}"
            next_cases.append(f"if ({current_state} == {state_id}) then {next_expression}")
            output = state.output.block.last_op
            if not isinstance(output, FSMOutputOp):
                raise UclidEmissionError("fsm state output must end in spechls.fsm.output")
            values = list(output.values.get_values())
            if len(values) != command_count:
                raise UclidEmissionError("fsm output width does not match trigger results")
            for index, constant in enumerate(values):
                result = trigger.result[index + 1]
                literal = "true" if self.type_of(result) == "boolean" and constant else "false" if self.type_of(result) == "boolean" else f"{constant}{self.type_of(result)}"
                command_cases[index].append(f"if ({current_state} == {state_id}) then {literal}")
        next_expression = current_state
        for case in reversed(next_cases):
            condition, value = case.removeprefix("if (").split(") then ", 1)
            next_expression = f"if ({condition}) then {value} else {next_expression}"
        self.add_define(trigger.result[0], next_expression)
        for index, cases in enumerate(command_cases):
            result = trigger.result[index + 1]
            expression = self.zero(result)
            for case in reversed(cases):
                condition, value = case.removeprefix("if (").split(") then ", 1)
                expression = f"if ({condition}) then {value} else {expression}"
            self.add_define(result, expression)

    def lower_operation(self, operation: Operation) -> None:
        """Lower one task operation and register its state updates or definitions."""
        if isinstance(operation, MuOp):
            result = operation.result
            name = f"mu_{_identifier(operation.sym_name.data)}"
            self.names[result] = name
            self.state_values.add(result)
            self.variables.append(f"  var {name}: {self.type_of(result)};")
            self.state_names.append(name)
            self.initializers.append(f"    {name} = {self.expression(operation.init_value)};")
            self.updates.append(f"    {name}' = {self.expression(operation.loop_value)};")
        elif isinstance(operation, FSMTriggerOp):
            self.lower_fsm(operation)
        elif isinstance(operation, DelayOp):
            output_name = self.value_name(operation.result)
            base = output_name.rsplit("_", 1)[0]
            stages = [f"{base}_{index}" for index in range(operation.depth.value.data)]
            for stage in stages:
                self.variables.append(f"  var {stage}: {self.type_of(operation.result)};")
                self.state_names.append(stage)
                self.initializers.append(f"    {stage} = {self.zero(operation.result)};")
            enable = self.same_cycle_expression(operation.enable) if operation.enable is not None else "true"
            self.updates.append(f"    {stages[0]}' = if ({enable}) then {self.same_cycle_expression(operation.input)} else {stages[0]};")
            for index in range(1, len(stages)):
                self.updates.append(f"    {stages[index]}' = if ({enable}) then {stages[index - 1]} else {stages[index]};")
            self.names[operation.result] = stages[-1]
            self.state_values.add(operation.result)
        elif isinstance(operation, RollbackOp):
            depth = max(operation.depths.get_values(), default=1)
            stages = [f"rollback_{self.value_name(operation.result)}_{index}" for index in range(depth)]
            for stage in stages:
                self.variables.append(f"  var {stage}: {self.type_of(operation.result)};")
                self.state_names.append(stage)
                self.initializers.append(f"    {stage} = {self.zero(operation.result)};")
            input_value = self.same_cycle_expression(operation.input)
            self.updates.append(f"    {stages[0]}' = {input_value};")
            for index in range(1, len(stages)):
                self.updates.append(f"    {stages[index]}' = {stages[index - 1]};")
            restored = f"if ({self.expression(operation.control)} == 0{self.type_of(operation.control)}) then {input_value} else {stages[-1]}"
            self.add_define(operation.result, f"if ({self.expression(operation.write_command)}) then {restored} else {input_value}")
        elif isinstance(operation, SyncOp):
            self.add_define(operation.result, self.expression(operation.inputs[0]))
        elif isinstance(operation, CallOp):
            self.add_define(operation.result[0], self.call_expression(operation))
        elif isinstance(operation, GammaOp):
            selector = self.expression(operation.select)
            if self.type_of(operation.select) == "boolean" and len(operation.inputs) == 2:
                expression = f"if ({selector}) then {self.expression(operation.inputs[1])} else {self.expression(operation.inputs[0])}"
            else:
                expression = self.expression(operation.inputs[-1])
                for index in reversed(range(len(operation.inputs) - 1)):
                    expression = f"if ({selector} == {index}{self.type_of(operation.select)}) then {self.expression(operation.inputs[index])} else {expression}"
            self.add_define(operation.result, expression)
        elif isinstance(operation, CommitOp):
            if len(operation.value) != 2 or self.type_of(operation.value[0]) != "boolean":
                raise UclidEmissionError("spechls.commit requires a boolean guard followed by one committed value")
            self.add_named_value("commit_guard", operation.value[0], self.expression(operation.value[0]))
            self.add_named_value("commit_value", operation.value[1], self.expression(operation.value[1]))
            if self.commit_stream_emitted:
                raise UclidEmissionError("only one architectural spechls.commit is supported per task")
            self.commit_stream_emitted = True
            value_type = self.type_of(operation.value[1])
            self.variables.extend((
                "  var committed_count: integer;",
                f"  var committed_stream: [integer] {value_type};",
            ))
            self.state_names.extend(("committed_count", "committed_stream"))
            self.initializers.extend((
                "    committed_count = 0;",
                f"    committed_stream[0] = {self.zero(operation.value[1])};",
            ))
            self.procedures.append(
                f"  procedure write_committed(index: integer, value: {value_type}) modifies committed_stream; {{\n"
                "    committed_stream[index] = value;\n"
                "  }"
            )
            self.updates.extend((
                "    if (commit_guard) {",
                "      committed_count' = committed_count + 1;",
                "      call write_committed(committed_count + 1, commit_value);",
                "    } else {",
                "      committed_count' = committed_count;",
                "      committed_stream' = committed_stream;",
                "    }",
            ))
        elif operation.name in {"hw.constant", "arith.constant"}:
            # A literal has no temporal state and must not become an SSA register.
            return
        elif operation.name.startswith(("arith.", "comb.")):
            self.add_define(operation.result, self.arithmetic_expression(operation))
        elif isinstance(operation, LoadOp):
            self.add_define(operation.result, f"{self.expression(operation.array)}[{self.expression(operation.index)}]")
        elif operation.name == "spechls.fsm.instance":
            return
        else:
            raise UclidEmissionError(f"unsupported transformed operation {operation.name}")

    def emit(self) -> str:
        """Generate a complete Uclid module for the selected transformed task."""
        for argument in self.task.body.block.args:
            name = f"input_{len(self.names)}"
            self.names[argument] = name
            self.state_values.add(argument)
            self.variables.append(f"  var {name}: {self.type_of(argument)};")
            self.state_names.append(name)
            self.initializers.append(f"    {name} = {self.zero(argument)};")
        # Assign loop-carried names before traversing operations because rollback
        # nodes are intentionally scheduled before the mu they reference.
        for operation in self.task.body.block.ops:
            if isinstance(operation, MuOp):
                self.names[operation.result] = f"mu_{_identifier(operation.sym_name.data)}"
                self.state_values.add(operation.result)
        for index, operation in enumerate(self.task.body.block.ops):
            if isinstance(operation, DelayOp):
                self.names[operation.result] = f"delay_{index}_{operation.depth.value.data - 1}"
                self.state_values.add(operation.result)
        machine = next((operation for operation in self.module.body.block.ops if isinstance(operation, FSMMachineOp)), None)
        if machine is not None:
            first_state = next((state for state in machine.body.block.ops if isinstance(state, FSMControllerStateOp)), None)
            first_output = first_state.output.block.last_op if first_state is not None else None
            if isinstance(first_output, FSMOutputOp):
                for operation in self.task.body.block.ops:
                    if isinstance(operation, FSMTriggerOp):
                        self.names[operation.result[0]] = f"fsm_{_identifier(operation.instance.data)}_next_state"
                        for result, field in zip(operation.result[1:], first_output.names):
                            self.names[result] = f"fsm_{_identifier(operation.instance.data)}_{_identifier(field.data)}"
        for operation in self.task.body.block.ops:
            self.lower_operation(operation)
        functions = [
            f"  function {name}({', '.join(f'a{index}: {type}' for index, type in enumerate(arguments))}): {result};"
            for name, (arguments, result) in sorted(self.functions.items())
        ]
        exposure_map = ["  // Dataflow exposure map: each parsed SSA result maps to the Uclid symbol below."]
        machine = next((operation for operation in self.module.body.block.ops if isinstance(operation, FSMMachineOp)), None)
        for index, operation in enumerate(self.task.body.block.ops):
            if isinstance(operation, CommitOp):
                exposure_map.append(
                    f"  // [{index}] spechls.commit -> commit_guard, commit_value"
                )
                continue
            if isinstance(operation, FSMTriggerOp) and machine is not None:
                state = next(
                    (state for state in machine.body.block.ops if isinstance(state, FSMControllerStateOp)),
                    None,
                )
                output = state.output.block.last_op if state is not None else None
                fields = ["next_state"]
                if isinstance(output, FSMOutputOp):
                    fields.extend(name.data for name in output.names)
                bindings = ", ".join(
                    f"{field}={self.expression(result)}"
                    for field, result in zip(fields, operation.result)
                )
                exposure_map.append(f"  // [{index}] spechls.fsm.trigger -> {bindings}")
                continue
            results = ", ".join(self.expression(result) for result in operation.results)
            if results:
                if isinstance(operation, MuOp):
                    label = f"spechls.mu<{operation.sym_name.data}>"
                elif isinstance(operation, GammaOp):
                    label = f"spechls.gamma<{operation.sym_name.data}>"
                elif isinstance(operation, CallOp):
                    label = f"spechls.call @{operation.callee.root_reference.data}"
                elif isinstance(operation, DelayOp):
                    label = f"spechls.delay depth={operation.depth.value.data}"
                elif isinstance(operation, RollbackOp):
                    label = f"spechls.rollback depths={list(operation.depths.get_values())}"
                else:
                    label = operation.name
                exposure_map.append(f"  // [{index}] {label} -> {results}")
        # Uclid procedures execute sequentially, whereas hardware registers update
        # simultaneously.  Stage each scalar register's next value locally so all
        # RHS expressions observe the pre-clock state; commit the staged values
        # only after dataflow and storage-update equations have been evaluated.
        staged_names = [
            name for name in self.state_names
            if name not in {"input_0", "committed_stream"}
        ]
        state_declarations = {
            declaration.removeprefix("  var ").split(": ", 1)[0]: declaration
            for declaration in self.variables
        }
        staged_variables = [
            f"    var next_{name}: {state_declarations[name].split(': ', 1)[1]}"
            for name in staged_names
        ]
        # The declarations above retain the declared Uclid type without trying to
        # infer it from MLIR a second time.  There is exactly one module declaration
        # for each persistent state name.
        staged_initializers = [f"    next_{name} = {name};" for name in staged_names]
        staged_updates = []
        for update in self.updates:
            staged = update
            for name in staged_names:
                staged = staged.replace(f"{name}'", f"next_{name}")
            staged = staged.replace("committed_stream'", "committed_stream")
            staged = staged.replace("if (commit_guard)", "if (next_commit_guard)")
            staged = staged.replace(", commit_value);", ", next_commit_value);")
            staged_updates.append(staged)
        staged_commits = [f"    {name} = next_{name};" for name in staged_names]
        tick = [
            f"  procedure tick() modifies {', '.join(self.state_names)}; {{",
            *self.local_variables,
            *staged_variables,
            *self.local_assignments,
            *staged_initializers,
            *staged_updates,
            *staged_commits,
            "  }",
        ]
        return "\n".join([
            f"// Generated from transformed SpecHLS kernel {self.kernel.sym_name.data}. Do not edit.",
            "module main {", *functions, *exposure_map, *self.variables, *self.procedures, "",
            "  init {", *self.initializers, "  }", "", *tick, "", "  next {", "    call tick();", "  }", "}", "",
        ])


def emit_uclid(module: ModuleOp, kernel_name: str | None = None, induction_depth: int = 4) -> str:
    """Return a Uclid model proving committed slow/fast values match ``step``.

    The generated controller has a normal proceed state and two recovery-only
    states.  Its committed state advances only through the selected semantic
    result, so recovery timing cannot expose a fast-path value as committed.
    """
    if induction_depth < 1:
        raise ValueError("induction_depth must be positive")
    kernel = _selected_kernel(module, kernel_name)
    tasks = [operation for operation in kernel.body.block.ops if isinstance(operation, TaskOp)]
    if len(tasks) == 1:
        if all(isinstance(operation, CommitOp) for operation in tasks[0].body.block.ops):
            raise UclidEmissionError("the task must contain modeled dataflow before its commit")
        return _DataflowUclidEmitter(module, kernel, tasks[0], induction_depth).emit()
    task = _extract_task(module, kernel_name)
    if task.controller_states:
        return _emit_transformed_uclid(task, induction_depth)
    value_type = f"bv{task.width}"
    return f'''// Generated by spechls.uclid.emit_uclid for kernel {task.kernel_name}. Do not edit.
module main {{
  function {task.condition}(a: {value_type}): boolean;
  function {task.fast}(a: {value_type}): {value_type};
  function {task.slow}(a: {value_type}): {value_type};

  define step(x: {value_type}) : {value_type} =
    if ({task.condition}(x)) then {task.slow}(x) else {task.fast}(x);

  type controller_state = enum {{ FILL, PROCEED, ROLLBACK }};
  var input_value: {value_type};
  var reference_value: {value_type};
  var committed_value: {value_type};
  var controller: controller_state;

  init {{
    reference_value = input_value;
    committed_value = input_value;
    controller = FILL;
  }}

  next {{
    case
      controller == FILL: {{
        reference_value' = reference_value;
        committed_value' = committed_value;
        controller' = PROCEED;
      }}
      controller == PROCEED: {{
        reference_value' = step(reference_value);
        committed_value' = step(committed_value);
        if ({task.condition}(committed_value)) {{
          controller' = ROLLBACK;
        }} else {{
          controller' = PROCEED;
        }}
      }}
      default: {{
        reference_value' = reference_value;
        committed_value' = committed_value;
        controller' = FILL;
      }}
    esac;
  }}

  invariant committed_matches_reference: committed_value == reference_value;

  control {{
    vobj = induction({induction_depth});
    check;
    print_results;
    vobj.print_cex(reference_value, committed_value, controller);
  }}
}}
'''


def _rename_module(source: str, name: str) -> str:
    """Rename the generated top-level module without inspecting or modifying MLIR."""
    return source.replace("module main {", f"module {name} {{", 1)


def _share_external_functions(reference: str, speculated: str) -> tuple[str, str, str]:
    """Hoist identical call declarations into one Uclid module shared by both models."""
    declarations: list[str] = []
    names: list[str] = []

    def rewrite(source: str) -> str:
        kept = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("function "):
                if line not in declarations:
                    declarations.append(line)
                names.append(stripped.split("(", 1)[0].split()[1])
            else:
                kept.append(line)
        result = "\n".join(kept)
        for name in set(names):
            result = result.replace(f"{name}(", f"common.{name}(")
        return result

    reference = rewrite(reference)
    speculated = rewrite(speculated)
    # Each generated slowfast model has one bv32 IR input.  A zero-argument
    # uninterpreted function is an arbitrary but shared initial value, unlike
    # two independent local havoc assignments.
    reference = reference.replace("input_0 = 0bv32;", "input_0 = common.initial_input_bv32();")
    speculated = speculated.replace("input_0 = 0bv32;", "input_0 = common.initial_input_bv32();")
    common = "\n".join((
        "module common {",
        "  function initial_input_bv32(): bv32;",
        *declarations,
        "}",
        "",
    ))
    return common, reference, speculated


def emit_stuttering_bisimulation_driver(
    reference: ModuleOp, speculated: ModuleOp, *, induction_depth: int = 12
) -> UclidVerificationBundle:
    """Build reference, transformed, and harness modules for a stuttering proof.

    The harness advances both instances on every physical clock.  Reference
    progress is observed through ``commit_value`` in both generated models while the transformed model
    exposes ``commit_guard`` and ``commit_value``.  A disabled transformed commit is a
    stuttering step; an enabled one must agree with the reference observation.
    """
    reference_module = _rename_module(emit_uclid(reference, induction_depth=induction_depth), "ReferenceModel")
    speculated_module = _rename_module(emit_uclid(speculated, induction_depth=induction_depth), "SpeculatedModel")
    common, reference_module, speculated_module = _share_external_functions(reference_module, speculated_module)
    driver_module = common + f'''module main {{
  instance reference: ReferenceModel();
  instance speculated: SpeculatedModel();

  var cycle: integer;
  var ref_x: [integer] bv32;
  var spec_x: [integer] bv32;

  procedure record_reference(index: integer, value: bv32) modifies ref_x; {{
    ref_x[index] = value;
  }}
  procedure record_speculated(index: integer, value: bv32) modifies spec_x; {{
    spec_x[index] = value;
  }}

  init {{
    cycle = 0;
    ref_x[0] = 0bv32;
    spec_x[0] = 0bv32;
  }}

  next {{
    next(speculated);
    next(reference);
    cycle' = cycle + 1;
    call record_reference(cycle + 1, reference.commit_value);
    call record_speculated(cycle + 1, speculated.commit_value);
  }}

  // If both executions agree at cycle n, the source value at n + 1 must
  // reappear in the speculative execution within the bounded recovery window.
  // Internal speculative cycles are ignored; only visible commits are sampled.
  invariant bounded_stuttering_bisimulation:
    speculated.commit_guard && ref_x[cycle] == spec_x[cycle] ==>
      (exists (k: integer) :: 0 <= k && k <= 5 &&
        ref_x[cycle + 1] == spec_x[cycle + k]);

  control {{
    vobj = induction({induction_depth});
    check;
    print_results;
    vobj.print_cex(cycle, ref_x, spec_x, reference.commit_value, speculated.commit_value);
  }}
}}
'''
    return UclidVerificationBundle(reference_module, speculated_module, driver_module)


__all__ = ["UclidEmissionError", "UclidVerificationBundle", "emit_stuttering_bisimulation_driver", "emit_uclid"]
