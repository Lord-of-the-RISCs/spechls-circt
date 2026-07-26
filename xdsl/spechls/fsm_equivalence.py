"""Limited, explicit Java/xDSL FSM equivalence checks.

The JSON model intentionally covers only constant state commands and exact
input-selector guards.  It is a comparison harness, not a claim that all
Java controller behavior is represented by an ``fsm.machine``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from xdsl.dialects import arith, fsm

from spechls.dialect import SpeculationRecoveryAttr


RECOVERY_SCENARIOS_ATTR = "spechls.recovery_scenarios"


@dataclass(frozen=True)
class NormalizedTransition:
    destination: str
    selectors: tuple[tuple[int, int], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"destination": self.destination, "selectors": [list(item) for item in self.selectors]}


@dataclass(frozen=True)
class NormalizedState:
    name: str
    commands: tuple[tuple[str, bool | int], ...]
    transitions: tuple[NormalizedTransition, ...]

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "commands": dict(self.commands), "transitions": [item.to_json() for item in self.transitions]}


@dataclass(frozen=True)
class NormalizedFSM:
    """JSON-compatible subset shared by Java fixture exports and xDSL."""

    initial_state: str
    inputs: tuple[str, ...]
    states: tuple[NormalizedState, ...]
    recovery_scenarios: tuple[dict[str, Any], ...] = ()
    unsupported_dimensions: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state,
            "inputs": list(self.inputs),
            "states": [state.to_json() for state in self.states],
            "recovery_scenarios": list(self.recovery_scenarios),
            "unsupported_dimensions": list(self.unsupported_dimensions),
        }


@dataclass(frozen=True)
class ComparisonResult:
    structural_differences: tuple[str, ...] = ()
    trace_differences: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()

    @property
    def equivalent(self) -> bool:
        return not self.structural_differences and not self.trace_differences and not self.gaps


def load_java_fsm_fixture(path: str | Path) -> NormalizedFSM:
    """Load the documented JSON fixture format emitted by a Java-side exporter."""
    with Path(path).open(encoding="utf-8") as fixture:
        data = json.load(fixture)
    required = {"initial_state", "inputs", "states"}
    if not required <= data.keys():
        raise ValueError(f"Java FSM fixture requires {sorted(required)}")
    states = tuple(
        NormalizedState(
            state["name"],
            tuple(sorted(state.get("commands", {}).items())),
            tuple(sorted(
                [NormalizedTransition(item["destination"], tuple(tuple(pair) for pair in item.get("selectors", ())))
                 for item in state.get("transitions", ())],
                key=lambda item: (item.destination, item.selectors),
            )),
        )
        for state in data["states"]
    )
    return NormalizedFSM(data["initial_state"], tuple(data["inputs"]), states, tuple(data.get("recovery_scenarios", ())), tuple(data.get("unsupported_dimensions", ())))


def _constant_value(operation: arith.ConstantOp, boolean: bool) -> bool | int:
    value = operation.value.value.data
    return bool(value) if boolean else value


def _selector_guard(transition: fsm.TransitionOp, inputs: Sequence[object]) -> tuple[tuple[int, int], ...] | None:
    if not transition.guard.blocks:
        return ()
    result = transition.guard.block.last_op
    if not isinstance(result, fsm.ReturnOp) or not result.operands:
        return None

    def collect(value) -> list[tuple[int, int]] | None:
        owner = value.owner
        if isinstance(owner, arith.AndIOp):
            left, right = collect(owner.lhs), collect(owner.rhs)
            return None if left is None or right is None else [*left, *right]
        if not isinstance(owner, arith.CmpiOp) or owner.predicate.value.data != 0:
            return None
        if owner.lhs not in inputs or not isinstance(owner.rhs.owner, arith.ConstantOp):
            return None
        return [(inputs.index(owner.lhs), _constant_value(owner.rhs.owner, False))]

    selectors = collect(result.operands[0])
    return None if selectors is None else tuple(sorted(selectors))


def export_xdsl_fsm(machine: fsm.MachineOp) -> NormalizedFSM:
    """Export modeled ``fsm.machine`` topology and recovery metadata to JSON data."""
    inputs = tuple(name.data for name in machine.arg_names)
    command_names = tuple(name.data for name in machine.res_names)
    gaps: list[str] = []
    states: list[NormalizedState] = []
    for state in machine.body.block.ops:
        if not isinstance(state, fsm.StateOp):
            continue
        commands: list[tuple[str, bool | int]] = []
        output = state.output.block.last_op
        if not isinstance(output, fsm.OutputOp) or len(output.operands) != len(command_names):
            gaps.append(f"state {state.sym_name.data}: non-constant or incomplete command output")
        else:
            for name, value in zip(command_names, output.operands):
                if not isinstance(value.owner, arith.ConstantOp):
                    gaps.append(f"state {state.sym_name.data}: command {name} is not constant")
                    continue
                commands.append((name, _constant_value(value.owner, name not in {"resolveStage", "rewindDepth"} and not name.startswith("slowPath_"))))
        transitions: list[NormalizedTransition] = []
        for transition in state.transitions.block.ops:
            if not isinstance(transition, fsm.TransitionOp):
                continue
            selectors = _selector_guard(transition, tuple(machine.body.block.args))
            if selectors is None:
                gaps.append(f"state {state.sym_name.data}: transition to {transition.nextState.root_reference.data} has an unsupported guard")
                continue
            transitions.append(NormalizedTransition(transition.nextState.root_reference.data, selectors))
        states.append(NormalizedState(state.sym_name.data, tuple(sorted(commands)), tuple(sorted(transitions, key=lambda item: (item.destination, item.selectors)))))
    scenarios = machine.attributes.get(RECOVERY_SCENARIOS_ATTR, ())
    recovery = tuple(
        {
            "kind": item.kind.data,
            "speculation_ids": [value.data for value in item.speculation_ids],
            "selectors": [value.data for value in item.selectors],
            "release_windows": [value.data for value in item.release_windows],
            "poisoned_speculation_ids": [value.data for value in item.poisoned_speculation_ids],
            "remaining_cycles": item.remaining_cycles.data,
            "destination_speculation_ids": [value.data for value in item.destination_speculation_ids],
            "destination_selectors": [value.data for value in item.destination_selectors],
        }
        for item in scenarios if isinstance(item, SpeculationRecoveryAttr)
    )
    return NormalizedFSM(machine.initialState.data, inputs, tuple(sorted(states, key=lambda item: item.name)), recovery, tuple(gaps))


def _state_map(machine: NormalizedFSM) -> dict[str, NormalizedState]:
    return {state.name: state for state in machine.states}


def _run_trace(machine: NormalizedFSM, inputs: Sequence[Mapping[str, int]], trace_name: str) -> tuple[tuple[tuple[str, tuple[tuple[str, bool | int], ...]], ...], tuple[str, ...]]:
    states = _state_map(machine)
    current = machine.initial_state
    observed = []
    gaps: list[str] = []
    for step, values in enumerate(inputs):
        state = states.get(current)
        if state is None:
            gaps.append(f"{trace_name} step {step}: missing state {current}")
            break
        observed.append((current, state.commands))
        matches = [transition for transition in state.transitions if all(values.get(machine.inputs[index]) == selector for index, selector in transition.selectors)]
        if len(matches) != 1:
            gaps.append(f"{trace_name} step {step}: {len(matches)} matching transitions from {current}; trace is not deterministic")
            break
        current = matches[0].destination
    return tuple(observed), tuple(gaps)


def compare_fsms(java: NormalizedFSM, xdsl: NormalizedFSM, traces: Sequence[Sequence[Mapping[str, int]]] = ()) -> ComparisonResult:
    """Compare structure and supplied bounded traces, reporting unmodeled behavior as gaps."""
    structural = []
    if java.initial_state != xdsl.initial_state:
        structural.append(f"initial state differs: Java {java.initial_state}, xDSL {xdsl.initial_state}")
    if java.inputs != xdsl.inputs:
        structural.append(f"inputs differ: Java {java.inputs}, xDSL {xdsl.inputs}")
    if java.states != xdsl.states:
        structural.append("normalized states, commands, or transitions differ")
    if java.recovery_scenarios != xdsl.recovery_scenarios:
        structural.append("recovery scenarios differ")
    gaps = [*(f"Java: {gap}" for gap in java.unsupported_dimensions), *(f"xDSL: {gap}" for gap in xdsl.unsupported_dimensions)]
    trace_differences = []
    for index, trace in enumerate(traces):
        java_trace, java_gaps = _run_trace(java, trace, f"trace {index} Java")
        xdsl_trace, xdsl_gaps = _run_trace(xdsl, trace, f"trace {index} xDSL")
        gaps.extend((*java_gaps, *xdsl_gaps))
        if not java_gaps and not xdsl_gaps and java_trace != xdsl_trace:
            trace_differences.append(f"trace {index} differs")
    return ComparisonResult(tuple(structural), tuple(trace_differences), tuple(gaps))
