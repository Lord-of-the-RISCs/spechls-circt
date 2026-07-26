"""Configuration-driven construction of conservative FSM topology."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product

from xdsl.dialects import arith, builtin, fsm
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, i1, i64
from xdsl.ir import Block, Operation, Region, SSAValue

from spechls.dialect import (
    SpeculationConfigAttr,
    SpeculationRecoveryAttr,
    FSMStateOp,
    FSMTransitionOp,
    TaskOp,
)


SPECULATION_CONFIG_ATTR = "spechls.speculation_config"
INFERRED_FSM_ATTR = "spechls.inferred_fsm"
RECOVERY_SCENARIOS_ATTR = "spechls.recovery_scenarios"


def normalized_transition_table(machine: fsm.MachineOp) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the machine topology in a stable, Java-comparison-friendly form."""
    states: list[tuple[str, tuple[str, ...]]] = []
    for operation in machine.body.block.ops:
        if not isinstance(operation, fsm.StateOp):
            continue
        destinations = sorted(
            transition.nextState.root_reference.data
            for transition in operation.transitions.block.ops
            if isinstance(transition, fsm.TransitionOp)
        )
        states.append((operation.sym_name.data, tuple(destinations)))
    return tuple(sorted(states))


def check_transition_table(
    machine: fsm.MachineOp,
    expected: Mapping[str, Sequence[str]] | Sequence[tuple[str, Sequence[str]]],
) -> None:
    """Raise ``ValueError`` when a normalized table differs from ``expected``."""
    items = expected.items() if isinstance(expected, Mapping) else expected
    expected_table = tuple(sorted((state, tuple(sorted(destinations))) for state, destinations in items))
    actual_table = normalized_transition_table(machine)
    if actual_table != expected_table:
        raise ValueError(f"FSM transition table mismatch: expected {expected_table}, got {actual_table}")


def spechls_fsm_body(machine: fsm.MachineOp) -> Region:
    """Serialize an xDSL FSM machine into the explicit ``spechls.fsm`` body.

    ``spechls.fsm`` keeps its established packed runtime interface. Its body
    carries this lossless constant-command/selector-transition description so
    MLIR and xDSL can inspect the same controller table.
    """
    output_names = tuple(name.data for name in machine.res_names)
    body = Block()
    for state in machine.body.block.ops:
        if not isinstance(state, fsm.StateOp):
            continue
        output = state.output.block.last_op
        if not isinstance(output, fsm.OutputOp):
            raise ValueError(f"state {state.sym_name.data} has no command output")
        commands: list[int] = []
        for value in output.operands:
            if not isinstance(value.owner, arith.ConstantOp):
                raise ValueError(f"state {state.sym_name.data} has a non-constant command output")
            commands.append(value.owner.value.value.data)
        if len(commands) != len(output_names):
            raise ValueError(f"state {state.sym_name.data} command count does not match the FSM interface")
        body.add_op(FSMStateOp.create(attributes={
            "name": StringAttr(state.sym_name.data),
            "commands": builtin.DenseArrayBase.from_list(i64, commands),
        }))
        for transition in state.transitions.block.ops:
            if not isinstance(transition, fsm.TransitionOp):
                continue
            guards: list[tuple[int, int]] = []
            if transition.guard.blocks:
                result = transition.guard.block.last_op
                if not isinstance(result, fsm.ReturnOp) or not result.operands:
                    raise ValueError(f"transition from {state.sym_name.data} has an unsupported guard")

                def collect(value: SSAValue) -> None:
                    owner = value.owner
                    if isinstance(owner, arith.AndIOp):
                        collect(owner.lhs)
                        collect(owner.rhs)
                        return
                    if (not isinstance(owner, arith.CmpiOp) or owner.predicate.value.data != 0 or
                            owner.lhs not in machine.body.block.args or
                            not isinstance(owner.rhs.owner, arith.ConstantOp)):
                        raise ValueError(f"transition from {state.sym_name.data} has an unsupported guard")
                    guards.append((machine.body.block.args.index(owner.lhs), owner.rhs.owner.value.value.data))

                collect(result.operands[0])
            body.add_op(FSMTransitionOp.create(attributes={
                "source": StringAttr(state.sym_name.data),
                "target": StringAttr(transition.nextState.root_reference.data),
                "kind": StringAttr("new_mispec" if transition.nextState.root_reference.data.startswith("NewMispec_") else "normal"),
                "input_ids": builtin.DenseArrayBase.from_list(i64, [item[0] for item in guards]),
                "selectors": builtin.DenseArrayBase.from_list(i64, [item[1] for item in guards]),
            }))
    return Region(body)


def _add_transition(
    state: fsm.StateOp,
    destination: str,
    *,
    mispec: SSAValue | None = None,
    slow_path_selector: int | None = None,
) -> None:
    # Empty regions are required for unconditional transitions by xDSL's FSM verifier.
    guard = Region()
    if mispec is not None:
        assert slow_path_selector is not None
        selector = arith.ConstantOp.from_int_and_width(slow_path_selector, 64)
        is_selected = arith.CmpiOp(mispec, selector, "eq")
        guard = Region(Block([selector, is_selected, fsm.ReturnOp(is_selected)]))
    state.transitions.block.add_op(fsm.TransitionOp(destination, guard, Region()))


def _add_exact_transition(state: fsm.StateOp, destination: str, selectors: Sequence[tuple[SSAValue, int]]) -> None:
    """Add a conjunction of literal Java selector predicates, never a wildcard."""
    operations: list[Operation] = []
    comparison: SSAValue | None = None
    for value, selector_value in selectors:
        selector = arith.ConstantOp.from_int_and_width(selector_value, 64)
        selected = arith.CmpiOp(value, selector, "eq")
        operations.extend((selector, selected))
        if comparison is None:
            comparison = selected
        else:
            combined = arith.AndIOp(comparison, selected)
            operations.append(combined)
            comparison = combined
    assert comparison is not None
    state.transitions.block.add_op(fsm.TransitionOp(destination, Region(Block([*operations, fsm.ReturnOp(comparison)])), Region()))


_INPUT_NAMES = ("mispec",)
_FIXED_OUTPUT_NAMES = ("nextInput", "commit", "rewind", "rbwe", "resolveStage", "rewindDepth")


def _target_ids(config: SpeculationConfigAttr, field: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        target.data for entry in config.entries for path in entry.slow_paths for target in getattr(path, field)
    ))


def _command_output_names(config: SpeculationConfigAttr) -> tuple[str, ...]:
    gamma_ids = tuple(entry.gamma_id.data for entry in config.entries)
    return (
        *_FIXED_OUTPUT_NAMES,
        *(f"muRollback_{target}" for target in _target_ids(config, "rollback_mu_ids")),
        *(f"arrayRollback_{target}" for target in _target_ids(config, "rollback_array_ids")),
        *(f"gammaRollback_{target}" for target in _target_ids(config, "rollback_gamma_ids")),
        *(f"slowPath_{gamma_id}" for gamma_id in gamma_ids),
    )


def _command_output_types(names: Sequence[str]) -> list:
    return [i64 if name.startswith(("slowPath_", "resolveStage", "rewindDepth")) else i1 for name in names]


def _state_output(
    output_names: Sequence[str],
    *,
    next_input: bool = False,
    commit: bool = False,
    rewind: bool = False,
    rbwe: bool = False,
    rollback_mu_ids: Sequence[str] = (),
    rollback_array_ids: Sequence[str] = (),
    rollback_gamma_ids: Sequence[str] = (),
    slow_paths: Mapping[str, int] | None = None,
    resolve_stage: int = -1,
    rewind_depth: int = 0,
) -> Region:
    """Build controller commands asserted while a state is active."""
    active = set(rollback_mu_ids) | set(rollback_array_ids) | set(rollback_gamma_ids)
    values = []
    for name in output_names:
        if name == "nextInput": value = int(next_input)
        elif name == "commit": value = int(commit)
        elif name == "rewind": value = int(rewind)
        elif name == "rbwe": value = int(rbwe)
        elif name == "resolveStage": value = resolve_stage
        elif name == "rewindDepth": value = rewind_depth
        elif name.startswith("slowPath_"): value = (slow_paths or {}).get(name.removeprefix("slowPath_"), -1)
        else: value = int(name.removeprefix("muRollback_").removeprefix("arrayRollback_").removeprefix("gammaRollback_") in active)
        values.append(value)
    commands = [arith.ConstantOp.from_int_and_width(value, 64 if name.startswith(("slowPath_", "resolveStage", "rewindDepth")) else 1) for name, value in zip(output_names, values)]
    return Region(Block([*commands, fsm.OutputOp(commands)]))


def validate_speculation_config(config: SpeculationConfigAttr) -> None:
    """Validate the timing and poison relation required by Java-style recovery."""
    entries = tuple(config.entries)
    dependencies: list[tuple[int, ...]] = []
    gamma_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if entry.cond_latency.data < 1:
            raise ValueError(f"speculation {index}: cond_latency must be at least one")
        if entry.resolve_stage.data < 0:
            raise ValueError(f"speculation {index}: resolve_stage must be non-negative")
        if not entry.gamma_id.data:
            raise ValueError(f"speculation {index}: gamma_id must not be empty")
        if entry.fast_selector.data < 0:
            raise ValueError(f"speculation {index}: fast_selector must be non-negative")
        if entry.gamma_id.data in gamma_ids:
            raise ValueError("speculation gamma IDs must be unique")
        gamma_ids.add(entry.gamma_id.data)
        selectors: set[int] = set()
        for path in entry.slow_paths:
            if path.selector.data < 0 or path.latency.data < 1 or path.rollback_depth.data < 0:
                raise ValueError(f"speculation {index}: slow-path selector, latency, and rollback depth must be non-negative (latency non-zero)")
            if path.selector.data in selectors:
                raise ValueError(f"speculation {index}: slow-path selectors must be unique")
            if path.selector.data == entry.fast_selector.data:
                raise ValueError(f"speculation {index}: slow-path selector must differ from fast_selector")
            selectors.add(path.selector.data)
            if path.rewind.data not in (0, 1) or path.rbwe.data not in (0, 1):
                raise ValueError(f"speculation {index}: rewind and rbwe must be boolean flags")
            for kind, targets in (("Mu", path.rollback_mu_ids), ("array", path.rollback_array_ids), ("gamma", path.rollback_gamma_ids)):
                target_ids = [target.data for target in targets]
                if not all(target_ids) or len(target_ids) != len(set(target_ids)):
                    raise ValueError(f"speculation {index}: rollback {kind} IDs must be non-empty and unique")
        dependency = tuple(item.data for item in entry.poison_speculation_ids)
        if len(dependency) != len(set(dependency)):
            raise ValueError(f"speculation {index}: poison dependencies must be unique")
        if any(item < 0 or item >= len(entries) or item == index for item in dependency):
            raise ValueError(f"speculation {index}: poison dependencies must name other configured speculations")
        dependencies.append(dependency)

    # Java obtains this relation from an acyclic transitive closure. A cycle has
    # no dependency-aware recovery order, so reject rather than use index order.
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(speculation_id: int) -> None:
        if speculation_id in visiting:
            raise ValueError("poison dependencies must be acyclic")
        if speculation_id not in visited:
            visiting.add(speculation_id)
            for dependency in dependencies[speculation_id]:
                visit(dependency)
            visiting.remove(speculation_id)
            visited.add(speculation_id)

    for index in range(len(entries)):
        visit(index)


def _release_window(entry, slow_path) -> int:
    # FSMUtils.RELEASE_DELAY: max(stallDelay + 1, 0) + fillDelay + 1.
    return max(slow_path.latency.data - entry.cond_latency.data - 1, 0) + entry.cond_latency.data


@dataclass(frozen=True)
class _RecoveryPlan:
    kind: str
    speculation_ids: tuple[int, ...]
    selectors: tuple[int, ...]
    release_windows: tuple[int, ...]
    poisoned_speculation_ids: tuple[int, ...] = ()
    remaining_cycles: int = -1
    destination_speculation_ids: tuple[int, ...] = ()
    destination_selectors: tuple[int, ...] = ()


def _recovery_destination(plan_starts: Mapping[tuple[tuple[int, int], ...], str], plan: _RecoveryPlan, kind: str) -> str:
    destination = plan_starts.get(tuple(zip(plan.destination_speculation_ids, plan.destination_selectors)))
    if destination is None:
        if kind == "CANCELED":
            raise ValueError("unsupported CANCELED destination fallback: Java requires an already-created canonical recovery path")
        raise ValueError(f"unsupported {kind} destination: no canonical recovery path")
    return destination


def _recovery_plans(config: SpeculationConfigAttr) -> tuple[_RecoveryPlan, ...]:
    """Materialize the Java worklist's observable recovery alternatives.

    The FSM input has no iteration tag.  Recording these scenarios explicitly
    preserves the temporal classification for a later stateful lowering rather
    than incorrectly treating every observed selector as an independent event.
    """
    entries = tuple(config.entries)

    def greater(left: int, right: int) -> bool:
        # Matches FSMConstructor.isGreaterForCombined exactly.
        left_poison = {value.data for value in entries[left].poison_speculation_ids}
        right_poison = {value.data for value in entries[right].poison_speculation_ids}
        if right in left_poison:
            return True
        if left in right_poison:
            return False
        if entries[left].cond_latency.data != entries[right].cond_latency.data:
            return entries[left].cond_latency.data < entries[right].cond_latency.data
        return left < right

    alternatives = [tuple(entry.slow_paths) for entry in entries]
    plans: list[_RecoveryPlan] = []
    canonical: dict[tuple[tuple[int, int], ...], _RecoveryPlan] = {}
    combinations_count = 0
    for size in range(1, len(entries) + 1):
        for selected_ids in combinations(range(len(entries)), size):
            if any(not alternatives[index] for index in selected_ids):
                continue
            for selected_paths in product(*(alternatives[index] for index in selected_ids)):
                combinations_count += 1
                if combinations_count > 256:
                    raise ValueError("unsupported speculation configuration: more than 256 recovery scenarios")
                ordered = sorted(zip(selected_ids, selected_paths), key=lambda item: item[0])
                # Python's comparator-free sort retains Java's deterministic pairwise order.
                for position in range(1, len(ordered)):
                    cursor = position
                    while cursor and greater(ordered[cursor][0], ordered[cursor - 1][0]):
                        ordered[cursor - 1], ordered[cursor] = ordered[cursor], ordered[cursor - 1]
                        cursor -= 1
                ids = [item[0] for item in ordered]
                selectors = [item[1].selector.data for item in ordered]
                windows = [_release_window(entries[index], path) for index, path in ordered]
                poisoned = tuple(sorted({poisoned for index in ids for poisoned in (item.data for item in entries[index].poison_speculation_ids)}))
                plan = _RecoveryPlan("initial" if size == 1 else "combined", tuple(ids), tuple(selectors), tuple(windows), poisoned)
                canonical[tuple(zip(ids, selectors))] = plan
                plans.append(plan)

    # These are explicit worklist edges.  They preserve the Java distinctions
    # between a future-iteration recovery and a same-iteration cancellation.
    # The stateful boundary consumes the iteration relation; FSM guards remain
    # exact selectors and must not manufacture an iteration match.
    for active in tuple(canonical.values()):
        active_pairs = tuple(zip(active.speculation_ids, active.selectors))
        for index, entry in enumerate(entries):
            if index in active.speculation_ids:
                continue
            for path in entry.slow_paths:
                remaining = _release_window(entry, path)
                destination_pairs = [*active_pairs, (index, path.selector.data)]
                for position in range(1, len(destination_pairs)):
                    cursor = position
                    while cursor and greater(destination_pairs[cursor][0], destination_pairs[cursor - 1][0]):
                        destination_pairs[cursor - 1], destination_pairs[cursor] = destination_pairs[cursor], destination_pairs[cursor - 1]
                        cursor -= 1
                plans.append(_RecoveryPlan("new_mispec", (index,), (path.selector.data,), (remaining,), tuple(sorted(set(active.poisoned_speculation_ids) | set(item.data for item in entry.poison_speculation_ids))), remaining, tuple(pair[0] for pair in destination_pairs), tuple(pair[1] for pair in destination_pairs)))
                higher = tuple((speculation_id, selector) for speculation_id, selector in active_pairs if greater(speculation_id, index))
                canceled_destination = (*higher, (index, path.selector.data))
                if higher:
                    destination_plan = canonical.get(canceled_destination)
                    if destination_plan is None:
                        raise ValueError("unsupported CANCELED destination fallback: Java requires an already-created canonical recovery path")
                    plans.append(_RecoveryPlan("canceled", (index,), (path.selector.data,), (remaining,), tuple(sorted(set(destination_plan.poisoned_speculation_ids) | set(item.data for item in entry.poison_speculation_ids))), remaining, destination_plan.speculation_ids, destination_plan.selectors))
    if len(plans) > 256:
        raise ValueError("unsupported speculation configuration: more than 256 recovery scenarios")
    return tuple(plans)


def _recovery_scenarios(config: SpeculationConfigAttr) -> ArrayAttr[SpeculationRecoveryAttr]:
    return ArrayAttr(
        SpeculationRecoveryAttr(plan.kind, plan.speculation_ids, plan.selectors, plan.release_windows, plan.poisoned_speculation_ids, plan.remaining_cycles, plan.destination_speculation_ids, plan.destination_selectors)
        for plan in _recovery_plans(config)
    )


def infer_speculation_fsm(
    source: Operation,
    config: SpeculationConfigAttr,
    module: builtin.ModuleOp,
    name: str,
) -> fsm.MachineOp:
    """Build timing-expanded recovery chains for every configured slow path."""
    validate_speculation_config(config)
    entries = tuple(config.entries)
    init_count = max([1, *(entry.cond_latency.data for entry in entries)])
    input_names = ArrayAttr(StringAttr(f"{_INPUT_NAMES[0]}_{index}") for index in range(len(entries)))
    command_names = _command_output_names(config)
    output_names = ArrayAttr(StringAttr(output) for output in command_names)
    machine = fsm.MachineOp(
        name,
        "Init_0",
        ([i64] * len(entries), _command_output_types(command_names)),
        ArrayAttr(DictionaryAttr({}) for _ in input_names),
        ArrayAttr(DictionaryAttr({}) for _ in output_names),
        input_names,
        output_names,
    )
    # xDSL's FSM constructor stores these optional attributes under the Python
    # field names, while the dialect declares their CIRCT spellings.
    machine.attributes["argNames"] = input_names
    machine.attributes["resNames"] = output_names
    machine.attributes[RECOVERY_SCENARIOS_ATTR] = _recovery_scenarios(config)
    recovery_plans = _recovery_plans(config)
    states: dict[str, fsm.StateOp] = {}

    def state(state_name: str, **commands: bool | int) -> fsm.StateOp:
        result = fsm.StateOp(state_name, _state_output(command_names, **commands))
        machine.body.block.add_op(result)
        states[state_name] = result
        return result

    for index in range(init_count):
        state(f"Init_{index}")
    state("Proceed", next_input=True, commit=True)
    for index, entry in enumerate(entries):
        for path_index, slow_path in enumerate(entry.slow_paths):
            stall_count = max(slow_path.latency.data - entry.cond_latency.data - 1, 0)
            fill_count = max(entry.cond_latency.data - 1, 0)
            prefix = f"{index}_{path_index}"
            for phase in range(stall_count):
                state(f"Stall_{prefix}_{phase}", slow_paths={entry.gamma_id.data: slow_path.selector.data}, resolve_stage=entry.resolve_stage.data)
            state(
                f"Rollback_{prefix}",
                rewind=bool(slow_path.rewind.data),
                rbwe=bool(slow_path.rbwe.data),
                rollback_mu_ids=[target.data for target in slow_path.rollback_mu_ids],
                rollback_array_ids=[target.data for target in slow_path.rollback_array_ids],
                rollback_gamma_ids=[target.data for target in slow_path.rollback_gamma_ids],
                slow_paths={entry.gamma_id.data: slow_path.selector.data},
                resolve_stage=entry.resolve_stage.data,
                rewind_depth=slow_path.rollback_depth.data,
            )
            for phase in range(fill_count):
                state(f"Fill_{prefix}_{phase}", slow_paths={entry.gamma_id.data: slow_path.selector.data}, resolve_stage=entry.resolve_stage.data)

    # A combined path has its own states.  Its last dependency-ordered member
    # performs the physical rollback; the complete per-member command set is
    # retained by the typed recovery boundary for stateful lowering.
    canonical_plans = [plan for plan in recovery_plans if plan.kind in ("initial", "combined")]
    plan_starts: dict[tuple[tuple[int, int], ...], str] = {}
    plan_states: dict[tuple[tuple[int, int], ...], tuple[str, ...]] = {}
    for plan_index, plan in enumerate(canonical_plans):
        if plan.kind == "initial":
            continue
        index, selector = plan.speculation_ids[-1], plan.selectors[-1]
        entry = entries[index]
        path = next(path for path in entry.slow_paths if path.selector.data == selector)
        stall_count = max(path.latency.data - entry.cond_latency.data - 1, 0)
        fill_count = max(entry.cond_latency.data - 1, 0)
        prefix = f"Combined_{plan_index}"
        names = [*(f"{prefix}_Stall_{phase}" for phase in range(stall_count)), f"{prefix}_Rollback", *(f"{prefix}_Fill_{phase}" for phase in range(fill_count))]
        for name in names:
            if "Rollback" in name:
                state(name, rewind=bool(path.rewind.data), rbwe=bool(path.rbwe.data), rollback_mu_ids=[target.data for target in path.rollback_mu_ids], rollback_array_ids=[target.data for target in path.rollback_array_ids], rollback_gamma_ids=[target.data for target in path.rollback_gamma_ids], slow_paths={entry.gamma_id.data: selector}, resolve_stage=entry.resolve_stage.data, rewind_depth=path.rollback_depth.data)
            else:
                state(name, slow_paths={entry.gamma_id.data: selector}, resolve_stage=entry.resolve_stage.data)
        key = tuple(zip(plan.speculation_ids, plan.selectors))
        plan_starts[key] = names[0]
        plan_states[key] = tuple(names)
    new_starts: dict[int, str] = {}
    for plan_index, plan in enumerate(recovery_plans):
        if plan.kind != "new_mispec":
            continue
        name = f"NewMispec_{plan_index}"
        entry = entries[plan.speculation_ids[0]]
        state(name, slow_paths={entry.gamma_id.data: plan.selectors[0]}, resolve_stage=entry.resolve_stage.data)
        new_starts[plan_index] = name

    for index in range(init_count):
        _add_transition(states[f"Init_{index}"], f"Init_{index + 1}" if index + 1 < init_count else "Proceed")
    for index, entry in enumerate(entries):
        for path_index, slow_path in enumerate(entry.slow_paths):
            stall_count = max(slow_path.latency.data - entry.cond_latency.data - 1, 0)
            fill_count = max(entry.cond_latency.data - 1, 0)
            prefix = f"{index}_{path_index}"
            first_stall = f"Stall_{prefix}_0" if stall_count else f"Rollback_{prefix}"
            _add_transition(states["Proceed"], first_stall, mispec=machine.body.block.args[index], slow_path_selector=slow_path.selector.data)
            for phase in range(stall_count):
                _add_transition(states[f"Stall_{prefix}_{phase}"], f"Stall_{prefix}_{phase + 1}" if phase + 1 < stall_count else f"Rollback_{prefix}")
            first_fill = f"Fill_{prefix}_0" if fill_count else "Proceed"
            _add_transition(states[f"Rollback_{prefix}"], first_fill)
            for phase in range(fill_count):
                _add_transition(states[f"Fill_{prefix}_{phase}"], f"Fill_{prefix}_{phase + 1}" if phase + 1 < fill_count else "Proceed")

    for plan in canonical_plans:
        if plan.kind == "initial":
            continue
        key = tuple(zip(plan.speculation_ids, plan.selectors))
        names = plan_states[key]
        _add_exact_transition(states["Proceed"], names[0], [(machine.body.block.args[index], selector) for index, selector in key])
        for state_name, destination in zip(names, (*names[1:], "Proceed")):
            _add_transition(states[state_name], destination)

    recovery_state_names = [name for name in states if name.startswith(("Stall_", "Rollback_", "Fill_", "Combined_"))]
    for plan_index, plan in enumerate(recovery_plans):
        if plan.kind == "new_mispec":
            destination = _recovery_destination(plan_starts, plan, "NEW_MISPEC")
            _add_transition(states[new_starts[plan_index]], destination)
            for state_name in recovery_state_names:
                _add_transition(states[state_name], new_starts[plan_index], mispec=machine.body.block.args[plan.speculation_ids[0]], slow_path_selector=plan.selectors[0])
        elif plan.kind == "canceled":
            destination = _recovery_destination(plan_starts, plan, "CANCELED")
            for state_name in recovery_state_names:
                _add_transition(states[state_name], destination, mispec=machine.body.block.args[plan.speculation_ids[0]], slow_path_selector=plan.selectors[0])
    _add_transition(states["Proceed"], "Proceed")

    module.body.block.add_op(machine)
    source.attributes[INFERRED_FSM_ATTR] = StringAttr(name)
    return machine


def _machine_name(source: Operation, used_names: set[str]) -> str:
    if isinstance(source, TaskOp):
        base = source.sym_name.data
    else:
        base = source.name.replace(".", "_")
    base = f"{base}_fsm"
    name = base
    suffix = 0
    while name in used_names:
        suffix += 1
        name = f"{base}_{suffix}"
    used_names.add(name)
    return name


def infer_configured_speculation_fsms(module: builtin.ModuleOp) -> list[fsm.MachineOp]:
    """Infer one machine for each configured task or source operation in ``module``."""
    used_names = {
        operation.sym_name.data
        for operation in module.body.block.ops
        if isinstance(operation, fsm.MachineOp)
    }
    machines: list[fsm.MachineOp] = []
    for source in list(module.walk()):
        config = source.attributes.get(SPECULATION_CONFIG_ATTR)
        if not isinstance(config, SpeculationConfigAttr):
            continue
        if INFERRED_FSM_ATTR in source.attributes:
            continue
        machines.append(infer_speculation_fsm(source, config, module, _machine_name(source, used_names)))
    return machines
