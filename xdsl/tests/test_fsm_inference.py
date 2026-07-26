import pytest
from xdsl.context import Context
from xdsl.dialects import arith, fsm
from xdsl.dialects.builtin import ModuleOp, i1, i64
from xdsl.parser import Parser

from spechls.dialect import (
    CommitOp,
    GammaOp,
    SpecHLS,
    SpeculationConfigAttr,
    SpeculationEntryAttr,
    SpeculationRecoveryAttr,
    SpeculationSlowPathAttr,
    StructType,
    TaskOp,
)
from spechls.transforms import (
    InferSpeculationFSMPass,
    check_transition_table,
    infer_configured_speculation_fsms,
    normalized_transition_table,
    spechls_fsm_body,
)
from spechls.transforms.fsm import RECOVERY_SCENARIOS_ATTR, _RecoveryPlan, _recovery_destination, validate_speculation_config


def config() -> SpeculationConfigAttr:
    return SpeculationConfigAttr(
        [
            SpeculationEntryAttr(1, 3, "g0", 0, [
                SpeculationSlowPathAttr(2, 4, True, True, ["iv"], ["memory"], ["g0"], 4),
                SpeculationSlowPathAttr(7, 2, False, False, [], [], [], 9),
            ]),
            SpeculationEntryAttr(2, 4, "g1", 0, [SpeculationSlowPathAttr(3, 1, True, True, ["iv"], ["memory"], ["g1", "g2"], 5)]),
        ]
    )


def test_speculation_attributes_print_and_parse():
    attribute = config()
    text = str(attribute)
    context = Context()
    context.load_dialect(SpecHLS)

    parsed = Parser(context, text).parse_attribute()

    assert parsed == attribute
    assert str(parsed) == text


def test_inference_emits_one_exactly_guarded_chain_per_slow_path():
    task = TaskOp(StructType("result", ["enable"], [i1]), "worker", [])
    enable = arith.ConstantOp.from_int_and_width(1, 1)
    selector0 = arith.ConstantOp.from_int_and_width(0, 64)
    selector1 = arith.ConstantOp.from_int_and_width(0, 64)
    false = arith.ConstantOp.from_int_and_width(0, 1)
    gamma0 = GammaOp("g0", selector0.result, [enable.result, false.result])
    gamma1 = GammaOp("g1", selector1.result, [enable.result, false.result])
    task.body.block.add_ops([selector0, selector1, false, gamma0, gamma1, enable, CommitOp([enable])])
    task.attributes["spechls.speculation_config"] = config()
    module = ModuleOp([task])
    original_task_ops = tuple(task.body.block.ops)

    machines = infer_configured_speculation_fsms(module)

    assert len(machines) == 1
    machine = machines[0]
    assert machine.sym_name.data == "worker_fsm"
    table = dict(normalized_transition_table(machine))
    assert table["Init_0"] == ("Init_1",)
    assert {"Proceed", "Rollback_0_1", "Rollback_1_0", "Stall_0_0_0"}.issubset(table["Proceed"])
    assert any(name.startswith("Combined_") for name in table)
    assert any(name.startswith("NewMispec_") for name in table)
    assert tuple(name.data for name in machine.arg_names) == ("mispec_0", "mispec_1")
    assert tuple(name.data for name in machine.res_names) == (
        "nextInput", "commit", "rewind", "rbwe", "resolveStage", "rewindDepth", "muRollback_iv", "arrayRollback_memory", "gammaRollback_g0", "gammaRollback_g1", "gammaRollback_g2", "slowPath_g0", "slowPath_g1",
    )
    proceed = next(op for op in machine.body.block.ops if isinstance(op, fsm.StateOp) and op.sym_name.data == "Proceed")
    rollback = next(op for op in machine.body.block.ops if isinstance(op, fsm.StateOp) and op.sym_name.data == "Rollback_0_0")
    assert isinstance(proceed.output.block.last_op, fsm.OutputOp)
    assert isinstance(rollback.output.block.last_op, fsm.OutputOp)
    assert all(transition.guard.block.last_op is not None for transition in proceed.transitions.block.ops if isinstance(transition, fsm.TransitionOp) and transition.nextState.root_reference.data != "Proceed")
    slow_transition = next(
        transition
        for transition in proceed.transitions.block.ops
        if isinstance(transition, fsm.TransitionOp)
        and transition.nextState.root_reference.data == "Stall_0_0_0"
    )
    comparison = slow_transition.guard.block.last_op.prev_op
    assert isinstance(comparison, arith.CmpiOp)
    assert comparison.lhs is machine.body.block.args[0]
    assert comparison.predicate.value.data == 0  # ``mispec_0 == 2``.
    assert comparison.rhs.owner.value.value.data == 2
    guarded_paths = {
        transition.nextState.root_reference.data: transition.guard.block.last_op.prev_op
        for transition in proceed.transitions.block.ops
        if isinstance(transition, fsm.TransitionOp)
        and transition.nextState.root_reference.data in {"Stall_0_0_0", "Rollback_0_1", "Rollback_1_0"}
    }
    assert {
        destination: (comparison.lhs, comparison.rhs.owner.value.value.data, comparison.predicate.value.data)
        for destination, comparison in guarded_paths.items()
    } == {
        "Stall_0_0_0": (machine.body.block.args[0], 2, 0),
        "Rollback_0_1": (machine.body.block.args[0], 7, 0),
        "Rollback_1_0": (machine.body.block.args[1], 3, 0),
    }
    assert tuple(rollback.output.block.ops)[2].value.value.data == -1  # i1 true: per-path rewind
    assert tuple(rollback.output.block.ops)[6].value.value.data == -1
    assert tuple(rollback.output.block.ops)[7].value.value.data == -1
    assert tuple(rollback.output.block.ops)[8].value.value.data == -1
    assert tuple(rollback.output.block.ops)[11].value.value.data == 2
    no_rewind = next(op for op in machine.body.block.ops if isinstance(op, fsm.StateOp) and op.sym_name.data == "Rollback_0_1")
    assert tuple(no_rewind.output.block.ops)[2].value.value.data == 0
    assert tuple(no_rewind.output.block.ops)[3].value.value.data == 0
    assert tuple(no_rewind.output.block.ops)[11].value.value.data == 7
    fill = next(op for op in machine.body.block.ops if isinstance(op, fsm.StateOp) and op.sym_name.data == "Fill_1_0_0")
    assert tuple(fill.output.block.ops)[12].value.value.data == 3
    assert tuple(fill.output.block.ops)[4].value.value.data == 4
    check_transition_table(machine, dict(normalized_transition_table(machine)))
    with pytest.raises(ValueError, match="transition table mismatch"):
        check_transition_table(machine, {"Proceed": ["Stall_0"]})
    assert infer_configured_speculation_fsms(module) == []
    assert tuple(task.body.block.ops) == original_task_ops
    assert gamma0.select is selector0.result
    assert gamma1.select is selector1.result
    body = spechls_fsm_body(machine)
    assert {op.state_name.data for op in body.block.ops if op.name == "spechls.fsm_state"} >= {"Init_0", "Proceed", "Rollback_0_0"}
    guarded = [op for op in body.block.ops if op.name == "spechls.fsm_transition" and op.source.data == "Proceed"]
    assert any(tuple(op.input_ids.get_values()) == (0,) and tuple(op.selectors.get_values()) == (2,) for op in guarded)
    assert all(op.kind.data == "normal" for op in guarded)
    module.verify()


def test_pass_accepts_configuration_on_non_task_source_operation():
    source = arith.ConstantOp.from_int_and_width(0, 32)
    source.attributes["spechls.speculation_config"] = SpeculationConfigAttr([])
    module = ModuleOp([source])

    InferSpeculationFSMPass().apply(Context(), module)

    machine = next(operation for operation in module.body.block.ops if isinstance(operation, fsm.MachineOp))
    assert machine.sym_name.data == "arith_constant_fsm"
    assert normalized_transition_table(machine) == (("Init_0", ("Proceed",)), ("Proceed", ("Proceed",)))
    module.verify()


def test_description_generation_does_not_require_a_task_gamma():
    task = TaskOp(StructType("result", ["enable"], [i1]), "unwired", [])
    enable = arith.ConstantOp.from_int_and_width(1, 1)
    task.body.block.add_ops([enable, CommitOp([enable])])
    task.attributes["spechls.speculation_config"] = SpeculationConfigAttr([
        SpeculationEntryAttr(1, 0, "missing", 0, [SpeculationSlowPathAttr(1, 1, False, False, [], [], [], 0)])
    ])
    module = ModuleOp([task])
    original_task_ops = tuple(task.body.block.ops)

    machine = infer_configured_speculation_fsms(module)[0]

    assert machine.sym_name.data == "unwired_fsm"
    assert tuple(task.body.block.ops) == original_task_ops
    module.verify()


def test_multi_mispec_metadata_models_java_recovery_classes_and_parser():
    # Speculation zero poisons one, so it is ordered first despite its larger
    # condition latency. The release windows use Java's RELEASE_DELAY formula.
    configuration = SpeculationConfigAttr([
        SpeculationEntryAttr(3, 0, "g0", 0, [SpeculationSlowPathAttr(4, 7, True, True, [], [], [], 2)], [1]),
        SpeculationEntryAttr(1, 0, "g1", 0, [SpeculationSlowPathAttr(9, 2, True, True, [], [], [], 1)]),
    ])
    task = TaskOp(StructType("result", ["enable"], [i1]), "multi", [])
    enable = arith.ConstantOp.from_int_and_width(1, 1)
    selector0 = arith.ConstantOp.from_int_and_width(0, 64)
    selector1 = arith.ConstantOp.from_int_and_width(0, 64)
    false = arith.ConstantOp.from_int_and_width(0, 1)
    task.body.block.add_ops([selector0, selector1, false, GammaOp("g0", selector0.result, [enable.result, false.result]), GammaOp("g1", selector1.result, [enable.result, false.result]), enable, CommitOp([enable])])
    task.attributes["spechls.speculation_config"] = configuration
    machine = infer_configured_speculation_fsms(ModuleOp([task]))[0]

    scenarios = tuple(machine.attributes[RECOVERY_SCENARIOS_ATTR])
    combined = next(item for item in scenarios if isinstance(item, SpeculationRecoveryAttr) and item.kind.data == "combined")
    new_mispec = next(item for item in scenarios if isinstance(item, SpeculationRecoveryAttr) and item.kind.data == "new_mispec")
    canceled = next(item for item in scenarios if isinstance(item, SpeculationRecoveryAttr) and item.kind.data == "canceled")
    assert tuple(item.data for item in combined.speculation_ids) == (0, 1)
    assert tuple(item.data for item in combined.selectors) == (4, 9)
    assert tuple(item.data for item in combined.release_windows) == (6, 1)
    assert tuple(item.data for item in combined.poisoned_speculation_ids) == (1,)
    assert tuple(item.data for item in new_mispec.destination_speculation_ids)
    assert new_mispec.remaining_cycles.data >= 0
    assert tuple(item.data for item in canceled.destination_speculation_ids) == (0, 1)
    assert tuple(item.data for item in canceled.destination_selectors) == (4, 9)
    context = Context()
    context.load_dialect(SpecHLS)
    assert Parser(context, str(combined)).parse_attribute() == combined
    assert Parser(context, str(new_mispec)).parse_attribute() == new_mispec
    assert Parser(context, str(canceled)).parse_attribute() == canceled
    assert any(name.startswith("Combined_") for name, _ in normalized_transition_table(machine))
    assert any(name.startswith("NewMispec_") for name, _ in normalized_transition_table(machine))


def test_missing_java_canceled_destination_is_rejected():
    plan = _RecoveryPlan("canceled", (1,), (9,), (1,), destination_speculation_ids=(0, 1), destination_selectors=(4, 9))

    with pytest.raises(ValueError, match="unsupported CANCELED destination fallback"):
        _recovery_destination({}, plan, "CANCELED")


@pytest.mark.parametrize("configuration, message", [
    (SpeculationConfigAttr([SpeculationEntryAttr(0, 0, "g0", 0, [])]), "cond_latency"),
    (SpeculationConfigAttr([SpeculationEntryAttr(1, 0, "g0", 0, [], [0])]), "other configured"),
    (SpeculationConfigAttr([SpeculationEntryAttr(1, 0, "g0", 0, [], [1]), SpeculationEntryAttr(1, 0, "g1", 0, [], [0])]), "acyclic"),
    (SpeculationConfigAttr([SpeculationEntryAttr(1, 0, "g0", 0, [SpeculationSlowPathAttr(1, 1, False, False, [], [], [], 0), SpeculationSlowPathAttr(1, 1, False, False, [], [], [], 0)])]), "unique"),
    (SpeculationConfigAttr([SpeculationEntryAttr(1, 0, "g0", 0, [SpeculationSlowPathAttr(1, 1, False, False, ["iv", "iv"], [], [], 0)])]), "Mu IDs"),
])
def test_invalid_multi_mispec_configuration_is_rejected(configuration, message):
    with pytest.raises(ValueError, match=message):
        validate_speculation_config(configuration)
