import shutil
import subprocess
from pathlib import Path

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, FlatSymbolRefAttr, FunctionType, ModuleOp, StringAttr, i1, i32, i64
from xdsl.ir import Block, Region
from xdsl.utils.test_value import create_ssa_value

from spechls.dialect import (
    CallOp,
    CommitOp,
    FSMMachineOp,
    FSMControllerStateOp,
    FSMInstanceOp,
    FSMOutputOp,
    FSMTriggerOp,
    GammaOp,
    KernelOp,
    MuOp,
    StructType,
    TaskOp,
)
from spechls.native_mlir import parse_native_mlir
from spechls.uclid import UclidEmissionError, emit_stuttering_bisimulation_driver, emit_uclid


def call(name, argument, result_type):
    return CallOp.create(
        operands=[argument],
        result_types=[result_type],
        attributes={"callee": FlatSymbolRefAttr(name)},
    )


def slowfast_module():
    kernel = KernelOp("slowfast", ([i32], [i32]))
    initial = create_ssa_value(i32)
    task = TaskOp(StructType("commit_type", ["enable", "value"], [i1, i32]), "slowfast", [initial])
    block = task.body.block
    mu = MuOp.create(
        operands=[block.args[0], block.args[0]],
        result_types=[i32],
        attributes={"sym_name": StringAttr("x")},
    )
    cond = call("cond", mu.result, i1)
    fast = call("fast", mu.result, i32)
    slow = call("slow", mu.result, i32)
    gamma = GammaOp("x", cond.result[0], [fast.result[0], slow.result[0]])
    mu.operands[1] = gamma.result
    enabled = arith.ConstantOp.from_int_and_width(1, 1)
    block.add_ops([mu, cond, fast, slow, gamma, enabled, CommitOp([enabled, gamma])])
    kernel.body.block.add_ops([task])
    return ModuleOp([kernel])


UCLID_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "test/SpecHLS/Transforms/output/slowfast-end-to-end.ucl"
)
TRANSFORMED_UCLID_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "test/SpecHLS/Transforms/output/slowfast-end-to-end.transformed.ucl"
)


def test_emit_uclid_source_slowfast_uses_the_ir_driven_generator_and_checks_it(tmp_path):
    model = emit_uclid(slowfast_module())

    assert "module main" in model
    assert "function cond(a0: bv32): boolean;" in model
    assert "var commit_guard: boolean;" in model
    assert "define " not in model

    uclid = shutil.which("uclid")
    if uclid is None:
        pytest.skip("uclid is not installed")
    generated = tmp_path / "source.ucl"
    generated.write_text(model, encoding="ascii")
    result = subprocess.run([uclid, str(generated)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Successfully parsed 1 and instantiated 1 module(s)." in result.stdout


def transformed_state(name, commands):
    output = FSMOutputOp.create(
        attributes={
            "names": ArrayAttr(StringAttr(field) for field in (
                "nextInput", "commit", "muRollBack", "arrayRollBack", "rewind", "rbwe",
                "gammaRollBack_x0", "selSlowPath_x0", "stall_x0_0", "stall_x0_1",
            )),
            "values": DenseArrayBase.from_list(i64, commands),
        }
    )
    return FSMControllerStateOp.create(
        attributes={"name": StringAttr(name)},
        regions=[Region(Block([output])), Region(Block())],
    )


def transformed_slowfast_module():
    states = [
        transformed_state("Init_0", [0] * 10),
        transformed_state("Init_1", [0] * 10),
        transformed_state("Proceed", [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        transformed_state("0_0_Rollback", [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]),
        transformed_state("0_0_Fill_0", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    ]
    machine = FSMMachineOp.create(
        attributes={
            "sym_name": StringAttr("fsm_x0"),
            "function_type": FunctionType.from_lists([i32, i1], [i32, i1, i1, i32, i32, i32, i1, i32, i32, i32, i32]),
        },
        regions=[Region(Block(states))],
    )
    kernel = KernelOp("slowfast", ([i32], [i32]))
    initial = create_ssa_value(i32)
    task = TaskOp(StructType("commit_type", ["enable", "value"], [i1, i32]), "slowfast", [initial])
    block = task.body.block
    controller_state = MuOp.create(
        operands=[create_ssa_value(i32), create_ssa_value(i32)], result_types=[i32],
        attributes={"sym_name": StringAttr("fsm_x0State")},
    )
    trigger = FSMTriggerOp.create(
        operands=[controller_state.result, create_ssa_value(i1)],
        result_types=[i32, i1, i1, i32, i32, i32, i1, i32, i32, i32, i32],
        attributes={"instance": StringAttr("fsm_x0")},
    )
    value = MuOp.create(
        operands=[block.args[0], block.args[0]], result_types=[i32], attributes={"sym_name": StringAttr("x")}
    )
    cond = call("cond", value.result, i1)
    fast = call("fast", value.result, i32)
    slow = call("slow", value.result, i32)
    gamma = GammaOp("x", trigger.result[8], [fast.result[0], slow.result[0]])
    block.add_ops([
        controller_state,
        FSMInstanceOp.create(attributes={"name": StringAttr("fsm_x0"), "machine": FlatSymbolRefAttr("fsm_x0")} ),
        trigger,
        value,
        cond,
        fast,
        slow,
        gamma,
        CommitOp([trigger.result[2], gamma]),
    ])
    kernel.body.block.add_op(task)
    return ModuleOp([machine, kernel])


def test_emit_uclid_transformed_slowfast_uses_variables_and_checks_it(tmp_path):
    source = (
        Path(__file__).resolve().parents[2]
        / "test/Uclid/slowfast-end-to-end.speculated.mlir"
    )
    model = emit_uclid(parse_native_mlir(source.read_text(encoding="ascii")))

    assert "var mu_fsm_x0State: bv32;" in model
    assert "var rollback_v_" in model
    assert "var delay_7_0: bv32;" in model
    assert "var commit_guard: boolean;" in model
    assert "var committed_stream: [integer] bv32;" in model
    # The procedure stages register writes, so commit observes the guard computed
    # during this clock rather than the guard retained from the preceding clock.
    assert "procedure tick() modifies" in model
    assert "var v_" in model
    assert "if (next_commit_guard) {" in model
    assert "define " not in model
    assert "control {" not in model

    uclid = shutil.which("uclid")
    if uclid is None:
        pytest.skip("uclid is not installed")
    generated = tmp_path / "slowfast-vars.ucl"
    generated.write_text(model, encoding="ascii")
    result = subprocess.run([uclid, str(generated)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Successfully parsed 1 and instantiated 1 module(s)." in result.stdout


def test_emit_uclid_rejects_tasks_outside_the_slowfast_contract():
    task = TaskOp(StructType("result", ["value"], [i32]), "invalid", [])
    task.body.block.add_op(CommitOp([create_ssa_value(i32)]))
    kernel = KernelOp("invalid", ([], []), Region(Block([task])))

    with pytest.raises(UclidEmissionError, match="modeled dataflow"):
        emit_uclid(ModuleOp([kernel]))


def test_stuttering_driver_instantiates_reference_and_speculated_models(tmp_path):
    source = (
        Path(__file__).resolve().parents[2]
        / "test/Uclid/slowfast-end-to-end.speculated.mlir"
    )
    bundle = emit_stuttering_bisimulation_driver(slowfast_module(), parse_native_mlir(source.read_text(encoding="ascii")))

    assert "module ReferenceModel {" in bundle.reference_module
    assert "module SpeculatedModel {" in bundle.speculated_module
    assert "instance reference: ReferenceModel();" in bundle.driver_module
    assert "instance speculated: SpeculatedModel();" in bundle.driver_module
    assert "invariant bounded_stuttering_bisimulation" in bundle.driver_module
    assert [path.name for path in bundle.write(tmp_path)] == ["reference.ucl", "speculated.ucl", "driver.ucl"]
