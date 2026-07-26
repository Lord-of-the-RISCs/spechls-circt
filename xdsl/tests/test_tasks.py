import pytest
from xdsl.dialects import arith
from xdsl.context import Context
from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr, i1, i32
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from spechls.dialect import CommitOp, ExitOp, FIFOOp, FieldOp, KernelOp, MuOp, StructType, TaskOp
from spechls.transforms import (
    ExtractTasksPass,
    ForwardStageValuesPass,
    extract_acyclic_tasks,
    extract_scc_tasks,
    fuse_tasks,
    forward_stage_values,
    inline_task,
    outline_task,
    synchronize_task_fifos,
)


def test_kernel_and_task_require_single_block_with_correct_terminator():
    kernel = KernelOp("k", ([i1], []))
    kernel.body.block.add_op(ExitOp.build(operands=[kernel.body.block.args[0], []]))
    kernel.verify()
    with pytest.raises(VerifyException, match="exactly one block"):
        KernelOp("k", ([], []), Region([Block(), Block()])).verify()
    with pytest.raises(VerifyException, match="expects at least a terminator"):
        KernelOp("k", ([], []), Region(Block())).verify()

    result = StructType("result", ["enable"], [i1])
    task = TaskOp(result, "t", [create_ssa_value(i1)])
    task.body.block.add_op(CommitOp([task.body.block.args[0]]))
    task.verify()
    with pytest.raises(VerifyException, match="expects at least a terminator"):
        TaskOp(result, "t", [], Region(Block())).verify()


def test_outline_and_inline_task_round_trip():
    lhs, rhs = create_ssa_value(i32), create_ssa_value(i32)
    producer = arith.AddiOp(lhs, rhs)
    consumer = arith.AddiOp(producer.result, rhs)
    block = Block([producer, consumer])

    task = outline_task("sum", [producer])
    task.verify()
    assert [argument.type for argument in task.body.block.args] == [i32, i32]
    assert tuple(task.result.type.field_names)[1].data == "commit_val_0"
    field = consumer.lhs.owner
    assert field.name == "spechls.field"

    inline_task(task)
    assert all(op.name != "spechls.task" for op in block.ops)
    assert consumer.lhs.owner.name == "arith.addi"


def test_extract_scc_tasks_outlines_mu_recurrence_in_block_order():
    kernel = KernelOp("loop", ([i1, i32], [i32]))
    block = kernel.body.block
    init = block.args[1]
    mu = MuOp.create(operands=[init, init], result_types=[i32], attributes={"sym_name": StringAttr("iv")})
    update = arith.AddiOp(mu.result, init)
    mu.operands[1] = update.result
    block.add_ops([mu, update, ExitOp.build(operands=[block.args[0], [mu.result]])])

    tasks = extract_scc_tasks(kernel)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.sym_name.data == "scc_0"
    assert [op.name for op in task.body.block.ops] == ["spechls.mu", "arith.addi", "arith.constant", "spechls.commit"]
    assert task.body.block.first_op.operands[1].owner.name == "arith.addi"
    assert block.last_op.values[0].owner.name == "spechls.field"
    kernel.verify()


def test_extract_acyclic_tasks_outlines_top_level_computation():
    kernel = KernelOp("linear", ([i1, i32], [i32]))
    block = kernel.body.block
    add = arith.AddiOp(block.args[1], block.args[1])
    block.add_ops([add, ExitOp.build(operands=[block.args[0], [add.result]])])

    tasks = extract_acyclic_tasks(kernel)

    assert [task.sym_name.data for task in tasks] == ["acyclic_0"]
    assert [op.name for op in tasks[0].body.block.ops] == ["arith.addi", "arith.constant", "spechls.commit"]
    assert [op.name for op in block.ops] == ["spechls.task", "spechls.field", "spechls.exit"]
    kernel.verify()


def test_fuse_tasks_inlines_adjacent_task_pair_then_outlines_combined_body():
    kernel = KernelOp("chain", ([i1, i32], [i32]))
    block = kernel.body.block
    first_add = arith.AddiOp(block.args[1], block.args[1])
    block.add_ops([first_add, ExitOp.build(operands=[block.args[0], [first_add.result]])])
    first = outline_task("first", [first_add])
    first_value = next(op for op in block.ops if op.name == "spechls.field")
    second_add = arith.AddiOp(first_value.result, block.args[1])
    block.insert_op_before(second_add, block.last_op)
    block.last_op.operands[1] = second_add.result
    second = outline_task("second", [second_add])

    fused = fuse_tasks(first, second, "fused")

    assert fused.sym_name.data == "fused"
    assert [op.name for op in fused.body.block.ops] == ["arith.addi", "arith.addi", "arith.constant", "spechls.commit"]
    assert [op.name for op in block.ops] == ["spechls.task", "spechls.field", "spechls.exit"]
    kernel.verify()


def test_fuse_tasks_rejects_nonadjacent_pair():
    kernel = KernelOp("chain", ([i1, i32], []))
    block = kernel.body.block
    block.add_op(ExitOp.build(operands=[block.args[0], []]))
    first_add = arith.AddiOp(block.args[1], block.args[1])
    block.insert_op_before(first_add, block.last_op)
    first = outline_task("first", [first_add])
    separator = arith.AddiOp(block.args[1], block.args[1])
    block.insert_op_before(separator, block.last_op)
    second_add = arith.AddiOp(block.args[1], block.args[1])
    block.insert_op_before(second_add, block.last_op)
    second = outline_task("second", [second_add])

    with pytest.raises(ValueError, match="adjacent"):
        fuse_tasks(first, second)


def test_extract_tasks_pass_processes_kernel_bodies():
    kernel = KernelOp("loop", ([i1, i32], [i32]))
    block = kernel.body.block
    mu = MuOp.create(
        operands=[block.args[1], block.args[1]],
        result_types=[i32],
        attributes={"sym_name": StringAttr("iv")},
    )
    update = arith.AddiOp(mu.result, block.args[1])
    mu.operands[1] = update.result
    block.add_ops([mu, update, ExitOp.build(operands=[block.args[0], [mu.result]])])

    ExtractTasksPass().apply(Context(), ModuleOp([kernel]))

    assert isinstance(block.first_op, TaskOp)


def test_extract_tasks_pass_outlines_acyclic_kernel_body():
    kernel = KernelOp("linear", ([i1, i32], [i32]))
    block = kernel.body.block
    add = arith.AddiOp(block.args[1], block.args[1])
    block.add_ops([add, ExitOp.build(operands=[block.args[0], [add.result]])])

    ExtractTasksPass().apply(Context(), ModuleOp([kernel]))

    assert isinstance(block.first_op, TaskOp)
    assert block.first_op.sym_name.data == "acyclic_0"
    kernel.verify()


def test_synchronize_task_fifos_inserts_one_payload_fifo_and_is_idempotent():
    kernel = KernelOp("chain", ([i1, i32], [i32]))
    block = kernel.body.block
    first_add = arith.AddiOp(block.args[1], block.args[1])
    block.add_ops([first_add, ExitOp.build(operands=[block.args[0], [first_add.result]])])
    first = outline_task("first", [first_add])
    first_field = next(op for op in block.ops if op.name == "spechls.field")
    second_add = arith.AddiOp(first_field.result, block.args[1])
    block.insert_op_before(second_add, block.last_op)
    outline_task("second", [second_add])

    fifos = synchronize_task_fifos(kernel)

    assert len(fifos) == 1
    fifo = fifos[0]
    assert fifo.input.owner is first
    assert fifo.depth.value.data == 192
    assert first_field.input.owner is fifo
    assert synchronize_task_fifos(kernel) == []
    assert sum(isinstance(op, FIFOOp) for op in block.ops) == 1
    kernel.verify()


def test_synchronize_task_fifos_skips_fields_not_consumed_by_tasks():
    kernel = KernelOp("output", ([i1, i32], [i32]))
    block = kernel.body.block
    add = arith.AddiOp(block.args[1], block.args[1])
    block.add_ops([add, ExitOp.build(operands=[block.args[0], [add.result]])])
    outline_task("producer", [add])

    assert synchronize_task_fifos(kernel) == []
    assert all(not isinstance(op, FIFOOp) for op in block.ops)
    kernel.verify()


def test_synchronize_task_fifos_groups_multiple_fields_from_one_payload():
    kernel = KernelOp("fanout", ([i1, i32], [i32]))
    block = kernel.body.block
    left = arith.AddiOp(block.args[1], block.args[1])
    right = arith.SubiOp(block.args[1], block.args[1])
    combine = arith.AddiOp(left.result, right.result)
    block.add_ops([left, right, combine, ExitOp.build(operands=[block.args[0], [combine.result]])])
    outline_task("producer", [left, right])
    outline_task("consumer", [combine])

    fifos = synchronize_task_fifos(kernel)

    assert len(fifos) == 1
    fifo = fifos[0]
    assert sum(isinstance(op, FieldOp) and op.input.owner is fifo for op in block.ops) == 2
    kernel.verify()


def test_forward_stage_values_forwards_a_task_field_through_intermediate_stages():
    kernel = KernelOp("chain", ([i1, i32], [i32]))
    block = kernel.body.block
    producer_add = arith.AddiOp(block.args[1], block.args[1])
    block.add_ops([producer_add, ExitOp.build(operands=[block.args[0], [producer_add.result]])])
    producer = outline_task("producer", [producer_add])
    producer_field = next(op for op in block.ops if isinstance(op, FieldOp))

    middle_add = arith.SubiOp(block.args[1], block.args[1])
    block.insert_op_before(middle_add, block.last_op)
    middle = outline_task("middle", [middle_add])

    later_add = arith.SubiOp(block.args[1], block.args[1])
    block.insert_op_before(later_add, block.last_op)
    later = outline_task("later", [later_add])

    consumer_add = arith.AddiOp(producer_field.result, block.args[1])
    block.insert_op_before(consumer_add, block.last_op)
    consumer = outline_task("consumer", [consumer_add])

    forwarded = forward_stage_values(kernel)

    assert len(forwarded) == 2
    middle_projection, later_projection = forwarded
    assert middle_projection.input.owner is middle
    assert middle_projection.field_name.data == "forward_1"
    assert later_projection.input.owner is later
    assert [argument.type for argument in middle.body.block.args] == [i32, i32]
    assert tuple(middle.result.type.field_types) == (i1, i32)
    assert middle.body.block.last_op.value[-1] is middle.body.block.args[-1]
    assert later.body.block.last_op.value[-1] is later.body.block.args[-1]
    assert consumer.args[0] is later_projection.result
    assert forward_stage_values(kernel) == []
    assert sum(isinstance(op, FieldOp) and op.input.owner is middle for op in block.ops) == 1
    assert sum(isinstance(op, FieldOp) and op.input.owner is later for op in block.ops) == 1
    kernel.verify()


def test_forward_stage_values_pass_processes_kernel_bodies():
    kernel = KernelOp("chain", ([i1, i32], [i32]))
    block = kernel.body.block
    producer_add = arith.AddiOp(block.args[1], block.args[1])
    block.add_ops([producer_add, ExitOp.build(operands=[block.args[0], [producer_add.result]])])
    outline_task("producer", [producer_add])
    producer_field = next(op for op in block.ops if isinstance(op, FieldOp))
    middle_add = arith.SubiOp(block.args[1], block.args[1])
    block.insert_op_before(middle_add, block.last_op)
    outline_task("middle", [middle_add])
    consumer_add = arith.AddiOp(producer_field.result, block.args[1])
    block.insert_op_before(consumer_add, block.last_op)
    outline_task("consumer", [consumer_add])

    ForwardStageValuesPass().apply(Context(), ModuleOp([kernel]))

    assert any(isinstance(op, FieldOp) and op.input.owner.sym_name.data == "middle" for op in block.ops)
    kernel.verify()
