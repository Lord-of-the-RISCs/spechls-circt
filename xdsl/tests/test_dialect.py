from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, FlatSymbolRefAttr, IntegerAttr, StringAttr, i1, i8, i32, i64
from xdsl.ir import Block, Region
from xdsl.utils.test_value import create_ssa_value

from spechls.dialect import (
    AlphaOp,
    ArrayType,
    FSMControllerStateOp,
    FSMControllerTransitionOp,
    FSMInputOp,
    FSMInstanceOp,
    FSMOutputOp,
    FSMReturnOp,
    FSMTriggerOp,
    GammaOp,
    StructType,
)


def test_types_use_spechls_syntax():
    assert str(ArrayType(i8, 16)) == "!spechls.array<i8, 16>"
    assert str(StructType("pair", ["valid", "data"], [i1, i32])) == '!spechls.struct<"pair" { "valid" : i1, "data" : i32 }>'


def test_core_operation_verifiers():
    array = ArrayType(i8, 4)
    a, index, value, we = create_ssa_value(array), create_ssa_value(i32), create_ssa_value(i8), create_ssa_value(i1)
    AlphaOp.create(operands=[a, index, value, we], result_types=[array]).verify()
    GammaOp("g", create_ssa_value(i1), [create_ssa_value(i8), create_ssa_value(i8)]).verify()


def test_native_speculation_fsm_operations_are_registered_and_well_formed():
    output = FSMOutputOp.create(
        attributes={
            "names": ArrayAttr([StringAttr("commit")]),
            "values": DenseArrayBase.from_list(i64, [1]),
        }
    )
    state = FSMControllerStateOp.create(
        attributes={"name": StringAttr("Proceed")},
        regions=[Region(Block([output])), Region(Block())],
    )
    transition = FSMControllerTransitionOp.create(
        attributes={"target": StringAttr("Proceed"), "kind": StringAttr("normal")},
        regions=[Region(Block())],
    )
    state.transitions.block.add_op(transition)
    output.verify()

    input_op = FSMInputOp.create(
        result_types=[i1], attributes={"index": IntegerAttr(1, i64)}
    )
    returned = FSMReturnOp.create(operands=[input_op.result])
    guarded = FSMControllerTransitionOp.create(
        attributes={"target": StringAttr("Rollback"), "kind": StringAttr("normal")},
        regions=[Region(Block([returned]))],
    )
    returned.verify()

    instance = FSMInstanceOp.create(
        attributes={"name": StringAttr("fsm_x0"), "machine": FlatSymbolRefAttr("fsm_x0")}
    )
    trigger = FSMTriggerOp.create(
        operands=[create_ssa_value(i32), create_ssa_value(i1)], result_types=[i32, i1],
        attributes={"instance": StringAttr("fsm_x0")},
    )
    assert instance.name == "spechls.fsm.instance"
    assert trigger.name == "spechls.fsm.trigger"
    assert len(trigger.result) == 2
