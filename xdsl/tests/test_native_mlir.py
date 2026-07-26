from pathlib import Path

from spechls.dialect import FieldOp, KernelOp, TaskOp
from spechls.native_mlir import NativeBoolConstantOp, parse_native_mlir


def test_native_boolean_constant_and_fsm_input_parse_without_explicit_boolean_type():
    module = parse_native_mlir("""module {
      %input = spechls.fsm.input 1 : i1
      %true = arith.constant true
      %match = arith.cmpi eq, %input, %true : i1
    }""")

    operations = tuple(module.body.block.ops)
    assert isinstance(operations[1], NativeBoolConstantOp)
    assert operations[0].result.type == operations[1].result.type
    assert operations[2].result.type == operations[1].result.type


def test_native_transformed_slowfast_parses_its_task_body_and_kernel_projection():
    source = (
        Path(__file__).resolve().parents[2]
        / "test/SpecHLS/Transforms/slowfast-end-to-end.transformed.mlir"
    )
    module = parse_native_mlir(source.read_text(encoding="ascii"))

    kernel = next(operation for operation in module.body.block.ops if isinstance(operation, KernelOp))
    task = next(operation for operation in kernel.body.block.ops if isinstance(operation, TaskOp))
    assert task.sym_name.data == "slowfast"
    assert task.body.block.args[0].type == kernel.body.block.args[0].type
    assert any(isinstance(operation, FieldOp) for operation in kernel.body.block.ops)
