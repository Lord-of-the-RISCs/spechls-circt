"""Deterministic outlining and inlining utilities for ``spechls.task``."""

from __future__ import annotations

from collections.abc import Iterable
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i1, i32
from xdsl.ir import Operation, SSAValue

from spechls.dialect import CommitOp, FIFOOp, FieldOp, KernelOp, StructType, TaskOp


def outline_task(name: str, operations: Iterable[Operation]) -> TaskOp:
    """Move operations into a task and expose every result used outside it.

    The source operations must belong to one block. Their source-block order, rather
    than the iterable order, determines captures and result-field ordering.
    """
    selected = set(operations)
    if not selected:
        raise ValueError("cannot outline an empty operation set")
    block = next(iter(selected)).parent
    if block is None or any(op.parent is not block for op in selected):
        raise ValueError("outlined operations must belong to one block")
    ordered = [op for op in block.ops if op in selected]
    inputs: list[SSAValue] = []
    outputs: list[SSAValue] = []
    for op in ordered:
        for operand in op.operands:
            if operand.owner not in selected and operand not in inputs:
                inputs.append(operand)
        for result in op.results:
            if any(use.operation not in selected for use in result.uses):
                outputs.append(result)

    fields = ["enable", *(f"commit_val_{index}" for index in range(len(outputs)))]
    result_type = StructType(f"{name}_result", fields, [i1, *(value.type for value in outputs)])
    task = TaskOp(result_type, name, inputs)
    block.insert_op_before(task, ordered[0])
    body = task.body.block
    for op in ordered:
        op.detach()
        body.add_op(op)
    for input_value, argument in zip(inputs, body.args, strict=True):
        for op in ordered:
            for index, operand in enumerate(op.operands):
                if operand is input_value:
                    op.operands[index] = argument

    insertion_point: Operation = task
    for index, output in enumerate(outputs):
        field = FieldOp.create(
            operands=[task.result],
            result_types=[output.type],
            attributes={"name": StringAttr(fields[index + 1])},
        )
        block.insert_op_after(field, insertion_point)
        insertion_point = field
        output.replace_uses_with_if(field.result, lambda use: use.operation not in selected)

    enabled = arith.ConstantOp.from_int_and_width(1, 1)
    body.add_op(enabled)
    body.add_op(CommitOp([enabled.result, *outputs]))
    return task


def inline_task(task: TaskOp) -> None:
    """Inline a task whose result is consumed only by direct field extractions."""
    task.verify()
    block = task.parent
    if block is None:
        raise ValueError("task must belong to a block")
    body = task.body.block
    commit = body.last_op
    assert isinstance(commit, CommitOp)
    fields = [use.operation for use in task.result.uses]
    if any(not isinstance(field, FieldOp) or field.parent is not block for field in fields):
        raise ValueError("task results must only be used by fields in the containing block")
    field_values = {
        field.field_name.data: commit.value[
            [name.data for name in task.result.type.field_names].index(field.field_name.data)
        ]
        for field in fields
    }
    for argument, operand in zip(body.args, task.args, strict=True):
        argument.replace_all_uses_with(operand)
    for field in fields:
        field.result.replace_all_uses_with(field_values[field.field_name.data])
        field.detach()
        field.erase()
    commit.detach()
    commit.erase()
    for op in list(body.ops):
        op.detach()
        block.insert_op_before(op, task)
    task.detach()
    task.erase()


def extract_scc_tasks(kernel: KernelOp, name_prefix: str = "scc") -> list[TaskOp]:
    """Outline legal cyclic dataflow components in a kernel body.

    Edges follow SSA def-use relationships, including a ``MuOp``'s explicit
    ``loop_value`` operand. Components with captures or escaping uses that would
    become backward references after outlining are left in the kernel.
    """
    block = kernel.body.block
    operations = [op for op in block.ops if op is not block.last_op]
    order = {op: index for index, op in enumerate(operations)}
    operation_set = set(operations)
    successors = {
        op: [use.operation for result in op.results for use in result.uses if use.operation in operation_set]
        for op in operations
    }

    # Tarjan's algorithm visits both nodes and edges in block order, making task
    # naming and outlining order independent of set iteration.
    index = 0
    indices: dict[Operation, int] = {}
    lowlinks: dict[Operation, int] = {}
    stack: list[Operation] = []
    on_stack: set[Operation] = set()
    components: list[list[Operation]] = []

    def visit(op: Operation) -> None:
        nonlocal index
        indices[op] = index
        lowlinks[op] = index
        index += 1
        stack.append(op)
        on_stack.add(op)
        for successor in successors[op]:
            if successor not in indices:
                visit(successor)
                lowlinks[op] = min(lowlinks[op], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[op] = min(lowlinks[op], indices[successor])
        if lowlinks[op] == indices[op]:
            component: list[Operation] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member is op:
                    break
            components.append(component)

    for operation in operations:
        if operation not in indices:
            visit(operation)

    tasks: list[TaskOp] = []
    for component in sorted(components, key=lambda members: min(order[op] for op in members)):
        selected = set(component)
        # A one-node component is cyclic only when its result is consumed by itself.
        if len(selected) == 1 and next(iter(selected)) not in successors[next(iter(selected))]:
            continue
        first_index = min(order[op] for op in selected)
        legal = True
        for op in selected:
            for operand in op.operands:
                owner = operand.owner
                if isinstance(owner, Operation) and owner not in selected and owner.parent is block and order.get(owner, first_index) >= first_index:
                    legal = False
            for result in op.results:
                for use in result.uses:
                    user = use.operation
                    if user not in selected and user.parent is block and order.get(user, first_index) < first_index:
                        legal = False
        if legal:
            tasks.append(outline_task(f"{name_prefix}_{len(tasks)}", selected))
    return tasks


def extract_acyclic_tasks(kernel: KernelOp, name_prefix: str = "acyclic") -> list[TaskOp]:
    """Outline contiguous top-level computation between task interfaces.

    Fields remain at the top level so they continue to represent task interfaces.
    A run is left in place when moving its task before the run would introduce a
    backward capture or a backward result use.
    """
    block = kernel.body.block
    operations = list(block.ops)
    order = {op: index for index, op in enumerate(operations)}
    tasks: list[TaskOp] = []
    run: list[Operation] = []

    def outline_run() -> None:
        nonlocal run
        if not run:
            return
        selected = set(run)
        first_index = order[run[0]]
        legal = all(
            not (
                isinstance(operand.owner, Operation)
                and operand.owner not in selected
                and operand.owner.parent is block
                and order.get(operand.owner, first_index) >= first_index
            )
            for op in run
            for operand in op.operands
        ) and all(
            not (
                use.operation not in selected
                and use.operation.parent is block
                and order.get(use.operation, first_index) < first_index
            )
            for op in run
            for result in op.results
            for use in result.uses
        )
        if legal:
            tasks.append(outline_task(f"{name_prefix}_{len(tasks)}", run))
        run = []

    for operation in operations:
        if operation is block.last_op or isinstance(operation, (TaskOp, FieldOp)):
            outline_run()
        else:
            run.append(operation)
    outline_run()
    return tasks


def fuse_tasks(first: TaskOp, second: TaskOp, name: str | None = None) -> TaskOp:
    """Fuse an adjacent pair of tasks by inlining their bodies then outlining them.

    Only field extractions of ``first`` may separate the tasks. This preserves
    source-block ordering without introducing forwarding or FIFO stages.
    """
    block = first.parent
    if block is None or second.parent is not block:
        raise ValueError("tasks to fuse must belong to the same block")
    operations = list(block.ops)
    try:
        first_index = operations.index(first)
        second_index = operations.index(second)
    except ValueError as error:
        raise ValueError("tasks to fuse must belong to their containing block") from error
    if first_index >= second_index:
        raise ValueError("tasks to fuse must be in source-block order")
    if any(
        not isinstance(operation, FieldOp) or operation.input.owner is not first
        for operation in operations[first_index + 1 : second_index]
    ):
        raise ValueError("tasks to fuse must be adjacent apart from first-task fields")

    fused_name = name or first.sym_name.data
    def body_operations(task: TaskOp) -> tuple[list[Operation], Operation | None]:
        commit = task.body.block.last_op
        assert isinstance(commit, CommitOp)
        enable = commit.value[0].owner
        generated_enable = enable if isinstance(enable, arith.ConstantOp) else None
        return (
            [operation for operation in task.body.block.ops if operation not in (commit, generated_enable)],
            generated_enable,
        )

    first_operations, first_enable = body_operations(first)
    second_operations, second_enable = body_operations(second)
    inline_task(first)
    inline_task(second)
    for enable in (first_enable, second_enable):
        if enable is not None and not any(result.uses for result in enable.results):
            enable.detach()
            enable.erase()
    # Outlined tasks end in a generated enable constant and commit; only their
    # original body operations belong to the fused computation.
    return outline_task(fused_name, [*first_operations, *second_operations])


def fuse_adjacent_tasks(kernel: KernelOp) -> list[TaskOp]:
    """Fuse every adjacent task pair in a kernel until no such pair remains."""
    fused: list[TaskOp] = []
    while True:
        tasks = [op for op in kernel.body.block.ops if isinstance(op, TaskOp)]
        for first, second in zip(tasks, tasks[1:]):
            try:
                fused.append(fuse_tasks(first, second))
                break
            except ValueError:
                continue
        else:
            return fused


def synchronize_task_fifos(kernel: KernelOp, depth: int = 192) -> list[FIFOOp]:
    """Insert one FIFO at each direct task-to-task struct payload boundary.

    A task result is synchronized only when its complete top-level interface is
    represented by fields in the kernel block and at least one such field is a
    direct argument of another task. Rewiring every field through the FIFO keeps
    the struct payload coherent; mixed or nested interfaces are left untouched.
    """
    if depth <= 0:
        raise ValueError("FIFO depth must be positive")
    block = kernel.body.block
    fifos: list[FIFOOp] = []
    for task in list(block.ops):
        if not isinstance(task, TaskOp):
            continue
        users = [use.operation for use in task.result.uses]
        if not users or any(not isinstance(user, FieldOp) or user.parent is not block for user in users):
            continue
        fields = [user for user in users if isinstance(user, FieldOp)]
        if not any(
            isinstance(use.operation, TaskOp) and use.operation is not task and use.operation.parent is block
            for field in fields
            for use in field.result.uses
        ):
            continue
        # A FIFO directly consuming this result already defines its payload boundary.
        if any(isinstance(user, FIFOOp) for user in users):
            continue
        fifo = FIFOOp.create(
            operands=[task.result],
            result_types=[task.result.type],
            attributes={"depth": IntegerAttr(depth, 32)},
        )
        block.insert_op_after(fifo, task)
        for field in fields:
            field.input = fifo.result
        fifos.append(fifo)
    return fifos


def forward_stage_values(kernel: KernelOp) -> list[FieldOp]:
    """Forward direct task fields through every intervening lexical task stage.

    Only ordinary top-level task interfaces are supported: their first result field
    and commit operand must be the ``i1`` enable, and every result field must match
    a commit operand. Dependencies with any other layout are left unchanged.
    """
    block = kernel.body.block
    added_fields: list[FieldOp] = []

    def task_commit(task: TaskOp) -> CommitOp | None:
        result_type = task.result.type
        commit = task.body.block.last_op
        if (
            not isinstance(commit, CommitOp)
            or not result_type.field_types
            or result_type.field_types.data[0] != i1
            or len(result_type.field_names) != len(commit.value)
            or tuple(value.type for value in commit.value) != result_type.field_types.data
        ):
            return None
        return commit

    def field_index(field: FieldOp) -> int | None:
        names = [name.data for name in field.input.type.field_names]
        try:
            index = names.index(field.field_name.data)
        except ValueError:
            return None
        return index if index and field.result.type == field.input.type.field_types.data[index] else None

    def forward(task: TaskOp, value: SSAValue) -> FieldOp | None:
        commit = task_commit(task)
        if commit is None or task.parent is not block:
            return None
        body = task.body.block
        # Reuse an existing passthrough, including one created by an earlier run.
        for operand, argument in zip(task.args, body.args, strict=True):
            if operand is value:
                index = next((index for index, output in enumerate(commit.value) if output is argument), None)
                if index is None or index == 0:
                    continue
                name = task.result.type.field_names[index].data
                projections = [
                    use.operation
                    for use in task.result.uses
                    if isinstance(use.operation, FieldOp)
                    and use.operation.parent is block
                    and use.operation.field_name.data == name
                ]
                return projections[0] if len(projections) == 1 else None

        names = [name.data for name in task.result.type.field_names]
        field_name = f"forward_{len(names)}"
        suffix = 0
        while field_name in names:
            suffix += 1
            field_name = f"forward_{len(names)}_{suffix}"
        task.operands = [*task.operands, value]
        argument = body.insert_arg(value.type, len(body.args))
        result_type = StructType(
            task.result.type.struct_name,
            [*task.result.type.field_names, field_name],
            [*task.result.type.field_types, value.type],
        )
        result = task.result.__replace__(_type=result_type)
        task.result.replace_all_uses_with(result)
        task.results = (result,)
        commit.operands = [*commit.operands, argument]
        projection = FieldOp.create(
            operands=[task.result], result_types=[value.type], attributes={"name": StringAttr(field_name)}
        )
        block.insert_op_after(projection, task)
        added_fields.append(projection)
        return projection

    # Take a snapshot because forwarding extends task interfaces while walking it.
    tasks = [operation for operation in block.ops if isinstance(operation, TaskOp)]
    positions = {task: index for index, task in enumerate(tasks)}
    for target in tasks:
        for operand_index, operand in enumerate(tuple(target.args)):
            source_field = operand.owner
            if not isinstance(source_field, FieldOp) or source_field.parent is not block:
                continue
            source = source_field.input.owner
            if (
                not isinstance(source, TaskOp)
                or source.parent is not block
                or task_commit(source) is None
                or field_index(source_field) is None
            ):
                continue
            source_index = positions.get(source)
            target_index = positions[target]
            if source_index is None or source_index >= target_index:
                continue
            stages = tasks[source_index + 1 : target_index]
            if any(task_commit(stage) is None for stage in stages):
                continue
            value = operand
            for stage in stages:
                projection = forward(stage, value)
                if projection is None:
                    break
                value = projection.result
            else:
                target.operands[operand_index] = value
    return added_fields
