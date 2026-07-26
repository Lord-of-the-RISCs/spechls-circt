"""Pass for inlining task operations."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass

from spechls.dialect import TaskOp

from .task_utils import inline_task


@dataclass(frozen=True)
class InlineTasksPass(ModulePass):
    """Inline every task whose result interface is directly represented by fields."""

    name = "spechls-inline-tasks"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for task in list(op.walk()):
            if isinstance(task, TaskOp):
                inline_task(task)
