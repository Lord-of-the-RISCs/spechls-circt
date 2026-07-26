"""Pass for inserting task-boundary FIFOs."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass

from spechls.dialect import KernelOp

from .task_utils import synchronize_task_fifos


@dataclass(frozen=True)
class SynchronizeTaskFIFOsPass(ModulePass):
    """Add FIFOs to direct task-result payload boundaries."""

    name = "spechls-synchronize-task-fifos"
    depth: int = 192

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for kernel in list(op.walk()):
            if isinstance(kernel, KernelOp):
                synchronize_task_fifos(kernel, self.depth)
