"""Pass for fusing adjacent task operations."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass

from spechls.dialect import KernelOp

from .task_utils import fuse_adjacent_tasks


@dataclass(frozen=True)
class FuseTasksPass(ModulePass):
    """Fuse adjacent tasks without introducing forwarding or FIFO stages."""

    name = "spechls-fuse-tasks"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for kernel in list(op.walk()):
            if isinstance(kernel, KernelOp):
                fuse_adjacent_tasks(kernel)
