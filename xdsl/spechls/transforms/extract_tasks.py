"""Pass for outlining kernel computations into tasks."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass

from spechls.dialect import KernelOp

from .task_utils import extract_acyclic_tasks, extract_scc_tasks


@dataclass(frozen=True)
class ExtractTasksPass(ModulePass):
    """Outline legal cyclic components and remaining top-level acyclic runs."""

    name = "spechls-extract-tasks"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for kernel in list(op.walk()):
            if isinstance(kernel, KernelOp):
                extract_scc_tasks(kernel)
                extract_acyclic_tasks(kernel)
