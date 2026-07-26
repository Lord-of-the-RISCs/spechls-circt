"""Pass for forwarding values across task stages."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass

from spechls.dialect import KernelOp

from .task_utils import forward_stage_values


@dataclass(frozen=True)
class ForwardStageValuesPass(ModulePass):
    """Forward direct task result fields through intervening task stages."""

    name = "spechls-forward-stage-values"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for kernel in list(op.walk()):
            if isinstance(kernel, KernelOp):
                forward_stage_values(kernel)
