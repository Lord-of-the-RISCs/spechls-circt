"""Pass for inferring speculation FSMs."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass

from .fsm_utils import infer_configured_speculation_fsms


@dataclass(frozen=True)
class InferSpeculationFSMPass(ModulePass):
    """Emit conservative FSM machines from explicit speculation configuration."""

    name = "spechls-infer-speculation-fsm"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        infer_configured_speculation_fsms(op)
