"""Transforms for SpecHLS xDSL operations."""

from .extract_tasks import ExtractTasksPass
from .forward_stage_values import ForwardStageValuesPass
from .fuse_tasks import FuseTasksPass
from .inline_tasks import InlineTasksPass
from .synchronize_task_fifos import SynchronizeTaskFIFOsPass
from .task_utils import (
    extract_acyclic_tasks,
    extract_scc_tasks,
    forward_stage_values,
    fuse_adjacent_tasks,
    fuse_tasks,
    inline_task,
    outline_task,
    synchronize_task_fifos,
)
from .infer_speculation_fsm import InferSpeculationFSMPass
from .fsm_utils import (
    check_transition_table,
    infer_configured_speculation_fsms,
    infer_speculation_fsm,
    normalized_transition_table,
    spechls_fsm_body,
    validate_speculation_config,
)
from spechls.fsm_equivalence import (
    ComparisonResult,
    NormalizedFSM,
    NormalizedState,
    NormalizedTransition,
    compare_fsms,
    export_xdsl_fsm,
    load_java_fsm_fixture,
)

__all__ = [
    "ExtractTasksPass",
    "ComparisonResult",
    "InferSpeculationFSMPass",
    "NormalizedFSM",
    "NormalizedState",
    "NormalizedTransition",
    "ForwardStageValuesPass",
    "FuseTasksPass",
    "InlineTasksPass",
    "SynchronizeTaskFIFOsPass",
    "extract_acyclic_tasks",
    "extract_scc_tasks",
    "forward_stage_values",
    "fuse_adjacent_tasks",
    "fuse_tasks",
    "inline_task",
    "check_transition_table",
    "compare_fsms",
    "export_xdsl_fsm",
    "infer_configured_speculation_fsms",
    "infer_speculation_fsm",
    "load_java_fsm_fixture",
    "normalized_transition_table",
    "spechls_fsm_body",
    "validate_speculation_config",
    "outline_task",
    "synchronize_task_fifos",
]
