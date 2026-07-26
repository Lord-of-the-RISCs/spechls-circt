"""Compatibility exports for task transform helpers and passes."""

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
