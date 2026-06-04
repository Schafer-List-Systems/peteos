from __future__ import annotations

from ._types import TaskState
from .edge import EdgeState
from .task import Task
from .workflow import Workflow


class Process(Workflow):
    """Runtime instance of a Workflow."""

    def __init__(self, json_path: str) -> None:
        super().__init__(json_path)
        self._active_tasks: list[str] = []
        self._pending_tasks: set[str] = set()
        # The directory this process instance lives in (set when the process
        # is created from a workflow in Workflow.create_process()).
        self._process_dir: str = ""

    def start(self) -> None:
        """Start the process with the start task in ACTIVE state.

        The start task begins as ACTIVE — it transitions to PENDING
        when it needs more information.
        """
        self._start_task.state = TaskState.ACTIVE

    def _activate_ready_tasks(self, completed_task: "Task") -> None:
        """Find all READY successor tasks of the completed task and transition them to ACTIVE."""
        for successor in completed_task.get_successor_tasks():
            if successor.state != TaskState.SCHEDULED:
                raise RuntimeError(
                    f"Successor {successor.task_id} is {successor.state}, "
                    f"expected SCHEDULED. A task should only be activated once."
                )

            if successor.is_ready():
                # At least one incoming edges of the successor must be enabled
                for edge in successor._incoming_edges:
                    if edge.state == EdgeState.ENABLED:
                        successor.activate()
                        break

    def _propagate_disabled(self, task_id: str) -> None:
        """Find downstream tasks that have become DISABLED and cascade.

        A task becomes DISABLED when all its incoming edges are DISABLED.
        Disabled tasks automatically disable their own outgoing edges,
        which may cause further tasks to become DISABLED.

        Start traversal from the given task's successors.
        """
        queue: list[str] = []
        for edge in self._tasks[task_id].get_outgoing_edges():
            if edge.state == EdgeState.DISABLED:
                queue.append(edge.get_to_task().task_id)

        while queue:
            successor_id = queue.pop(0)
            task = self._tasks[successor_id]
            if task.state != TaskState.SCHEDULED:
                continue

            incoming_edges = task.get_incoming_edges()

            if all(e.state == EdgeState.DISABLED for e in incoming_edges):
                # All incoming edges disabled — this task becomes DISABLED
                task.state = TaskState.DISABLED
                for edge in task.get_outgoing_edges():
                    edge.state = EdgeState.DISABLED
                for edge in task.get_outgoing_edges():
                    queue.append(edge.get_to_task().task_id)
            elif task.is_ready():
                task.activate()
                # An ACTIVE task does not propagate DISABLED further

    def is_terminated(self) -> bool:
        """True if no tasks remain active, pending, or can become active."""
        terminal_states = {TaskState.OK, TaskState.DENIED, TaskState.DISABLED}
        return all(t.state in terminal_states for t in self._tasks.values())

    def is_denied(self) -> bool:
        """True if any task has been denied."""
        return any(t.state == TaskState.DENIED for t in self._tasks.values())

    def is_accepted(self) -> bool:
        """True if the process is terminated and not denied."""
        return self.is_terminated() and not self.is_denied()

    @property
    def pending_tasks(self) -> set[str]:
        """Return the set of pending task IDs."""
        return self._pending_tasks

    @property
    def active_tasks(self) -> list[str]:
        """Return the list of active task IDs."""
        return self._active_tasks

    def get_task(self, task_id: str) -> Task | None:
        """Return the task with the given ID, or None."""
        return self._tasks.get(task_id)

    @property
    def metadata(self) -> dict:
        """Return the process JSON metadata."""
        return self._json_data

    @property
    def process_dir(self) -> str:
        """Return the directory where this process instance lives."""
        return self._process_dir
