from __future__ import annotations

from ._types import TaskState
from .edge import EdgeState
from .workflow import Workflow


class Process(Workflow):
    """Runtime instance of a Workflow."""

    def __init__(self, json_path: str) -> None:
        super().__init__(json_path)
        self._active_tasks: list[str] = []
        self._pending_tasks: set[str] = set()

    def start(self) -> None:
        """Activate the start task and enqueue it."""
        self._activate_task(self._start_task)

    def _activate_task(self, task: "Task") -> None:
        """Transition a task to ACTIVE and enqueue it."""
        task.state = TaskState.ACTIVE
        self._active_tasks.append(task.task_id)

    async def update(self) -> bool:
        """Process one step of the engine.

        Must be called iteratively by an external scheduler.
        Each call processes one task from the active queue.

        Returns True if there is more work to process, False otherwise.
        """
        if not self._active_tasks:
            return False

        # 1. Dequeue the first active task
        task_id = self._active_tasks.pop(0)
        task = self._tasks[task_id]

        # 2. Trigger the agent on this task and handle state changes
        state = await task.proceed()

        if state == TaskState.PENDING:
            # 3a. Task is waiting for external input
            #     Remove from active queue (already popped) and add to pending set
            self._pending_tasks.add(task_id)

        elif state == TaskState.DENIED:
            # 3b. Process is terminated
            #     Disable all outgoing edges on this task
            for edge in task.get_outgoing_edges():
                edge.state = EdgeState.DISABLED

        elif state == TaskState.OK:
            # 3c. Task completed successfully
            #     Agent has already evaluated outgoing edges and set their states.
            #     Now find all READY successor tasks and activate them.
            self._activate_ready_tasks(task)

        # 4. Check if this task's downstream has become DISABLED
        #    (all incoming edges of successor tasks are now DISABLED)
        self._propagate_disabled(task_id)

        return bool(self._active_tasks)

    def _activate_ready_tasks(self, completed_task: "Task") -> None:
        """Find all READY successor tasks of the completed task and transition them to ACTIVE."""
        for successor in completed_task.get_successor_tasks():
            if successor.state != TaskState.SCHEDULED:
                raise RuntimeError(
                    f"Successor {successor.task_id} is {successor.state}, "
                    f"expected SCHEDULED. A task should only be activated once."
                )

            if successor.is_ready():
                self._activate_task(successor)

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
                self._activate_task(task)
                # An ACTIVE task does not propagate DISABLED further