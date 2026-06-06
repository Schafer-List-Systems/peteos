from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import TYPE_CHECKING

from .edge import Edge
from .task import Task

if TYPE_CHECKING:
    from .process import Process


class Workflow:
    """Wraps a JSON workflow definition from a file."""

    def __init__(self, json_path: str) -> None:
        self.load(json_path)

    def load(self, json_path: str) -> Workflow:
        """Load a workflow definition from a JSON file."""
        with open(json_path, "r") as f:
            json_data = json.load(f)

        # Build edges first
        edge_objs: list[Edge] = []
        for edge_d in json_data.get("edges", []):
            edge_d.setdefault("_edge_state", "scheduled")
            edge_objs.append(Edge(edge_d))

        # Build tasks
        process_id = json_data.get("metadata", {}).get("process_id")
        tasks: dict[str, Task] = {}
        for node_d in json_data.get("nodes", []):
            node_d.setdefault("_task_state", None)
            tasks[node_d["id"]] = Task(node_d, process_id=process_id)

        # Wire edges to tasks
        for edge_obj in edge_objs:
            edge_obj._from_task = tasks[edge_obj.from_task_id]
            edge_obj._to_task = tasks[edge_obj.to_task_id]

        # Wire tasks to edges
        for node_d in json_data.get("nodes", []):
            task = tasks[node_d["id"]]
            task._outgoing_edges = [
                e for e in edge_objs if e.from_task_id == node_d["id"]
            ]
            task._incoming_edges = [
                e for e in edge_objs if e.to_task_id == node_d["id"]
            ]

        # Identify supervisor node: the one node with no incoming AND no outgoing edges
        supervisor_node_data = None
        for node_d in json_data.get("nodes", []):
            task = tasks[node_d["id"]]
            if not task._incoming_edges and not task._outgoing_edges:
                supervisor_node_data = node_d
                break

        if supervisor_node_data:
            del tasks[supervisor_node_data["id"]]

        # Find start task (no incoming edges, from remaining tasks)
        start_task = next(
            (t for t in tasks.values() if not t._incoming_edges),
            None,
        )

        # Wire tasks to their parent workflow/process instance
        for task in tasks.values():
            task._process = self

        working_directory = os.path.dirname(json_path) or "."

        self._json_path = json_path
        self._auto_flush: bool = True
        self._json_data = json_data
        self._tasks = tasks
        self._edges = edge_objs
        self._start_task = start_task
        self._working_directory = working_directory
        self._supervisor_node_data = supervisor_node_data

        if not self.is_valid():
            raise ValueError("Workflow is not valid")

        return self

    def store(self, json_path: str | None = None) -> None:
        """Write the JSON data back to the workflow file."""
        path = json_path or self._json_path
        with open(path, "w") as f:
            json.dump(self._json_data, f, indent=2)

    def refresh(self) -> None:
        """Rebuild active_tasks and pending_tasks from current task states.

        Calls the Task.state setter on every task to repopulate the
        _active_tasks and _pending_tasks sets without writing the file N times.
        Disables auto-flush, re-applies each task's state, then stores once.
        """
        had_flush = self._auto_flush
        self._auto_flush = False
        for task in self._tasks.values():
            current = task.state
            task.state = current
        self._auto_flush = had_flush
        self.store()

    def get_start(self) -> Task:
        """Return the start task (the one with no incoming edges)."""
        return self._start_task

    def is_valid(self) -> bool:
        """True if exactly one start node, no cycles, all nodes reachable."""
        if self._start_task is None:
            return False
        if len([t for t in self._tasks.values() if not t._incoming_edges]) != 1:
            return False
        # Check all nodes reachable from start
        visited: set[str] = set()
        queue: list[Task] = [self._start_task]
        while queue:
            current = queue.pop(0)
            if current.task_id in visited:
                continue
            visited.add(current.task_id)
            queue.extend(current.get_successor_tasks())
        if len(visited) != len(self._tasks):
            return False
        # Check no cycles
        color: dict[str, int] = {t.task_id: 0 for t in self._tasks.values()}
        for t in self._tasks.values():
            if color[t.task_id] == 0 and not self._dfs_has_cycle(t, color):
                return False
        return True

    def _dfs_has_cycle(self, task: Task, color: dict[str, int]) -> bool:
        """Returns True if no cycle was found."""
        color[task.task_id] = 1
        for succ in task.get_successor_tasks():
            if color[succ.task_id] == 1:
                return False
            if color[succ.task_id] == 0 and not self._dfs_has_cycle(succ, color):
                return False
        color[task.task_id] = 2
        return True

    def is_predecessor(self, candidate: Task, descendant: Task) -> bool:
        """True if candidate is a direct or indirect predecessor of descendant."""
        if candidate.task_id == descendant.task_id:
            return False
        visited: set[str] = set()
        queue: list[Task] = [descendant]
        while queue:
            current = queue.pop(0)
            if current.task_id == candidate.task_id:
                return True
            if current.task_id in visited:
                continue
            visited.add(current.task_id)
            queue.extend(current.get_predecessor_tasks())
        return False

    def create_process(self) -> Process:
        """Factory: create a new independent Process from this Workflow."""
        process_id = uuid.uuid4().hex[:4].upper()
        dirname, basename = os.path.split(self._json_path)
        name, ext = os.path.splitext(basename)
        process_dir = os.path.join(dirname, f"{name}-{process_id}")
        process_path = os.path.join(process_dir, f"{name}-{process_id}{ext}")

        if os.path.exists(process_dir):
            raise FileExistsError(
                f"Process directory already exists: {process_dir}. "
                "Expected an empty directory for a new process instance."
            )
        if os.path.exists(process_path):
            raise FileExistsError(
                f"Process canvas already exists: {process_path}. "
                "Existing processes must not be overwritten."
            )
        os.makedirs(process_dir)
        shutil.copy2(self._json_path, process_path)

        # Write process_id to the copied canvas's metadata
        with open(process_path, "r") as f:
            process_json_data = json.load(f)
        process_json_data["metadata"]["process_id"] = process_id
        with open(process_path, "w") as f:
            json.dump(process_json_data, f, indent=2)

        from .process import Process

        process = Process(process_path)
        process._process_dir = process_dir
        return process
