from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

from apps.agentic_process._types import EdgeState, TaskState
from apps.agentic_process.edge import Edge
from apps.agentic_process.process import Process
from apps.agentic_process.task import Task
from apps.agentic_process.workflow import Workflow


# ── Fixtures ──────────────────────────────────────────────────────────


def _create_workflow_json(
    nodes: list, edges: list, metadata: dict | None = None
) -> str:
    """Write a workflow JSON to a temp file and return the path."""
    if metadata is None:
        metadata = {"version": "1.0-1.0", "frontmatter": {}}
    fd, path = tempfile.mkstemp(suffix=".canvas")
    data = {"nodes": nodes, "edges": edges, "metadata": metadata}
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


def _chain_workflow(task_ids: list[str]) -> str:
    """Create a simple chain workflow: a -> b -> c ..."""
    nodes = [
        {"id": tid, "text": tid, "_task_state": "scheduled"} for tid in task_ids
    ]
    edges = [
        {"fromNode": task_ids[i], "toNode": task_ids[i + 1], "label": "Yes"}
        for i in range(len(task_ids) - 1)
    ]
    return _create_workflow_json(nodes, edges)


def _cycle_workflow(task_ids: list[str]) -> str:
    """Create a cycle workflow: a -> b -> ... -> a"""
    nodes = [
        {"id": tid, "text": tid, "_task_state": "scheduled"} for tid in task_ids
    ]
    edges = []
    for i in range(len(task_ids)):
        edges.append(
            {
                "fromNode": task_ids[i],
                "toNode": task_ids[(i + 1) % len(task_ids)],
                "label": "Yes",
            }
        )
    return _create_workflow_json(nodes, edges)


def _disjoint_workflow() -> str:
    """Create a workflow with an unreachable node."""
    nodes = [
        {"id": "a", "text": "A", "_task_state": "scheduled"},
        {"id": "b", "text": "B", "_task_state": "scheduled"},
        {"id": "c", "text": "C", "_task_state": "scheduled"},
    ]
    edges = [{"fromNode": "a", "toNode": "b", "label": "Yes"}]
    return _create_workflow_json(nodes, edges)


# ── Workflow loading ─────────────────────────────────────────────────


def test_load_creates_workflow():
    """Test that a valid workflow loads successfully."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        assert wf is not None
    finally:
        os.unlink(path)


def test_valid_workflow_has_one_start():
    """A valid workflow has exactly one start task."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        start = wf.get_start()
        assert start is not None
        assert start.task_id == "a"
    finally:
        os.unlink(path)


def test_valid_workflow_all_reachable():
    """All tasks are reachable from the start in a valid workflow."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        assert wf.is_valid()
    finally:
        os.unlink(path)


def test_cycle_detection():
    """A workflow with a cycle is invalid."""
    path = _cycle_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        assert not wf.is_valid()
    except ValueError:
        pass


def test_disjoint_graph_is_invalid():
    """A workflow with unreachable nodes is invalid."""
    path = _disjoint_workflow()
    try:
        wf = Workflow(path)
        assert not wf.is_valid()
    except ValueError:
        pass


# ── Graph queries ────────────────────────────────────────────────────


def test_is_predecessor_chain():
    """In a chain a->b->c, a is predecessor of c."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        c = wf._tasks["c"]
        assert wf.is_predecessor(a, c)
    finally:
        os.unlink(path)


def test_is_predecessor_not():
    """In a chain a->b->c, c is not predecessor of a."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        c = wf._tasks["c"]
        a = wf._tasks["a"]
        assert not wf.is_predecessor(c, a)
    finally:
        os.unlink(path)


def test_is_predecessor_direct():
    """In a chain a->b->c, a is direct predecessor of b."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        b = wf._tasks["b"]
        assert wf.is_predecessor(a, b)
    finally:
        os.unlink(path)


def test_is_predecessor_self():
    """A task is not its own predecessor."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        assert not wf.is_predecessor(a, a)
    finally:
        os.unlink(path)


# ── Task navigation ─────────────────────────────────────────────────


def test_task_navigation():
    """Test that tasks can navigate to successors and predecessors."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        b = wf._tasks["b"]
        c = wf._tasks["c"]
        assert b in a.get_successor_tasks()
        assert a in b.get_predecessor_tasks()
        assert c in b.get_successor_tasks()
        assert b in c.get_predecessor_tasks()
    finally:
        os.unlink(path)


# ── Edge tests ───────────────────────────────────────────────────────


def test_edge_condition():
    """Test that edge conditions are properly read from the label field."""
    path = _create_workflow_json(
        [
            {"id": "a", "text": "A", "_task_state": "SCHEDULED"},
            {"id": "b", "text": "B", "_task_state": "SCHEDULED"},
        ],
        [{"fromNode": "a", "toNode": "b", "label": "Ja"}],
    )
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        edge = a.get_outgoing_edges()[0]
        assert edge.condition == "Ja"
    finally:
        os.unlink(path)


def test_edge_default_state_is_scheduled():
    """A newly loaded edge should be in SCHEDULED state."""
    path = _create_workflow_json(
        [
            {"id": "a", "text": "A", "_task_state": "scheduled"},
            {"id": "b", "text": "B", "_task_state": "scheduled"},
        ],
        [{"fromNode": "a", "toNode": "b", "label": "Yes"}],
        metadata={"version": "1.0-1.0", "frontmatter": {}, "_edge_state": "SCHEDULED"},
    )
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        edge = a.get_outgoing_edges()[0]
        assert edge.state == EdgeState.SCHEDULED
    finally:
        os.unlink(path)


def test_edge_state_setter():
    """Edge state can be set to ENABLED or DISABLED."""
    path = _create_workflow_json(
        [
            {"id": "a", "text": "A", "_task_state": "scheduled"},
            {"id": "b", "text": "B", "_task_state": "scheduled"},
        ],
        [{"fromNode": "a", "toNode": "b", "label": "Yes"}],
    )
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        edge = a.get_outgoing_edges()[0]
        edge.state = EdgeState.ENABLED
        assert edge.state == EdgeState.ENABLED
        edge.state = EdgeState.DISABLED
        assert edge.state == EdgeState.DISABLED
    finally:
        os.unlink(path)


# ── Store / writeback ────────────────────────────────────────────────


def test_store_writes_json():
    """Test that store writes the workflow JSON to disk."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        wf._json_data["nodes"][0]["text"] = "Modified"
        wf.store()
        with open(path, "r") as f:
            data = json.load(f)
        assert data["nodes"][0]["text"] == "Modified"
    finally:
        os.unlink(path)


# ── Process creation ─────────────────────────────────────────────────


def test_create_process_copies_file():
    """Creating a process copies the JSON file to a new path."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        process = wf.create_process()
        assert process is not None
        assert isinstance(process, Process)
        assert process._json_path != path
    finally:
        os.unlink(path)


def test_process_inherits_workflow_methods():
    """A Process should inherit all Workflow methods."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        process = wf.create_process()
        assert hasattr(process, "is_valid")
        assert hasattr(process, "is_predecessor")
        assert hasattr(process, "get_start")
        assert hasattr(process, "store")
    finally:
        os.unlink(path)


def test_process_state_reset():
    """Test that created process has tasks in SCHEDULED state."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        wf._tasks["a"].state = wf._tasks["a"].state.__class__.OK
        process = wf.create_process()
        assert process._tasks["a"].state == wf._tasks["a"].state.__class__.SCHEDULED
    finally:
        os.unlink(path)


def test_process_id_written_to_metadata():
    """create_process writes process_id into the copied canvas metadata."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        process = wf.create_process()
        with open(process._json_path, "r") as f:
            data = json.load(f)
        assert "process_id" in data.get("metadata", {})
    finally:
        os.unlink(path)


def test_process_id_passed_to_tasks():
    """Tasks loaded from a process file should have _process_id set."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        process = wf.create_process()
        for task in process._tasks.values():
            assert task._process_id is not None
            assert len(task._process_id) == 4
    finally:
        os.unlink(path)


# ── Task.is_ready tests ──────────────────────────────────────────────


def test_is_ready_returns_false_no_incoming_edges():
    """is_ready returns False for tasks with no incoming edges (start task)."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        start = wf._tasks["a"]
        assert not start.is_ready()
    finally:
        os.unlink(path)


def test_is_ready_returns_false_with_scheduled_edges():
    """is_ready returns False when any incoming edge is SCHEDULED."""
    path = _create_workflow_json(
        [
            {"id": "a", "text": "A", "_task_state": "scheduled"},
            {"id": "b", "text": "B", "_task_state": "scheduled"},
        ],
        [{"fromNode": "a", "toNode": "b", "label": "Yes"}],
    )
    try:
        wf = Workflow(path)
        b = wf._tasks["b"]
        assert b.is_ready() is False
    finally:
        os.unlink(path)


def test_is_ready_returns_true_when_all_edges_resolved_and_ok():
    """is_ready returns True when all incoming edges are ENABLED and predecessors are OK."""
    path = _create_workflow_json(
        [
            {"id": "a", "text": "A", "_task_state": "ok"},
            {"id": "b", "text": "B", "_task_state": "scheduled"},
        ],
        [{"fromNode": "a", "toNode": "b", "label": "Yes"}],
    )
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        b = wf._tasks["b"]
        a_edge = a.get_outgoing_edges()[0]
        a_edge.state = EdgeState.ENABLED
        assert b.is_ready() is True
    finally:
        os.unlink(path)


def test_is_ready_returns_false_when_predecessor_not_ok():
    """is_ready returns False when an enabled predecessor is not OK."""
    path = _create_workflow_json(
        [
            {"id": "a", "text": "A", "_task_state": "active"},
            {"id": "b", "text": "B", "_task_state": "scheduled"},
        ],
        [{"fromNode": "a", "toNode": "b", "label": "Yes"}],
    )
    try:
        wf = Workflow(path)
        a = wf._tasks["a"]
        b = wf._tasks["b"]
        a_edge = a.get_outgoing_edges()[0]
        a_edge.state = EdgeState.ENABLED
        assert b.is_ready() is False
    finally:
        os.unlink(path)


# ── Process.start tests ──────────────────────────────────────────────


def test_start_activates_start_task():
    """start() activates the start task and adds it to _active_tasks."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        process = wf.create_process()
        assert process._active_tasks == []
        assert process._tasks["a"].state == TaskState.SCHEDULED
        process.start()
        assert "a" in process._active_tasks
        assert process._tasks["a"].state == TaskState.ACTIVE
    finally:
        os.unlink(path)


# ── Process.update tests ─────────────────────────────────────────────


def test_update_returns_false_when_no_active_tasks():
    """update() returns False when there are no active tasks."""
    path = _chain_workflow(["a", "b", "c"])
    try:
        wf = Workflow(path)
        process = wf.create_process()
        result = asyncio.run(process.update())
        assert result is False
    finally:
        os.unlink(path)


# ── Process._activate_ready_tasks tests ──────────────────────────────


def test_activate_ready_tasks_transitions_ready_successors():
    """_activate_ready_tasks transitions READY successors to ACTIVE."""
    path = _chain_workflow(["a", "b"])
    try:
        wf = Workflow(path)
        process = wf.create_process()
        process._tasks["a"].state = TaskState.OK
        process._tasks["a"].get_outgoing_edges()[0].state = EdgeState.ENABLED
        process._activate_ready_tasks(process._tasks["a"])
        assert process._tasks["b"].state == TaskState.ACTIVE
        assert "b" in process._active_tasks
    finally:
        os.unlink(path)


def test_activate_ready_tasks_skips_non_ready():
    """_activate_ready_tasks does not activate successors whose edges are not all resolved."""
    path = _create_workflow_json(
        [
            {"id": "start", "text": "S", "_task_state": "scheduled"},
            {"id": "a", "text": "A", "_task_state": "scheduled"},
            {"id": "b", "text": "B", "_task_state": "scheduled"},
        ],
        [
            {"fromNode": "start", "toNode": "a", "label": "to a"},
            {"fromNode": "start", "toNode": "b", "label": "to b"},
            {"fromNode": "a", "toNode": "b", "label": "Yes"},
        ],
    )
    try:
        wf = Workflow(path)
        process = wf.create_process()
        process._tasks["a"].state = TaskState.OK
        process._tasks["a"].get_outgoing_edges()[0].state = EdgeState.ENABLED
        process._activate_ready_tasks(process._tasks["a"])
        # b has 2 incoming edges: start→b (SCHEDULED), a→b (ENABLED)
        # Not READY because start→b is still SCHEDULED
        assert process._tasks["b"].state == TaskState.SCHEDULED
        assert "b" not in process._active_tasks
    finally:
        os.unlink(path)


def test_activate_ready_tasks_raises_on_already_active():
    """_activate_ready_tasks raises RuntimeError for non-SCHEDULED successors."""
    path = _create_workflow_json(
        [
            {"id": "a", "text": "A", "_task_state": "ok"},
            {"id": "b", "text": "B", "_task_state": "active"},
        ],
        [{"fromNode": "a", "toNode": "b", "label": "Yes"}],
    )
    try:
        wf = Workflow(path)
        process = wf.create_process()
        with pytest.raises(RuntimeError):
            process._activate_ready_tasks(process._tasks["a"])
    finally:
        os.unlink(path)


# ── Process._propagate_disabled tests ────────────────────────────────


def test_propagate_disabled_cascades():
    """_propagate_disabled cascades DISABLED state through the graph."""
    path = _chain_workflow(["a", "b"])
    try:
        wf = Workflow(path)
        process = wf.create_process()
        process._tasks["a"].get_outgoing_edges()[0].state = EdgeState.DISABLED
        process._propagate_disabled("a")
        assert process._tasks["b"].state == TaskState.DISABLED
    finally:
        os.unlink(path)


def test_propagate_disabled_activates_when_edge_enabled():
    """_propagate_disabled activates when DISABLED edge coexists with an ENABLED edge and predecessor OK."""
    path = _create_workflow_json(
        [
            {"id": "start", "text": "S", "_task_state": "scheduled"},
            {"id": "a", "text": "A", "_task_state": "scheduled"},
            {"id": "c", "text": "C", "_task_state": "scheduled"},
            {"id": "b", "text": "B", "_task_state": "scheduled"},
        ],
        [
            {"fromNode": "start", "toNode": "a", "label": "to a"},
            {"fromNode": "start", "toNode": "c", "label": "to c"},
            {"fromNode": "a", "toNode": "b", "label": "No"},
            {"fromNode": "c", "toNode": "b", "label": "Yes"},
        ],
    )
    try:
        wf = Workflow(path)
        process = wf.create_process()
        a_edge = process._tasks["a"].get_outgoing_edges()[0]
        c_edge = process._tasks["c"].get_outgoing_edges()[0]
        a_edge.state = EdgeState.DISABLED
        c_edge.state = EdgeState.ENABLED
        process._tasks["c"].state = TaskState.OK
        process._propagate_disabled("a")
        assert process._tasks["b"].state == TaskState.ACTIVE
    finally:
        os.unlink(path)


# ── Convergent graph test ────────────────────────────────────────────


def test_convergent_graph_activation():
    """A task with two predecessors activates only when both predecessors are OK."""
    path = _create_workflow_json(
        [
            {"id": "start", "text": "S", "_task_state": "scheduled"},
            {"id": "a", "text": "A", "_task_state": "scheduled"},
            {"id": "c", "text": "C", "_task_state": "scheduled"},
            {"id": "b", "text": "B", "_task_state": "scheduled"},
        ],
        [
            {"fromNode": "start", "toNode": "a", "label": "to a"},
            {"fromNode": "start", "toNode": "c", "label": "to c"},
            {"fromNode": "a", "toNode": "b", "label": "No"},
            {"fromNode": "c", "toNode": "b", "label": "Yes"},
        ],
    )
    try:
        wf = Workflow(path)
        process = wf.create_process()

        # Only task a is OK — edge is DISABLED, b has an ENABLED edge from c
        # whose predecessor (c) is not OK → b stays SCHEDULED
        process._tasks["a"].state = TaskState.OK
        a_edge = process._tasks["a"].get_outgoing_edges()[0]
        c_edge = process._tasks["c"].get_outgoing_edges()[0]
        a_edge.state = EdgeState.DISABLED
        c_edge.state = EdgeState.ENABLED
        process._activate_ready_tasks(process._tasks["a"])
        assert process._tasks["b"].state == TaskState.SCHEDULED

        # Now c is also OK → b becomes READY and should activate
        process._tasks["c"].state = TaskState.OK
        process._activate_ready_tasks(process._tasks["c"])
        assert process._tasks["b"].state == TaskState.ACTIVE
    finally:
        os.unlink(path)
