"""Persistence — list persisted AgenticObject instances."""

from peteos.config import ConfigManager
from peteos.oap.agentic_object import AgenticObject
from peteos.utils import json


def list_instances() -> list[str]:
    """Return all persisted instance names found in INSTANCES_DIR, oldest first."""
    # Locate the instances directory via ConfigManager variable expansion
    inst_dir = ConfigManager.resolve_path("{INSTANCES_DIR}")

    # Guard: nothing persisted yet
    if not inst_dir.exists():
        return []

    # Pair each instance directory with its inode-change time (creation time on Linux)
    # to sort oldest first
    entries = [(d.stat().st_ctime, d.name) for d in inst_dir.iterdir() if d.is_dir()]
    entries.sort(key=lambda x: x[0])

    # Extract names in creation-order
    return [name for _, name in entries]

