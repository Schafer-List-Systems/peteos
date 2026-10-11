"""Persistence — list and restore persisted AgenticObject instances."""

import importlib

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


def restore_instance(name: str) -> AgenticObject:
    """Restore a persisted instance by name.

    Loads agentic_object.json, imports the class, instantiates it,
    then calls restore(). Returns the restored instance.
    """
    # Resolve and load the persisted JSON for this instance
    json_path = ConfigManager.resolve_path(f"{{INSTANCES_DIR}}/{name}/agentic_object.json")
    with open(json_path) as f:
        data = json.load(f)

    # Import the class from its stored module path
    class_info = data["class"]
    mod = importlib.import_module(class_info["module"])
    cls = getattr(mod, class_info["name"])

    # Instantiate with the persisted name — persistent=True ensures
    # the existing JSON is not overwritten on re-initialisation
    instance: AgenticObject = cls(name=name, persistent=True)

    # Run all restore hooks in dependency order
    instance.restore()

    # Return the fully restored instance
    return instance
