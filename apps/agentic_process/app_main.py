"""AppMain - central orchestrator for the agentic process engine.

Owns the process registry, workflow config, IMAP settings, and escalation
email address. Provides unified access to dispatcher, supervisor, and
process lifecycle.
"""

from __future__ import annotations

import glob as glob_mod
import json
import logging
from pathlib import Path

from .dispatcher import EmailDispatcher  # type: ignore[import-untyped]
from .process import Process
from .supervisor import ProcessSupervisor  # type: ignore[import-untyped]
from .workflow import Workflow

logger = logging.getLogger(__name__)

# Default workflow path relative to the app directory
_DEFAULT_WORKFLOW_PATH = str(Path(__file__).parent / "workflow.canvas")


class AppMain:
    """Central orchestrator for the agentic process application.

    Owns the process registry, IMAP connection state, workflow reference,
    and escalation email address. All routing is delegated through the
    dispatcher and supervisors.
    """

    def __init__(
        self,
        workflow_path: str | None = None,
        escalation_email: str | None = None,
    ) -> None:
        # Initialize empty registries
        self._processes: dict[str, Process] = {}
        self._supervisors: dict[str, ProcessSupervisor] = {}

        # Store the workflow path (loaded at startup, fixed for the lifetime
        # of the application — callers never pass a workflow path).
        self._workflow_path = workflow_path or _DEFAULT_WORKFLOW_PATH
        self._workflow: Workflow | None = None
        self.escalation_email = escalation_email

        # Dispatcher is created once at startup and stays alive.
        self.dispatcher: EmailDispatcher = EmailDispatcher(self)

        # Load the workflow definition used for all new processes.
        self._workflow = Workflow(self._workflow_path)

    @property
    def processes(self) -> dict[str, Process]:
        """The process registry."""
        return self._processes

    def create_process_from_workflow(self) -> str:
        """Create a new process and its supervisor from the workflow.

        Creates the process, persists it to disk, registers both the
        process and its supervisor, and returns the new process ID.

        Returns:
            The new process ID.
        """
        process = self._workflow.create_process()
        process.start()
        pid = process.metadata["metadata"]["process_id"]
        self._processes[pid] = process
        supervisor = ProcessSupervisor(self, pid)
        self._supervisors[pid] = supervisor
        return pid

    def get_or_create_supervisor(self, process_id: str) -> ProcessSupervisor:
        """Get the ProcessSupervisor for a process.

        Returns the cached supervisor if it exists. Otherwise, loads
        the process and supervisor from disk via `_load_supervisor`.

        Args:
            process_id: The process ID.

        Returns:
            The ProcessSupervisor for the process.
        """
        if process_id in self._supervisors:
            return self._supervisors[process_id]

        return self._load_supervisor(process_id)

    def _load_supervisor(self, process_id: str) -> ProcessSupervisor:
        """Load a process from disk and create its supervisor.

        Returns the cached supervisor if it exists. Loads the process
        from disk if not already in the registry, creates the supervisor,
        registers it, and returns it.

        Args:
            process_id: The process ID to load.

        Returns:
            The ProcessSupervisor for the process.

        Raises:
            RuntimeError: If the process is not found on disk and not
                in the registry.
        """
        if process_id in self._supervisors:
            return self._supervisors[process_id]

        base = Path(self._workflow_path).parent
        pattern = str(base / f"*-{process_id}")
        matches = glob_mod.glob(pattern)
        if not matches:
            raise RuntimeError(
                f"Process {process_id} not found on disk"
            )
        process_path = Path(matches[0]) / f"{Path(matches[0]).stem}.canvas"
        process = Process(str(process_path))
        self._processes[process_id] = process

        supervisor = ProcessSupervisor(self, process_id)
        self._supervisors[process_id] = supervisor
        return supervisor

    def find_process(self, process_id: str) -> Process | None:
        """Look up a process in the registry cache.

        Args:
            process_id: The process ID.

        Returns:
            The Process instance, or None if not found.
        """
        return self._processes.get(process_id)
