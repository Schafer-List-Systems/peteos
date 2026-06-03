"""ProcessSupervisor - OAP object that routes emails to pending tasks.

Each process has one supervisor instance that persists for the lifetime
of the process. Each invocation uses the same persistent thread ID
during a single dispatch cycle.
"""

from __future__ import annotations

import asyncio
import uuid as uuid_mod

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool

from apps.agentic_process._types import TaskState
from apps.agentic_process.email_client import (
    fetch_email,
    send_email,
)
from apps.agentic_process import config
from pathlib import Path


class ProcessSupervisor(AgenticObjectBase):
    """You are a process supervisor for the agentic process engine.

    Your job is to receive an email and decide which task(s) inside the
    process should handle it.

    Use your `get_email_subject` and `get_email_body` tools to read the
    email content.

    Then follow these steps:

    1. Extract any task IDs from the email subject or body. Task IDs
       are provided by task agents when they send requests to the user.
    2. If you found one or more task IDs, call `deliver_to_task` with
       each task ID. The tool will invoke the task agent and return a
       status. Follow the tool's instructions.
    3. If you found no task IDs, call `no_task_id_found` to reply to
       the user asking them to provide a task ID.
    4. If the process is closed, call `no_task_id_found` which will
       inform the user instead.
    5. If you encounter an error you cannot resolve, call `escalate`.

    Use `reply_to_user` whenever you need to communicate directly with
    the user. Use `escalate` for error conditions you cannot resolve.
    """

    def __init__(self, app_main, process_id: str):
        """Initialize with the AppMain instance and the target process ID.

        Args:
            app_main: The AppMain orchestrator instance.
            process_id: The ID of the process this supervisor manages.
        """
        super().__init__()
        self._uid: int | None = None
        self._dispatch_thread_id: str | None = None
        self._process_id = process_id
        self.app_main = app_main

    def _set_uid(self, uid: int) -> None:
        """Set the current email UID for dispatch.

        Raises:
            RuntimeError: If UID is already set.
        """
        if self._uid is not None:
            raise RuntimeError(
                f"UID already set to {self._uid}, cannot set to {uid}"
            )
        self._uid = uid

    def _get_uid(self) -> int | None:
        """Get the current email UID, or None if not set."""
        return self._uid

    def _set_dispatch_thread_id(self, thread_id: str) -> None:
        """Set the thread ID for this dispatch cycle.

        Raises:
            RuntimeError: If a thread ID is already set.
        """
        if self._dispatch_thread_id is not None:
            raise RuntimeError(
                f"Dispatch thread ID already set to {self._dispatch_thread_id}"
            )
        self._dispatch_thread_id = thread_id

    def _get_dispatch_thread_id(self) -> str | None:
        """Get the current dispatch thread ID, or None."""
        return self._dispatch_thread_id

    async def dispatch_email(self, uid: int, max_retries: int = 3) -> None:
        """Dispatch an email to the supervisor agent.

        Sets the UID and thread ID, invokes the agent with a persistent
        session, then resets both on completion.

        If the agent does not produce meaningful output, retries up to
        max_retries times before giving up.

        Args:
            uid: The IMAP UID of the email to process.
            max_retries: Number of retry attempts if the agent produces
                no meaningful output.
        """
        self._set_uid(uid)
        dispatch_id = uuid_mod.uuid4().hex[:8]
        self._set_dispatch_thread_id(dispatch_id)
        # Cache the email once per dispatch cycle
        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry "
            "when dispatch_email was called"
        )
        cached_inbox = Path(process.process_dir) / "cached_inbox"
        fetch_email(uid, cached_inbox=cached_inbox)
        try:
            for attempt in range(max_retries):
                await self.invoke_agent(
                    prompt=(
                        f"You received an email for process {self._process_id}. "
                        f"Route it to the correct task. "
                        f"If you need more context, use your tools. "
                        f"If you are sure what to do, take action now."
                    ),
                    persistent_thread_id=self._dispatch_thread_id,
                )
                # No specific completion signal — the supervisor's work
                # is done when the agent calls its tools.
                break
        finally:
            self._uid = None
            self._dispatch_thread_id = None

    # --- Tools available to the agent running the supervisor ---

    @tool
    def get_email_subject(self) -> str:
        """Fetch the subject line of the current email."""
        uid = self._get_uid()
        return fetch_email(uid).subject

    @tool
    def get_email_body(self) -> str:
        """Fetch the body text of the current email."""
        uid = self._get_uid()
        return fetch_email(uid).body_text

    @tool
    async def deliver_to_task(self, task_id: str) -> str:
        """Deliver the email to the specified task agent.

        Invokes the task via proceed() which handles activation,
        agent invocation, edge evaluation, and PENDING revert.

        Args:
            task_id: The ID of the task to deliver to.

        Returns:
            A status string describing the outcome.
        """
        uid = self._get_uid()
        if uid is None:
            return "ERROR: No email UID available."

        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry"
        )

        task = process.get_task(task_id)
        if task is None:
            return (
                f"ERROR: Task {task_id} not found in process "
                f"{self._process_id}."
            )

        # Invoke the task agent via proceed
        result_state = await task.proceed(email_uid=uid)

        # Check the result by looking at the task state
        if result_state == TaskState.OK:
            return (
                f"Task {task_id} processed the email successfully. "
                "Continue to the next pending task or finish."
            )
        elif task.state.value == "DENIED":
            return (
                f"Task {task_id} denied the request. "
                "Send an escalation for the entire process."
            )
        else:
            return (
                f"Task {task_id} is in state {task.state.value}. "
                "Check the task state and act accordingly."
            )

    @tool
    def no_task_id_found(self) -> str:
        """No task ID was found in the email.

        Checks if the process is terminated first. If so, informs the
        user that the process is closed. If the process is in its
        initial state (only one pending task), instructs the agent to
        route to that task and notify the user. Otherwise, asks the
        user to provide a task ID.

        Returns:
            A status string with instructions for the next action.
        """
        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry"
        )
        if process.is_terminated():
            return (
                f"Process {self._process_id} is closed. "
                "Inform the user that the process cannot receive emails."
            )
        start_id = process.get_start().task_id
        if len(process.pending_tasks) == 1 and next(iter(process.pending_tasks)) == start_id:
            return (
                f"Process {self._process_id} is in its initial state "
                f"with the start task '{start_id}' pending. Deliver the "
                "email to this task and notify the user that the process "
                f"has been opened, providing the process ID so they "
                "can reference it in future replies."
            )
        return (
            f"No task ID was found in the email. "
            "Inform the user that the email could not be associated "
            "with a specific task and they must include the task ID."
        )

    @tool
    def reply_to_user(self, body: str) -> None:
        """Send a reply to the user who sent the current email.

        The subject includes the process ID.

        Args:
            body: The body text of the reply.
        """
        uid = self._get_uid()
        info = fetch_email(uid)
        subject = f"[{self._process_id}] Re: {info.subject}"
        send_email(to=info.from_addr, subject=subject, body=body)

    @tool
    def escalate(self, subject: str, body: str) -> None:
        """Forward the email to the escalation address with an error note.

        The subject includes the process ID.

        Args:
            subject: Subject of the escalation email.
            body: Body text of the escalation email.
        """
        full_subject = f"ProcessSupervisor: {subject} [{self._process_id}]"
        send_email(
            to=config.ESCALATION_EMAIL,
            subject=full_subject,
            body=body,
        )