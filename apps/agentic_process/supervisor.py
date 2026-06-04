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

    Get the list of pending tasks and read their purpose to get the
    relevant task IDs for this email. Then trigger the relevant
    tasks using `trigger_task`. If no task is relevant then either
    escalate or reply to the email depending on the email you got and
    the tasks that are pending.

    Use `reply_to_email` whenever you need to communicate directly with
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
        self._tasks_triggered: int = 0
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

        Sets the UID and thread ID, then:
        1. Programmatically triggers all ACTIVE tasks.
        2. Invokes the agent to reason about PENDING tasks.
        3. Repeats steps 1-2 if new PENDING tasks appeared.

        If no task was triggered at all and no pending tasks exist,
        invokes the agent to decide whether to reply or escalate.

        Args:
            uid: The IMAP UID of the email to process.
            max_retries: Number of retry attempts if the agent produces
                no meaningful output.
        """
        self._set_uid(uid)
        dispatch_id = uuid_mod.uuid4().hex[:8]
        self._set_dispatch_thread_id(dispatch_id)
        self._tasks_triggered = 0

        # Cache the email once per dispatch cycle
        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry "
            "when dispatch_email was called"
        )
        cached_inbox = Path(process.process_dir) / "cached_inbox"
        fetch_email(uid, cached_inbox=cached_inbox)

        pending_ids = process.pending_tasks
        try:
            for attempt in range(max_retries):
                # Step 1: Programmatically trigger all ACTIVE tasks
                for active_id in process.active_tasks:
                    task = process.get_task(active_id)
                    if task is not None:
                        await task.proceed(email_uid=uid)
                        self._tasks_triggered += 1

                # Step 2: Agent reasons about PENDING tasks
                pending_text = "\n".join(sorted(pending_ids))
                prompt = (
                    f"You received an email. "
                    f"Read the email content with your tools. "
                    f"Then decide for each pending task if the email is relevant for that task. "
                    f"Read the task's text using `get_task_text` to understand the task's purpose and what information it needs. "
                    f"Trigger relevant tasks calling `trigger_task`. "
                    f"The process has the following pending tasks:\n{pending_text}"
                )
                await self.invoke_agent(
                    prompt=prompt,
                    persistent_thread_id=self._dispatch_thread_id,
                )

                # Check for new pending tasks
                new_pending = process.pending_tasks
                new_ids = new_pending - pending_ids
                if new_ids:
                    pending_ids = new_pending
                    continue
                if self._tasks_triggered == 0 and len(pending_ids) == 0:
                    await self.invoke_agent(
                        prompt=(
                            "You did not trigger any task and there are no "
                            "pending tasks. Decide whether to reply to the "
                            "user for more information or escalate the issue."
                        ),
                        persistent_thread_id=self._dispatch_thread_id,
                    )
                break
        finally:
            self._tasks_triggered = 0
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
    async def trigger_task(self, task_id: str) -> str:
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

        self._tasks_triggered += 1

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
    def get_pending_task_ids(self) -> str:
        """Get the list of task IDs for pending tasks in this process.

        Returns:
            A newline-separated list of pending task IDs.
        """
        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry"
        )
        if not process.pending_tasks:
            return "No pending tasks."
        return "\n".join(sorted(process.pending_tasks))

    @tool
    def get_task_text(self, task_id: str) -> str:
        """Get the current text of a task by its ID.

        Args:
            task_id: The ID of the task.

        Returns:
            The task text, or an error message if the task is not found.
        """
        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry"
        )
        task = process.get_task(task_id)
        if task is None:
            return f"ERROR: Task {task_id} not found in process {self._process_id}."
        return task.text

    @tool
    def reply_to_email(self, body: str) -> None:
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
