"""ProcessSupervisor - OAP object that routes emails to pending tasks.

Each process has one supervisor instance that persists for the lifetime
of the process. Each invocation uses the same persistent thread ID
during a single dispatch cycle.
"""

from __future__ import annotations

import asyncio
import uuid as uuid_mod

from apps.agentic_process.process import Process
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
    process should handle it. Your job is also to communicate with the user
    DELEGATE RELEVANT INFORMATION FROM THE TASK AGENTS TO THE USER.
    USE THE FEEDBACK AND TASK DESCRIPTIONS OF YOUR TASK AGENTS TO GATHER
    THE INFORMATION!

    Use your `get_email_subject` and `get_email_body` tools to read the
    user email content.

    Get the list of pending task agents and read their purpose to get the
    relevant task IDs for this email. Then trigger the relevant
    task agents using `trigger_task`. If no task is relevant then
    escalate or reply to the email depending on the email you got and
    the tasks that are pending.

    Use `set_reply` to compose a reply to the original sender and
    `set_escalation` for error conditions you cannot resolve. DO NOT
    MAKE UP INFORMATION, BUT USE THE EMAILS, THE TASK AGENT'S FEEDBACK
    AND THE TASK AGENT'S TEXTS TO COMPOSE THE ANSWER! DO NOT LEAVE
    THE USER IN THE DARK! When you are done composing your answer,
    then use `produce_output` to finish you turn and wait for the
    next E-Mail!
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
        self._seen_emails: dict[int, set[str]] = {}
        self._reply_body: str | None = None
        self._escalation: tuple[str, str] | None = None
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
        self._seen_emails = {uid: set()}
        self._reply_body = None
        self._escalation = None

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

    def _gather_feedback(self) -> dict[str, str]:
        """Collect non-empty feedback from all tasks in this process."""
        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry"
        )
        feedback: dict[str, str] = {}
        for task in process._tasks.values():
            fb = task.get_supervisor_feedback()
            if fb:
                feedback[task.task_id] = fb
        return feedback

    def _clear_feedback(self) -> None:
        """Clear the pending feedback from all tasks in this process."""
        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry"
        )
        for task in process._tasks.values():
            task._clear_feedback()

    async def dispatch_email(self, uid: int, max_retries: int = 3) -> None:
        """Dispatch an email to the supervisor agent.

        Sets the UID and thread ID, then:
        1. Process all ACTIVE tasks (including newly activated ones) until
           none remain.
        2. If PENDING tasks remain, invoke the agent to reason about them.
        3. If neither ACTIVE nor PENDING tasks remain, the process is
           terminated — invoke the agent to inform the user of the outcome.

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

        try:
            for attempt in range(max_retries):
                # Step 1: Process all ACTIVE tasks (including newly activated ones)
                while process.active_tasks:
                    task_id = process.active_tasks[-1]
                    task = process.get_task(task_id)
                    if task is not None:
                        await task.proceed(email_uid=uid)
                        self._tasks_triggered += 1
                        self._seen_emails[uid].add(task_id)

                # Filter out tasks that have already seen this email
                pending_ids = process.pending_tasks - self._seen_emails[uid]
                if pending_ids:
                    # Step 2: Agent reasons about remaining pending tasks
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
                    continue

                # No active tasks and no pending tasks — process is terminated.
                # Inform the user of the outcome.
                feedback = self._gather_feedback()
                if process.is_denied():
                    outcome_prompt = (
                        f"The process {self._process_id} is finished and has been denied. "
                        f"Inform the user accordingly! "
                        f"Escalate to an admin for review! "
                        f"When you are done, call the `produce_output` tool with an empty string."
                    )
                elif process.is_accepted():
                    if feedback:
                        outcome_prompt = (
                            f"The process {self._process_id} has been accepted, "
                            f"but the following tasks still have pending feedback:\n{feedback}\n"
                            f"Review the feedback! If necessary, escalate to an admin for review! "
                            f"Reply to the user with the necessary or required information! "
                            f"When you are done, call the `produce_output` tool with an empty string."
                        )
                    else:
                        outcome_prompt = (
                            f"The process {self._process_id} is finished and has been accepted. "
                            f"Inform the user accordingly! "
                            f"Escalate to an admin for review! "
                            f"Reply to the user with the necessary or required information! "
                            f"When you are done, call the `produce_output` tool with an empty string."
                        )
                elif feedback:
                    outcome_prompt = (
                        f"The process is paused due to task feedback. The following tasks have feedback:\n"
                        f"{feedback}\n"
                        f"Review the task feedback, formulate an appropriate reply to the user's email "
                        f"using the `set_reply` tool, and escalate if needed. "
                        f"When you are done, call the `produce_output` tool with an empty string."
                    )
                else:
                    outcome_prompt = (
                        f"The process {self._process_id} cannot proceed further. "
                        f"There are no active tasks, no pending tasks, "
                        f"and the process is neither denied nor accepted. "
                        f"Escalate this state to an admin. "
                        f"When you are done, call the `produce_output` tool with an empty string."
                    )
                if outcome_prompt:
                    await self.invoke_agent(
                        prompt=outcome_prompt,
                        persistent_thread_id=self._dispatch_thread_id,
                    )
                break
        finally:
            # Send formulated emails if any were set and clear pending feedback
            self._send_reply()
            self._send_escalation()
            self._clear_feedback()
            self._tasks_triggered = 0
            self._seen_emails.clear()
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
        elif task_id in self._seen_emails[uid]:
            return (
                f"Task {task_id} has already processed this email (UID {uid}). "
                "Nothing to do."
            )

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
        self._seen_emails[uid].add(task_id)

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
        """Get the list of pending task IDs in this process that have not
        yet seen the current email.

        Returns:
            A newline-separated list of pending task IDs.
        """
        process = self.app_main.find_process(self._process_id)
        assert process is not None, (
            f"Process {self._process_id} not in registry"
        )
        if not process.pending_tasks:
            return "No pending tasks."
        uid = self._get_uid()
        seen = self._seen_emails.get(uid, set()) if uid is not None else set()
        pending_ids = process.pending_tasks - seen
        if not pending_ids:
            return "No pending tasks."
        return "\n".join(sorted(pending_ids))

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
    def set_reply(self, body: str) -> str:
        """Set the reply email body to send to the original sender.

        Each call overwrites any previously set reply. The subject and
        recipient are derived automatically from the incoming email.
        The reply is only sent at the end of the dispatch cycle if
        non-empty. Use ``get_reply`` to recheck what is currently set.

        Args:
            body: The body text of the reply.

        Returns:
            A confirmation string.
        """
        self._reply_body = body
        return "Reply has been set. USE THE `produce_output` TOOL NOW, UNLESS YOU NEED TO ALSO ESCALATE AN ISSUE!"

    @tool
    def get_reply(self) -> str:
        """Get the currently set reply email body, or empty string."""
        if self._reply_body is None:
            return ""
        return self._reply_body

    def _send_reply(self) -> None:
        """Internal: send the formulated reply and reset."""
        if self._reply_body is None:
            return
        uid = self._get_uid()
        info = fetch_email(uid)
        subject = f"[{self._process_id}] Re: {info.subject}"
        send_email(to=info.from_addr, subject=subject, body=self._reply_body)
        self._reply_body = None

    @tool
    def set_escalation(self, subject: str, body: str) -> str:
        """Set the escalation email to send to the escalation address.

        Each call overwrites any previously set escalation. The subject
        is prefixed with ``ProcessSupervisor:`` and the process ID.
        The escalation is only sent at the end of the dispatch cycle if
        non-empty. Use ``get_escalation`` to recheck what is currently set.

        Args:
            subject: Subject of the escalation email.
            body: Body text of the escalation email.

        Returns:
            A confirmation string.
        """
        self._escalation = (subject, body)
        return "Escalation has been set. USE THE `produce_output` TOOL NOW!"

    @tool
    def get_escalation(self) -> str:
        """Get the currently set escalation email as 'subject | body', or empty string."""
        if self._escalation is None:
            return ""
        subject, body = self._escalation
        return f"{subject} | {body}"

    def _send_escalation(self) -> None:
        """Internal: send the formulated escalation and reset."""
        if self._escalation is None:
            return
        subject, body = self._escalation
        full_subject = f"ProcessSupervisor: {subject} [{self._process_id}]"
        send_email(
            to=config.ESCALATION_EMAIL,
            subject=full_subject,
            body=body,
        )
        self._escalation = None
