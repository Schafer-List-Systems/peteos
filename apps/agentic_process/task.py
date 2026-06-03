from __future__ import annotations

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool

from ._types import EdgeState, TaskState
from apps.agentic_process.email_client import (
    get_cached_email,
    list_cached_emails,
    send_email,
)
from pathlib import Path


class Task(AgenticObjectBase):
    """You are a task agent within an agentic process engine.

    Read your purpose from the task text using `get_text`. During the
    conversation, update the task text via `set_text` to persist facts
    and provide progress information. This text is also read by your
    supervisor and used to redirect incoming emails towards you if
    there is relevant information for you. That's why you also need to
    provide the information about requests in the task text.

    You get the information required to fulfill your purpose from the
    emails. If the information you need to fulfill your purpose is not
    available, you can request it via email. If your purpose cannot be
    satisfied even after requesting further information When you deny
    this task and therefore the whole process you are part of.

    Use `accept` to complete the task successfully or `deny` to reject it.
    """

    def __init__(self, node_data: dict, process_id: str | None = None) -> None:
        super().__init__()
        self._node_data = node_data
        self._outgoing_edges: list[Edge] = []
        self._incoming_edges: list[Edge] = []
        self._process_id = process_id
        self._process: object | None = None

    def get_outgoing_edges(self) -> list[Edge]:
        return self._outgoing_edges

    def get_incoming_edges(self) -> list[Edge]:
        return self._incoming_edges

    def get_successor_tasks(self) -> list[Task]:
        return [edge.get_to_task() for edge in self._outgoing_edges]

    def get_predecessor_tasks(self) -> list[Task]:
        return [edge.get_from_task() for edge in self._incoming_edges]

    @property
    def node_data(self) -> dict:
        return self._node_data

    @property
    def task_id(self) -> str:
        return self._node_data["id"]

    _COLORS: dict[TaskState, str] = {
        TaskState.SCHEDULED: "#808080",
        TaskState.ACTIVE: "#3498db",
        TaskState.OK: "#2ecc71",
        TaskState.DENIED: "#e74c3c",
        TaskState.PENDING: "#f39c12",
        TaskState.DISABLED: "#636e72",
    }

    @property
    def state(self) -> TaskState:
        raw = self._node_data.get("_task_state")
        return TaskState(raw) if raw is not None else TaskState.SCHEDULED

    @state.setter
    def state(self, value: TaskState) -> None:
        self._node_data["_task_state"] = value.value
        self._node_data["color"] = self._COLORS[value]
        process = self._process
        if process is not None:
            if value == TaskState.PENDING and self.task_id not in process._pending_tasks:
                process._pending_tasks.add(self.task_id)
            elif value != TaskState.PENDING and self.task_id in process._pending_tasks:
                process._pending_tasks.discard(self.task_id)
            process.store()

    @property
    def text(self) -> str:
        return self._node_data.get("text", "")

    @text.setter
    def text(self, value: str) -> None:
        self._node_data["text"] = value
        process = self._process
        if process is not None:
            process.store()

    @tool
    def get_text(self) -> str:
        """Get the current text of this task."""
        return self.text

    @tool
    def set_text(self, value: str) -> None:
        """Set the text of this task."""
        self.text = value

    @property
    def metadata(self) -> dict:
        return self._node_data

    def is_ready(self) -> bool:
        """True if all incoming edges are resolved and all enabled edges have OK predecessors."""
        incoming_edges = self._incoming_edges
        if not incoming_edges:
            return False

        # All incoming edges must be in terminal states
        if any(e.state == EdgeState.SCHEDULED for e in incoming_edges):
            return False

        # All enabled edges must point to OK predecessors
        for edge in incoming_edges:
            if edge.state == EdgeState.ENABLED:
                predecessor = edge.get_from_task()
                if predecessor.state != TaskState.OK:
                    return False

        return True

    @tool
    def accept(self) -> None:
        """Mark this task as completed successfully."""
        self.state = TaskState.OK

    @tool
    def deny(self) -> None:
        """Deny the task and reject the overall process."""
        self.state = TaskState.DENIED

    @property
    def _cached_inbox(self) -> Path:
        process = self._process
        if process is not None:
            return Path(process.process_dir) / "cached_inbox"
        return Path()

    @tool
    def list_emails(self) -> str:
        """List all cached emails in this task's cached inbox.

        Returns:
            A string listing each cached email with its UID and subject.
        """
        inbox = self._cached_inbox
        if not inbox.is_dir():
            return "No cached emails found."
        emails = list_cached_emails(inbox)
        if not emails:
            return "No cached emails found."
        lines = []
        for uid, dir_path in emails:
            try:
                info = get_cached_email(uid, inbox)
                lines.append(f"UID: {uid}, Subject: {info.subject}")
            except FileNotFoundError:
                lines.append(f"UID: {uid}, Subject: (metadata unavailable)")
        return "\n".join(lines)

    @tool
    def read_email(self, uid: int) -> str:
        """Read the body of a cached email by UID.

        Args:
            uid: The IMAP UID of the cached email.

        Returns:
            The plain text body of the email.
        """
        try:
            info = get_cached_email(uid, self._cached_inbox)
            return info.body_text
        except FileNotFoundError:
            return f"ERROR: Email with UID {uid} not found in cached inbox."

    @tool
    def get_attachments(self, uid: int) -> str:
        """List the attachment filenames for a cached email.

        Args:
            uid: The IMAP UID of the cached email.

        Returns:
            A newline-separated list of attachment filenames.
        """
        try:
            info = get_cached_email(uid, self._cached_inbox)
            if not info.attachments:
                return "No attachments."
            return "\n".join(a.name for a in info.attachments)
        except FileNotFoundError:
            return f"ERROR: Email with UID {uid} not found in cached inbox."

    @tool
    def read_attachment(self, uid: int, filename: str) -> str:
        """Read an attachment from a cached email.

        Args:
            uid: The IMAP UID of the cached email.
            filename: The name of the attachment file.

        Returns:
            The content of the attachment file as a string.
        """
        return f"ERROR: read_attachment is not yet implemented."

    @tool
    def request(self, uid: int, body: str) -> None:
        """Send a reply to the sender of the specified cached email.

        Args:
            uid: The IMAP UID of the email to reply to.
            body: The body text of the reply.
        """
        try:
            info = get_cached_email(uid, self._cached_inbox)
        except FileNotFoundError:
            return
        subject = f"[{self.task_id}] Re: {info.subject}"
        send_email(to=info.from_addr, subject=subject, body=body)

    async def proceed(self, email_uid: int | None = None) -> TaskState:
        """Invoke the agent on this task, evaluate edge conditions, and return the new state.

        Activates the task before invoking the agent, then returns it
        to PENDING if not yet completed or terminated.

        Args:
            email_uid: If provided, the agent processes this email.
                Otherwise, the agent uses the task's existing text.
        """
        # Activate the task before invoking the agent
        self.state = TaskState.ACTIVE
        thread_id = f"{self._process_id}/{self.task_id}" if self._process_id else None

        if email_uid is not None:
            prompt = (
                f"Process the incoming email (IMAP UID {email_uid}). "
                f"Use your tools to read and act on the email content."
            )
        else:
            prompt = self.text

        # 1. Invoke agent — it calls accept(), deny(), or set_text() tools
        await self.invoke_agent(
            prompt=prompt,
            persistent_thread_id=thread_id,
        )

        # 2. If OK, evaluate outgoing edge conditions
        new_state = self.state
        if new_state == TaskState.OK:
            for edge in self._outgoing_edges:
                edge_prompt = f"Is the condition '{edge.condition}' met? Reply with a single word: 'yes' or 'no'."
                result = await self.invoke_agent(
                    prompt=edge_prompt,
                    output_schema=str,
                    persistent_thread_id=thread_id,
                )
                is_met = result == "yes" if result else False
                edge.state = EdgeState.ENABLED if is_met else EdgeState.DISABLED

        # If not done, return to PENDING
        if new_state not in (TaskState.OK, TaskState.DENIED):
            self.state = TaskState.PENDING

        return new_state
