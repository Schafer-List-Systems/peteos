from __future__ import annotations

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool

from ._types import EdgeState, TaskState, TaskStatus
from apps.agentic_process.email_client import (
    get_cached_email,
    list_cached_emails,
    send_email,
)
from pathlib import Path


class Task(AgenticObjectBase):
    """You are a task agent within an agentic process engine. Your
    purpose is to gather the information in order to evaluate the
    conditions of all outgoing edges regarding the text of this task.

    Read your task text using `get_text` tool. During the conversation,
    update the task text via `append_text` to persist facts and provide
    progress information. Avoid appending redundant information! This
    text is also read by your supervisor and used to redirect incoming
    emails towards you if there is relevant information for you.
    That's why you also need to provide the information about requests
    in the task text.

    When you deny, then the whole process is denied. when you are
    ready for evaluation of the outgoing edges, Call `produce_output`
    with the decision string `ready`. The agentic harness will then
    ask you for an evaluation of each condition separately. Your
    evaluation steers the process, conditionally activating successive
    tasks as nodes in the process graph.

    You get the information required to evaluate the conditions from the
    emails. If the information you need for evaluation is not
    available, you can request it vug now you introduced a bug there ia email. Call `request` AT MOST ONCE
    per incoming email to not spam the user! If your purpose cannot be
    satisfied even after requesting further information, then you deny
    this task and therefore the whole process you are part of.
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
            if value == TaskState.ACTIVE and self.task_id not in process._active_tasks:
                process._active_tasks.append(self.task_id)
            elif value != TaskState.ACTIVE and self.task_id in process._active_tasks:
                process._active_tasks.remove(self.task_id)
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
    def append_text(self, value: str) -> None:
        """Append a bullet point to the text of this task. Be concise!"""
        self.text = self.text + "\n- " + value

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

    def activate(self) -> None:
        """Transition this task to the ACTIVE state."""
        self.state = TaskState.ACTIVE

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
    def get_outgoing_conditions(self) -> str:
        """Get the condition strings of all outgoing edges.

        Returns:
            A newline-separated list of edge conditions.
        """
        if not self._outgoing_edges:
            return "No outgoing edges."
        return "\n".join(f"- {e.condition}" for e in self._outgoing_edges)

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

        Activates the task before invoking the agent, then forces a
        structured response (ready, deny, or pending) via produce_output.

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

        status = await self.invoke_agent(
            prompt = (
                f"Process the incoming email (IMAP UID {email_uid}). "
                f"Use your tools to read and act on the email content."
                f"When you have analyzed all relevant information, produce one word as output:\n"
                f"- ready: you have everything to evaluate edge conditions\n"
                f"- deny: you reject the task and the overall process\n"
                f"- pending: you need more information that you already requested via request()"
            ),
            output_schema=TaskStatus,
            persistent_thread_id=thread_id,
        )

        decision = status.decision

        # 3. Set task state based on decision
        if decision == "deny":
            new_state = TaskState.DENIED
        elif decision == "ready":
            new_state = TaskState.OK
        else:
            new_state = TaskState.PENDING
        self.state = new_state

        # 4. If OK, evaluate outgoing edge conditions
        if new_state == TaskState.OK:
            for edge in self._outgoing_edges:
                edge_decision = None
                max_attempts = 3
                for attempt in range(max_attempts):
                    result = await self.invoke_agent(
                        prompt=(
                            f"An outgoing edge of the task has the condition "
                            f"label '{edge.condition}'. Use the tool `produce_output` with a "
                            f"single 'yes' or 'no' as decision string depending on whether the "
                            f"process should go along that edge or not."
                        ),
                        output_schema=TaskStatus,
                        persistent_thread_id=thread_id,
                    )
                    edge_decision = result.decision
                    if edge_decision in ("yes", "no"):
                        break
                    _logger.warning(
                        "Task %s: edge '%s' decision was %r (attempt %d/%d), retrying",
                        self.task_id, edge.condition, edge_decision,
                        attempt + 1, max_attempts,
                    )

                is_met = (edge_decision == "yes")
                edge.state = EdgeState.ENABLED if is_met else EdgeState.DISABLED

            # 5. Activate ready successor tasks
            self._process._activate_ready_tasks(self)
            # 6. Propagate DISABLED edges downstream
            self._process._propagate_disabled(self.task_id)

        # If not done, return to PENDING
        if new_state not in (TaskState.OK, TaskState.DENIED):
            self.state = TaskState.PENDING

        return new_state
