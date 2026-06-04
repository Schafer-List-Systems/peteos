from __future__ import annotations

from peteos.logger import get_logger
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool
from .edge import Edge

_logger = get_logger(__name__)

from ._types import EdgeState, TaskState, TaskStatus, EdgeEvaluation
from typing import Any
from apps.agentic_process.email_client import (
    get_cached_email,
    list_cached_emails,
)
from pathlib import Path


class Task(AgenticObjectBase):
    """You are a task agent within a user-interaction workflow.
    Your purpose is to handle or execute a task in that human-in-the-loop
    workflow.

    FOLLOW THE INSTRUCTIONS OF YOUR TASK DESCRIPTION THOROUGHLY! READ IT
    USING THE `get_text` TOOL! Make important notes in the task description via
    the tool `append_text` to PERSIST FACTS AND PROVIDE PROGRESS INFORMATION.
    Avoid appending redundant information! This text is also read by your
    supervisor and used to redirect incoming emails towards you. Therefore,
    you must also provide the information about requests in the task
    description!

    When you deny, then the whole process is denied. when you are
    ready for evaluation of the outgoing edges, Call `produce_output`
    with the decision string `ready`. The agentic harness will then
    ask you for an evaluation of each condition of the outgoing edges
    separately. Your evaluation steers the process, conditionally
    activating successive tasks as nodes in the process graph.

    You get the information required to evaluate the conditions from the
    emails. Request information ONLY WHEN THE INFORMATION YOU NEED FOR
    EVALUATION OF THE OUTGOING EDGES IS NOT AVAILABLE. Call `request` AT MOST ONCE
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
        self._request: str | None = None

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
    def append_text(self, value: str) -> str:
        """Append a bullet point to the text of this task. Be concise!

        Returns:
            The updated task text.
        """
        timestamp = self._format_timestamp()
        self.text = self.text + f"\n- {timestamp} {value}"
        return self.text

    @staticmethod
    def _format_timestamp() -> str:
        """Return the current local time as [YYYY-MM-DD HH:MM:SS]."""
        from datetime import datetime
        now = datetime.now()
        return f"[{now.strftime('%d.%m.%Y %H:%M:%S')}]"

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
        return "\n".join(f"- {e.to_task_id}: {e.condition}" for e in self._outgoing_edges)

    @tool
    def set_request(self, body: str) -> str:
        """Set a request for information from the sender of the current email.

        Each call overwrites any previously set request. The subject
        and recipient are derived automatically from the incoming email.
        The request is sent by the supervisor at the end of the dispatch
        cycle if non-empty. Use ``get_request`` to recheck what is
        currently set.

        Args:
            body: The body text of the request.
        """
        self._request = body

        if body == "":
            return "Request has been cleared! USE THE `produce_output` TOOL WITH THE 'decision' FIELD SET TO 'pending' NOW, UNLESS YOU NEED TO ALSO ESCALATE AN ISSUE!"
        else:
            return "Request has been set! USE THE `produce_output` TOOL WITH THE 'decision' FIELD SET TO 'pending' NOW, UNLESS YOU NEED TO ALSO ESCALATE AN ISSUE!"

    @tool
    def get_request(self) -> str:
        """Get the currently set request body, or empty string."""
        if self._request is None:
            return ""
        return self._request

    def _clear_request(self) -> None:
        """Clear the pending request after the supervisor processes it."""
        self._request = None

    def _validate_edge_evaluations(
        self, evaluations: list[Any] | None
    ) -> tuple[bool, str]:
        """Validate that edge evaluations are well-formed.

        Args:
            evaluations: The evaluations returned by the agent.

        Returns:
            A tuple of (is_valid, error_message). The error message is
            non-empty when validation fails so it can be fed back to
            the agent on retry.
        """
        if not evaluations or not isinstance(evaluations, list):
            return False, f"Expected a non-empty list of evaluations, got: {evaluations!r}"
        expected_ids = {e.to_task_id for e in self._outgoing_edges}
        if len(evaluations) != len(expected_ids):
            return (
                False,
                f"Expected {len(expected_ids)} evaluations but got {len(evaluations)}",
            )
        seen_ids: set[str] = set()
        for i, ev in enumerate(evaluations):
            if not isinstance(ev, EdgeEvaluation):
                return False, f"Evaluation at index {i} is not an EdgeEvaluation: {ev!r}"
            edge_id = ev.edge_id
            if edge_id not in expected_ids:
                return False, f"Unknown edge_id '{edge_id}' in evaluation at index {i}"
            if edge_id in seen_ids:
                return False, f"Duplicate edge_id '{edge_id}' in evaluations"
            seen_ids.add(edge_id)
            met = ev.met
            if not isinstance(met, bool):
                return (
                    False,
                    f"Evaluation for edge '{edge_id}' at index {i}: 'met' must be a boolean, got: {met!r}",
                )
        return True, ""

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

        # 4. If OK, evaluate all outgoing edge conditions in one batch
        if new_state == TaskState.OK and self._outgoing_edges:
            edge_list = "\n".join(
                f"- {edge.to_task_id}: {edge.condition}" for edge in self._outgoing_edges
            )
            max_attempts = 3
            evaluations: list[Any] | None = None
            retry_prompt_suffix = ""
            for attempt in range(max_attempts):
                prompt = (
                    f"The task reached a 'ready' state and all its outgoing edges need to be evaluated. "
                    f"The task has the following outgoing edges:\n{edge_list}\n"
                    f"Evaluate each edge condition and use the `produce_output` tool to return a list of evaluations. "
                    f"For each edge produce one evaluation with edge_id and met fields (true/false)."
                    f"{retry_prompt_suffix}"
                )
                evaluations = await self.invoke_agent(
                    prompt=prompt,
                    output_schema=list[EdgeEvaluation],
                    persistent_thread_id=thread_id,
                )
                valid, error_msg = self._validate_edge_evaluations(evaluations)
                if valid:
                    break
                retry_prompt_suffix = (
                    f"\n\nYour previous attempt was invalid:\n{error_msg}\n"
                    f"Please correct the output and try again."
                )
                _logger.warning(
                    "Task %s: invalid edge evaluations (attempt %d/%d), retrying",
                    self.task_id, attempt + 1, max_attempts,
                )

            # Apply valid evaluations to edges
            eval_map: dict[str, bool] = {}
            if evaluations is not None:
                for ev in evaluations:
                    eval_map[ev.edge_id] = ev.met
            for edge in self._outgoing_edges:
                is_met = eval_map.get(edge.to_task_id, False)
                edge.state = EdgeState.ENABLED if is_met else EdgeState.DISABLED

            # 5. Activate ready successor tasks
            self._process._activate_ready_tasks(self)
            # 6. Propagate DISABLED edges downstream
            self._process._propagate_disabled(self.task_id)

        # If not done, return to PENDING
        if new_state not in (TaskState.OK, TaskState.DENIED):
            self.state = TaskState.PENDING

        return new_state
