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

from . import utils as _utils


class Task(AgenticObjectBase):
    """You are a task agent within a user-interactiorkflow
    assisting the process supervisor (user).

    Your purpose is to handle or accustom a client on a SINGLE
    TASK in that process. Your job is to make sure, that all
    conditions in your task description are met by the client. If
    necessary information is missing to meet the task's requirements,
    and it is not available in the E-Mail inbox, request it from the
    client by calling `set_feedback` with the request.

    - INITIALLY, READ THE TASK INSTRUCTION USING THE `get_text` TOOL!
    - FOLLOW THE INSTRUCTIONS OF THE TASK PRECISELY!
    - RETURN VALUES OF TOOLS MAY GIVE YOU INSTRUCTIONS TOO, WHICH YOU SHOULD CONSIDER!
    . MAKE NOTES OF ALL PROGRESS IN THE TASK DESCRIPTION VIA THE TOOL `append_text!
    - Avoid appending redundant information!
    - USE THE `set_feedback` TOOL FOR PROVIDING RELEVANT INFORMATION TO THE SUPERVISOR AND REQUESTING INFORMATION FROM THE CLIENT!

    - The client's E-Mails are available in the inbox.
    - The E-Mail inbox can be listed using the `list_inbox` tool.
    - The first E-Mail in that inbox is sent by the client.
    - NOT ALL E-MAILS IN THE INBOX ARE NECESSARILY FROM THE CLIENT! E-MAILS FROM OTHER SENDERS THAN THE CLIENT ARE ONLY RELEVANT IF YOUR TASK DESCRIPTION EXPLICITELY STATES THAT!
    - You can read particular E-Mails calling the `read_email` tool.
    - E-Mails can have attachments (images, texts, PDFs) that you can list via `list_attachments` and `read_attachment`.
ome on, man.
    - When you got all information, such that the task is met, accept the task.
    - When the task can NEVER be met because the client is NOT ABLE to provide the necessary information, deny the task (and therefore the process).
      bE CAREFUL with denial, as this is terminal for the process! In doubt, request for more information and go into `pending` state!
    - USE tHE `produce_output` TOOL TO PROVIDE OUTPUT REQUESTED BY THE SUPERVISOR.

    If you are instructed to evaluate the outgoing edges, then
    decide for each edge if its condition is met regarding your
    task and the client's information. Your evaluation steers the
    process, conditionally activating successive tasks as nodes
    in the process graph.
    """

    def __init__(self, node_data: dict, process_id: str | None = None) -> None:
        super().__init__()
        self._node_data = node_data
        self._outgoing_edges: list[Edge] = []
        self._incoming_edges: list[Edge] = []
        self._process_id = process_id
        self._process: object | None = None
        self._feedback: str | None = None

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
            if process._auto_flush:
                process.store()

    @property
    def text(self) -> str:
        return self._node_data.get("text", "")

    @text.setter
    def text(self, value: str) -> None:
        self._node_data["text"] = value
        process = self._process
        if process is not None:
            if process._auto_flush:
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
        timestamp = _utils.format_timestamp()
        self.text = self.text + f"\n- {timestamp} {value}"
        return f"New text is now:\n{self.text}"

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
    def list_inbox(self) -> str:
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
        """Read the sender, subject and body of an email by UID.

        Args:
            uid: The IMAP UID of the cached email.

        Returns:
            The plain "from" address, subject and text body of the email.
        """
        try:
            info = get_cached_email(uid, self._cached_inbox)
            return (
                f"sender: {info.from_addr}\n"
                f"subject: {info.subject}\n"
                f"text:\n{info.body_text}"
            )
        except FileNotFoundError:
            return f"ERROR: Email with UID {uid} not found in cached inbox."

    @tool
    def list_attachments(self, uid: int) -> str:
        """List the attachment filenames for a one particular E-Mail.

        Args:
            uid: The IMAP UID of the E-Mail.

        Returns:
            A newline-separated list of attachment filenames.
        """
        try:
            info = get_cached_email(uid, self._cached_inbox)
            if not info.attachments:
                return "E-Mail {uid} has no attachments."
            return f"E-Mail {uid} has the following attachmentes:\n" + (
                "\n".join(a.name for a in info.attachments)
            )
        except FileNotFoundError:
            return f"ERROR: Email with UID {uid} not found in cached inbox."

    @tool
    async def read_attachment(self, uid: int, filename: str, session: "Session | None" = None) -> str:
        """Read an attachment from a cached email.

        For text files (.txt, .csv, .md, .json, .xml, .html) returns the
        content directly. For PDFs extracts text and converts pages to
        images for the agent to review. For images queues them to the
        session so the agent can review them via the media ContentPart flow.

        Args:
            uid: The IMAP UID of the cached email.
            filename: The name of the attachment file.
            session: The session (injected by the execution environment).

        Returns:
            File content for text files, or a status message for media.
        """
        import mimetypes

        safe_name = Path(filename).name
        if safe_name != filename or not safe_name:
            return f"ERROR: Invalid filename '{filename}'."

        try:
            get_cached_email(uid, self._cached_inbox)
        except FileNotFoundError:
            return f"ERROR: Email with UID {uid} not found in cached inbox."

        attachment_path = self._cached_inbox / str(uid) / safe_name
        if not attachment_path.is_file():
            return f"ERROR: Attachment '{safe_name}' not found for email UID {uid}."

        mime_type, _ = mimetypes.guess_type(safe_name)
        if mime_type is None:
            return f"ERROR: Unknown file type for '{safe_name}'."

        # Text files: return content directly
        if mime_type.startswith("text/") or mime_type == "application/json":
            try:
                return attachment_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"ERROR: '{safe_name}' is not a valid text file."

        # PDFs: extract text and convert pages to images
        if mime_type == "application/pdf":
            return await self._read_pdf(attachment_path, session=session)

        # Images: queue via _read_media
        if mime_type.startswith("image/"):
            if session is None:
                return "ERROR: Session not available."
            try:
                await self._read_media(str(attachment_path.resolve()), session=session)
                return f"OK: '{safe_name}' loaded for review."
            except Exception as e:
                return f"ERROR: Failed to load '{safe_name}': {e}"

        return f"ERROR: '{safe_name}' (type: {mime_type}) cannot be read."

    async def _read_pdf(
        self,
        pdf_path: Path,
        session: "Session | None" = None,  # session not used — PdfTranscriber handles everything
    ) -> str:
        """Transcribe a PDF using tesseract + LLM vision via PdfTranscriber.

        Caches the transcription at ``{pdf_path}.md`` so repeated reads
        don't re-transcribe the same file.

        Args:
            pdf_path: Path to the PDF file.
            session: Unused (kept for API compatibility).

        Returns:
            A formatted summary of page transcriptions and image descriptions.
        """
        cache_path = Path(str(pdf_path) + ".md")

        try:
            cached = cache_path.read_text(encoding="utf-8")
            return cached
        except FileNotFoundError:
            pass

        from apps.agentic_process.pdf_transcriber import PdfTranscriber

        try:
            results = await PdfTranscriber().transcribe(str(pdf_path))
        except FileNotFoundError:
            return f"ERROR: PDF '{pdf_path.name}' not found."
        except Exception as e:
            return f"ERROR: Failed to transcribe '{pdf_path.name}': {e}"

        if not results:
            return f"OK: '{pdf_path.name}' — no pages to transcribe."

        lines = []
        for i, (text, img_desc) in enumerate(results, 1):
            lines.append(f"--- Page {i} ---")
            if text:
                lines.append(text)
            if img_desc:
                lines.append(f"[Visual elements: {img_desc}]")
        output = "\n".join(lines)

        try:
            cache_path.write_text(output, encoding="utf-8")
        except OSError:
            pass  # Cache write failure must not break the caller

        return output

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
    def set_feedback(self, body: str) -> str:
        """Set feedback to the supervisor (e.g., results, status, or a request for more information from the client).

        Each call overwrites any previously set feedback. Use this to
        inform the supervisor so it can plan its next action — reply
        to the user, escalate, or request more information.

        Use ``get_feedback`` to recheck what is currently set.

        Args:
            body: The body text of the feedback.
        """
        self._feedback = body

        if body == "":
            return "Feedback has been cleared! USE THE `produce_output` TOOL WITH THE 'decision' FIELD SET TO 'pending' NOW, UNLESS YOU NEED TO ALSO ESCALATE AN ISSUE!"
        else:
            return "Feedback has been set! USE THE `produce_output` TOOL WITH THE 'decision' FIELD SET TO 'pending' NOW, UNLESS YOU NEED TO ALSO ESCALATE AN ISSUE!"

    @tool
    def get_feedback(self) -> str:
        """Get the currently set feedback for the supervisor, or empty string."""
        if self._feedback is None:
            return ""
        return self._feedback

    def _clear_feedback(self) -> None:
        """Clear the pending feedback after the supervisor processes it."""
        self._feedback = None

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
                f"Process the incoming E-Mail with UID {email_uid} (and older E-Mails if necessary) according your task description. "
                f"IF THAT E-MAIL IS IRRELEVANT FOR YOUR TASK, CHECK IF OLDER E-MAILS ARE RELEVANT! "
                f"Use your tools to read and act on the email content."
                f"If NEITHER that E-Mail NOR any other E-Mail is relevant for your task, then call `set_feedback` and `append_text` mentioning what you need!\n"
                f"WHEN YOU HAVE ANALYZED ALL RELEVANT INFORMATION, USE THE `produce_output` TOOL and one of the following arguments:\n"
                f"- ready: you have everything to evaluate edge conditions\n"
                f"- deny: you reject the task and the overall process\n"
                f"- pending: You used `set_feedback()` and you need more information from the user via the supervisor OR the E-Mail is irrelevant for your task."
            )
        else:
            prompt = (
                f"Proceed executing your task! "
                f"WHEN YOU HAVE ANALYZED ALL RELEVANT INFORMATION, USE THE `produce_output` TOOL and one of the following arguments:\n"
                f"- ready: you have everything to evaluate edge conditions\n"
                f"- deny: you reject the task and the overall process\n"
                f"- pending: You used `set_feedback()` and you need more information from the user via the supervisor OR the E-Mail is irrelevant for your task."
            )

        # Capture text before invoking; loop until agent calls append_text
        text_before = self.text
        for attempt in range(3):
            reminder = "" if attempt == 0 else f" Reminder: your text was not updated or you provided no feedback. You have one last chance to do that via `append_text` and `set_feedback`. THEN PRODUCE OUTPUT VIA THE `produce_output` TOOL!"
            status = await self.invoke_agent(
                prompt=prompt + reminder,
                output_schema=TaskStatus,
                persistent_thread_id=thread_id,
            )
            if self.text != text_before and self._feedback is not None:
                break

        decision = status.decision

        # 3a. If the agent provided feedback, append it to the task text
        if self._feedback is not None:
            self.text = self.text + f"\n- [FEEDBACK] {self._feedback}"

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
            # If no outgoing edges have a label, enable all of them without
            # invoking the agent — the agent is only needed to evaluate
            # labelled (conditional) edges.
            has_labelled_edges = any(e.condition for e in self._outgoing_edges)

            if not has_labelled_edges:
                for edge in self._outgoing_edges:
                    edge.state = EdgeState.ENABLED
            else:
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

                # Apply valid evaluations to edges — non-labeled edges are
                # always ENABLED regardless of agent output.
                eval_map: dict[str, bool] = {}
                if evaluations is not None:
                    for ev in evaluations:
                        eval_map[ev.edge_id] = ev.met
                for edge in self._outgoing_edges:
                    if edge.condition:
                        is_met = eval_map.get(edge.to_task_id, True)
                    else:
                        is_met = True
                    edge.state = EdgeState.ENABLED if is_met else EdgeState.DISABLED

            # 5. Activate ready successor tasks
            self._process._activate_ready_tasks(self)
            # 6. Propagate DISABLED edges downstream
            self._process._propagate_disabled(self.task_id)

        # If not done, return to PENDING
        if new_state not in (TaskState.OK, TaskState.DENIED):
            self.state = TaskState.PENDING

        return new_state
