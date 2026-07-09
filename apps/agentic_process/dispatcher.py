"""EmailDispatcher - OAP object that receives incoming emails and routes them.

The dispatcher is the entry point for all incoming emails. It is created
once at application startup and persists for the lifetime of the
application.
"""

from __future__ import annotations

import uuid as uuid_mod

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from apps.agentic_process.email_client import fetch_email, send_email
from apps.agentic_process import config


class EmailDispatcher(AgenticObject):
    """You are an email dispatcher.

    Your job is to read the email you received and decide whether it
    references an existing process or requires a new one.

    Use your `get_email_subject` and `get_email_body` tools to read the
    email content and FIND THE PROCESS ID!.

    Then follow these steps:

    1. Extract any process ID (usually a 4-character BASE64 String) from
       the email subject or body. It is most likely in brackets in the subject.
    2. If you found a process ID, call `process_id_found` with that ID.
       The tool will check the process and return a status. Follow the
       tool's instructions in the return value.
    3. ONLY If you could NOT find a process ID, call `process_id_not_found`.
       The tool will create a new process and return a status. Follow
       the tool's instructions in the return value.
    4. If you encounter an error you cannot resolve, call `escalate`.
    """

    def __init__(self, app_main):
        """Initialize with the AppMain instance.

        Args:
            app_main: The AppMain orchestrator instance.
        """
        super().__init__()
        self._uid: int | None = None
        self._dispatch_thread_id: str | None = None
        self._routing_result: str | None = None
        self._last_process_id: str | None = None
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
        """Dispatch an email to the dispatcher agent.

        Sets the UID and thread ID, invokes the agent with a persistent
        session, then resets both on completion. After the agent
        finishes, forwards the email to the process supervisor if needed.

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
        self._routing_result = None
        self._last_process_id = None
        try:
            for attempt in range(max_retries):
                await self.invoke_agent(
                    prompt=(
                        f"You got a new email. Dispatch it. "
                        f"If you need more context, use your tools. "
                        f"If you are sure what to do, take action now."
                    ),
                    persistent_thread_id=self._dispatch_thread_id,
                )
                if self._routing_result is not None:
                    break
        finally:
            self._uid = None
            self._dispatch_thread_id = None

        # Post-processing: forward to supervisor if a process was found
        if self._routing_result in ("active", "new_process"):
            supervisor = self.app_main.get_or_create_supervisor(
                self._last_process_id
            )
            await supervisor.dispatch_email(uid)

    # --- Tools available to the agent running this dispatcher ---

    @tool
    def get_email_subject(self) -> str:
        """Fetch the subject line of the current email."""
        uid = self._get_uid()
        info = fetch_email(uid)
        return info.subject

    @tool
    def get_email_body(self) -> str:
        """Fetch the body text of the current email."""
        uid = self._get_uid()
        info = fetch_email(uid)
        return info.body_text

    @tool
    def process_id_found(self, process_id: str) -> str:
        """Check if the process exists and is active.

        Returns a status string that tells you what to do next.
        Follow the instructions in the return value.

        Args:
            process_id: The process ID extracted from the email.

        Returns:
            A status string with instructions for the next action.
        """
        process = self.app_main.find_process(process_id)
        if process is None:
            self._routing_result = "not_found"
            return (
                f"Process {process_id} was not found in the registry. "
                "Compose an escalation explaining that the referenced "
                "process could not be found."
            )
        elif process.is_terminated():
            self._routing_result = "terminated"
            return (
                f"Process {process_id} is already closed. "
                "Compose a reply informing the user that this process "
                "cannot receive emails anymore."
            )
        else:
            self._routing_result = "active"
            self._last_process_id = process_id
            return (
                f"Process {process_id} is active. "
                "The email will be forwarded to the process supervisor "
                "after your reply. Reply to the user confirming the email "
                "was received and routed."
            )

    @tool
    def process_id_not_found(self) -> str:
        """Create a new process and return its details.

        Returns a status string that tells you what to do next.
        Follow the instructions in the return value.

        Returns:
            A status string with the new process ID and instructions.
        """
        new_id = self.app_main.create_process_from_workflow()
        self._routing_result = "new_process"
        self._last_process_id = new_id
        return (
            f"New process created with ID {new_id}. "
            f"You can stop your work here, the supervisor of the process will take it from here."
        )

    @tool
    def escalate(self, subject: str, body: str) -> None:
        """Forward the email to the escalation address with an error note.

        The subject includes the process ID if one was assigned.

        Args:
            subject: Subject of the escalation email.
            body: Body text of the escalation email.
        """
        full_subject = f"EmailDispatcher: {subject}"
        pid = self._last_process_id
        if pid is not None:
            full_subject = f"{full_subject} [{pid}]"
        send_email(to=config.ESCALATION_EMAIL, subject=full_subject, body=body)
