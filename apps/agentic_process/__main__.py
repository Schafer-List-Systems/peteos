#!/usr/bin/env python3
"""Agentic Process Engine — runtime entry point.

Loads a workflow definition, connects to an LLM backend,
and drives the process engine loop.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python -m apps.agentic_process \
        workflow.canvas http://localhost:8080
"""

import asyncio
import sys

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase

from .process import Process
from .workflow import Workflow


def test_send_email():
    """A small manual test for sending a mail."""
    from .email import send_email

    send_email(
        to="info@aios.tools",
        subject="Agentic Process Test Email",
        body="This is a test email from the agentic process engine.",
    )


async def main() -> None:
    test_send_email()
    return

    """Set up backend, load workflow, and run engine loop."""
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <workflow-canvas> <backend-url>")
        sys.exit(1)

    workflow_path, backend_url = sys.argv[1], sys.argv[2]

    # --- Connect to LLM backend ---
    chatbot_manager = ChatBotManager(timeout=60)
    await chatbot_manager.add_backend("local", backend_url)

    # --- Load workflow and create process ---
    workflow = Workflow(workflow_path)
    process = workflow.create_process()
    process.start()

    # --- Engine loop ---
    while process._active_tasks:
        more = await process.update()
        if not more:
            break

    print("Process complete")


if __name__ == "__main__":
    asyncio.run(main())
