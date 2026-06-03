#!/usr/bin/env python3
"""Agentic Process Engine — runtime entry point.

Watches an IMAP mailbox for incoming emails and routes them through
the dispatcher OAP into agentic processes.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos \
      python apps/agentic_process/__main__.py <workflow-canvas> <backend-url>
"""

import asyncio
import sys
import time

from apps.agentic_process.app_main import AppMain
from peteos.chatbot.manager import ChatBotManager


async def main() -> None:
    """Watch for incoming emails and route them via the dispatcher."""
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <workflow-canvas> <backend-url>")
        sys.exit(1)

    workflow_path, backend_url = sys.argv[1], sys.argv[2]

    # --- Connect to LLM backend ---
    chatbot_manager = ChatBotManager(timeout=60)
    await chatbot_manager.add_backend("local", backend_url)

    app_main = AppMain(workflow_path=workflow_path)

    print("Agentic process engine started. Waiting for emails...")

    while True:
        try:
            uids = app_main.poll_new_emails()
        except Exception as e:
            print(f"IMAP error: {e}", file=sys.stderr)
            uids = []

        for uid in uids:
            try:
                await app_main.dispatcher.dispatch_email(uid)
            except Exception as e:
                print(f"Dispatcher error for UID {uid}: {e}", file=sys.stderr)

        time.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
