"""pq — Quick model inquiry CLI.

Lists all available models from configured LLM backends.
Auto-configures on import from standard peteos.json locations.
"""

import asyncio
import sys

import peteos  # triggers auto-configuration
from peteos.chatbot import ChatBotManager


def main() -> None:
    args = sys.argv[1:]

    if "--list-models" in args or "-lm" in args:
        bots = ChatBotManager.list_chatbots(".*")
        for model_id, chatbot in bots:
            print(f"{model_id} ({chatbot.priority})")
        return

    print("Usage: pq [--list-models | -lm]", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
