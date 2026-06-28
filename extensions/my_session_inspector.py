from peteos.engine.channel import Channel

def get_active_session_info():
    """
    Retrieves the active session UUID and chat history.
    """
    # 1. Retrieve the running channel instance from the global registry.
    nextcloud_channel = Channel._registry.get("nextcloud")

    if not nextcloud_channel:
        return None, "Channel 'nextcloud' not found in registry."

    # 2. Get the session UUID from the channel.
    session_uuid = nextcloud_channel._session_uuid

    if not session_uuid:
        return None, "No active session (bot is idle)."

    # 3. Get the chat history from the runner's session.
    runner = nextcloud_channel._runner
    session = runner.session

    if not session:
        return None, "Session object not found."

    # 4. Get the chat history.
    chat_history = session.active_context

    return session_uuid, chat_history
