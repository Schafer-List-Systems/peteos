from peteos.channels.channel import Channel

def get_active_session_info():
    """
    Retrieves the active session UUID and chat history.
    """
    # 1. Retrieve the running channel instance from the global registry.
    # The key "nextcloud" corresponds to the `name` argument passed in the example script.
    nextcloud_channel = Channel._registry.get("nextcloud")
    
    if not nextcloud_channel:
        return None, "Channel 'nextcloud' not found in registry."
    
    # 2. Get the active session UUID from the channel.
    # The channel tracks the session currently being processed.
    session_uuid = nextcloud_channel._active_session_uuid
    
    if not session_uuid:
        return None, "No active session (bot is idle)."

    # 3. Retrieve the Session object from the Agent.
    agent = nextcloud_channel._agent
    session = agent.get_session(session_uuid)
    
    if not session:
        return None, "Session object not found in Agent."
    
    # 4. Get the chat history.
    chat_history = session.chat_history
    
    return session_uuid, chat_history
