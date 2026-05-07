import gc
from uuid import UUID

# Global dictionary to store chatbot references
chatbot_store = {}

def find_and_store_chatbot(session_uuid_str):
    """
    Locates the Agent instance, finds the specific session by UUID,
    retrieves its chatbot_manager, and stores it in the global dictionary.
    """
    try:
        # Find the Agent instance
        agent_instance = None
        for obj in gc.get_objects():
            if type(obj).__name__ == 'Agent':
                agent_instance = obj
                break

        if agent_instance is None:
            return f"Agent instance not found."

        # Find the session
        target_id = UUID(session_uuid_str)
        active_session = None
        session_key = None

        for k, v in agent_instance._sessions.items():
            if k == target_id:
                active_session = v
                session_key = k
                break

        if active_session is None:
            return f"Session {session_uuid_str} not found."

        # Retrieve the chatbot manager
        chatbot_manager = getattr(active_session, 'chatbot_manager', None)

        if chatbot_manager:
            chatbot_store[session_key] = chatbot_manager
            return f"Success: Stored {type(chatbot_manager).__name__} under key {session_key}."
        else:
            return "No 'chatbot_manager' found in the session."

    except Exception as e:
        return f"Error: {e}"


def get_current_session_id():
    """
    Finds the InteractiveShellChannel instance and retrieves the UUID of the active session.
    This is useful when the session ID is not known.
    """
    try:
        # Search for the InteractiveShellChannel
        for obj in gc.get_objects():
            if type(obj).__name__ == 'InteractiveShellChannel':
                channel = obj
                # Get the _active_session_uuid attribute
                active_uuid = getattr(channel, '_active_session_uuid', None)
                
                if active_uuid is not None:
                    return str(active_uuid)
        
        return "No active session found in InteractiveShellChannel."
    except Exception as e:
        return f"Error: {e}"
