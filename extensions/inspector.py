import gc
from uuid import UUID

from peteos.chatbot import ChatBotManager

# Global dictionary to store chatbot references
chatbot_store = {}

def find_and_store_chatbot(session_uuid_str):
    """
    Retrieves the ChatBotManager from its class-level state and stores it
    in the global dictionary.
    """
    try:
        manager = ChatBotManager
        if not manager._backends:
            return "No ChatBotManager backends configured."
        chatbot_store[session_uuid_str] = manager
        return f"Success: Stored {type(manager).__name__} under key {session_uuid_str}."

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
