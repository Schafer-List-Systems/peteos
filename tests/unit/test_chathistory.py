from peteos.chatbot import ChatHistory
from peteos.chatbot import Message


def test_chat_history_initialization():
    """Test that ChatHistory initializes with empty messages list."""
    history = ChatHistory()
    assert history.messages == []


def test_chat_history_append_message():
    """Test appending a message to chat history."""
    history = ChatHistory()
    message = Message(content={"text": "Hello"}, creation_timestamp=None)

    history.append_message(message)

    assert len(history.messages) == 1
    assert history.messages[0] is message


def test_chat_history_get_content():
    """Test getting content from chat history."""
    history = ChatHistory()

    msg1 = Message(content={"role": "user", "text": "Hi"}, creation_timestamp=None)
    msg2 = Message(content={"role": "assistant", "text": "Hello"}, creation_timestamp=None)
    msg3 = Message(content={"role": "user", "text": "How are you?"}, creation_timestamp=None)

    history.append_message(msg1)
    history.append_message(msg2)
    history.append_message(msg3)

    contents = history.get_content()

    assert len(contents) == 3
    assert contents[0] == {"role": "user", "text": "Hi"}
    assert contents[1] == {"role": "assistant", "text": "Hello"}
    assert contents[2] == {"role": "user", "text": "How are you?"}


def test_chat_history_get_content_empty():
    """Test getting content from empty chat history."""
    history = ChatHistory()
    contents = history.get_content()

    assert contents == []
