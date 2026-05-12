import tiktoken

def count_tiktokens_per_message():
    """
    Counts tiktokens for each message in the chat history stored in ACTIVE_CHAT_HISTORY.
    
    Returns:
        list: A list of integers where each integer is the token count for the corresponding message.
    """
    # Check if ACTIVE_CHAT_HISTORY is available
    if "ACTIVE_CHAT_HISTORY" not in globals():
        print("Error: ACTIVE_CHAT_HISTORY not found in global namespace.")
        return []

    # Initialize tokenizer (cl100k_base is standard for GPT-3.5 and GPT-4)
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    history = globals()["ACTIVE_CHAT_HISTORY"]
    token_counts = []

    for msg in history.messages:
        # Use Message.serialize_content() for consistent serialization
        parts_text = []
        for item in msg.serialize_content():
            try:
                ptype = item.get("type", "")
                if ptype == "text":
                    parts_text.append(item.get("text", ""))
                elif ptype == "reasoning":
                    parts_text.append(item.get("reasoning", ""))
                elif ptype == "tool_result":
                    parts_text.append(item.get("content", ""))
                else:
                    parts_text.append(str(item.get(ptype, item)))
            except Exception:
                parts_text.append(str(item))

        # Format: "role: content" is standard for context window counting
        full_text = f"{msg.get_role()}: {' '.join(parts_text)}"

        tokens = enc.encode(full_text)
        token_counts.append(len(tokens))

    return token_counts

def reduce_history_to_token_limit(limit: int = 100000):
    """
    Reduces the chat history stored in ACTIVE_CHAT_HISTORY by removing messages from the beginning
    until the sum of tiktokens is below the specified limit.
    
    This function modifies the history.messages list in-place.
    
    Args:
        limit (int): The maximum number of tokens allowed (default: 100,000).
        
    Returns:
        int: The number of messages removed from the history.
    """
    # Check if ACTIVE_CHAT_HISTORY is available
    if "ACTIVE_CHAT_HISTORY" not in globals():
        print("Error: ACTIVE_CHAT_HISTORY not found in global namespace.")
        return 0

    history = globals()["ACTIVE_CHAT_HISTORY"]
    
    # Get token counts for all messages
    token_counts = count_tiktokens_per_message()
    
    if not token_counts:
        return 0

    messages = history.messages
    total_tokens = sum(token_counts)

    if total_tokens <= limit:
        return total_tokens

    # Drop messages from the beginning
    start_index = 0
    current_sum = total_tokens

    while current_sum > limit and start_index < len(messages):
        current_sum -= token_counts[start_index]
        start_index += 1
        
    # Modify the history's messages list in place
    del messages[:start_index]
    
    return current_sum
