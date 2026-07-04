# State and Persistence

Agentic objects are the center of persistence in the sOAP paradigm.
Unlike conventional agentic frameworks that rely on external memory stores or databases, the object itself maintains its own state, memory, and agency.

## Internal State

Agentic objects manage their own instance attributes, just like any Python object.
Tools and sandboxed code can read and modify this state directly.

## Session Persistence

Sessions (conversations) between an agentic object and its agent can be persisted to disk as JSON.
Each session is stored in its own directory as a `session.json` file.

### Context Serialization

The conversation history is represented as [`Context`](../advanced/context.md) objects, which are serialized to a `_json_dict` containing:
- `id` — a UUID identifying the context.
- `messages` — a list of serialized `Message` objects.
- `content_map` — a mapping from content hashes to their text.
- `anchor_points` — named references to specific messages.
- `message_sequence` — an integer counter tracking message order.
- `children` — references to forked child contexts.

Each [`Message`](../reference/message.md) is also fully serializable, preserving all content parts including text, images, video, and PDFs.

## Context Forking

Contexts can be forked to create branches of conversation history.
A forked context shares messages with its parent up to the fork point and can diverge independently.
This creates a tree of contexts and their origins, allowing the system to represent parallel conversations, experiments, and hypotheses.

Forks record:
- The origin context ID.
- The parent mutation counter at the time of the fork.
- Selected anchor messages from the parent.

## Implications for Debugging and Improvement

The serialization and forking architecture is designed with debugging and reproducibility in mind:

- **Reproducible outputs:** Because the full conversation context can be serialized, it is possible to reproduce particular agent outputs by re-invooking the agent with the same context.
- **Debugging utilities:** The ability to understand exactly what the materialized chat history was when an LLM produced a particular response enables the development of debugging tools that inspect agent reasoning.
- **Prompt tuning:** Reproducing outputs allows developers to iterate on system prompts and tool descriptions, measuring improvements quantitatively.
- **Auto-tuning:** The structural foundation supports future tools that could automatically tune agentic systems by comparing forked context outputs.

> **Note:** Some of these features are not yet fully implemented. The serialization infrastructure is in place with these use cases in mind.

## Token Counting

Every message in a context tracks its token count using the `tiktoken` library with the `cl100k_base` encoding (compatible with GPT-4 and later models).
Token counting is used to determine when a context has grown too large and needs compaction or truncation.

### Caching

Token counts are cached on each `Message` object.
Once computed, the count is stored in the message's metadata and returned on subsequent calls without re-computation.
This avoids the relatively expensive `tiktoken` encoding call on every context size check.

When a message's content is dynamically modified (e.g., thinking content removed via strip), the cached count may be invalidated and recomputed.
In most cases, however, token counts remain stable across a message's lifetime.

### Compaction Strategies

The token count is the basis for all context compaction decisions.
Common strategies include:

- **Rolling token window:** Keep only the last N tokens of conversation history, discarding older messages. This is the simplest strategy and ensures the context stays within a predictable size.
- **Rolling sequence window:** Keep only the last N messages, regardless of token count. Useful when message count matters more than token budget.
- **Tool-based compaction:** Summarize older messages by invoking an agent to produce a condensed version, preserving the semantic content while reducing token usage.

The Log Analyzer app demonstrates a rolling token window hook: before each call to the chatbot, it checks the total token count and, if it exceeds a configurable threshold (default 60,000 tokens), compacts the session by keeping only the most recent half of the budget.

### API

`Context.total_token_count(encoding)` returns the sum of all message token counts.
Each `Message.count_tokens(encoding)` returns the cached count for that message.
Both accept an optional `encoding` parameter (default: `cl100k_base`).
