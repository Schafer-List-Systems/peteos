"""Conversation — the data model for sessions, contexts, and messages.

A session contains one or more contexts, each representing a single step
or turn in the interaction.  A context is a list of serialised messages
(``ContentPart`` objects) with support for forked branches, content-addressed
storage, and hook-based materialisation of dynamic content.

Classes: ``ContentPart``, ``Message``, ``MessageRegistry``, ``Context``,
``Session``, ``SystemPromptMessage``, ``ToolDefinitionsMessage``.
"""