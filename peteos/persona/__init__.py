"""Persona — the behavioural layer: roles, tools, and agents.

A role defines *who* is acting — its identity, capabilities (tools),
behaviour policy, and system prompt.  An agent binds a role to a tool
manager and orchestrates sessions on its behalf.

Classes: ``Agent``, ``Role``, ``RoleManager``, ``Tool``, ``ToolManager``.
"""