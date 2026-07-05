# RoleManager

A class-method-only manager for registering and loading roles.

## `register_role`

```python
RoleManager.register_role(role)
```

Registers a role by name.

## `load_from_dir`

```python
RoleManager.load_from_dir(directory) → list[str]
```

Loads all roles from a directory where each subdirectory is a role name. Returns a list of successfully loaded role names.

Each subdirectory may contain a `description.md` and/or `system_prompt.md` along with an optional `config.json`:

```
my-role/
  description.md          # Optional
  system_prompt.md        # Optional
  config.json             # Optional
```

**Raises:** `FileNotFoundError` if the directory does not exist.

## `get_role`

```python
RoleManager.get_role(name) → Role | None
```

Looks up a role by name. Returns `None` if not found.

## `list_roles`

```python
RoleManager.list_roles() → list[tuple[str, Role]]
```

Lists all registered roles. Returns a list of `(name, Role)` tuples.
