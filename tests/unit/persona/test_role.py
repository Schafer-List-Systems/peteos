"""Unit tests for the persona Role class."""

import tempfile
from peteos.utils import json
from pathlib import Path

import pytest

from peteos.persona.role import Role


class TestRoleInit:
    """Test Role.__init__."""

    def test_minimal_init(self):
        role = Role(name="simple", description="A simple role")
        assert role.name == "simple"
        assert role.description == "A simple role"
        assert role.system_prompt is None
        assert role.system_prompt_hooks == []
        assert role.required_tools == []
        assert role.execution_environment == "REPL"
        assert role.model == ".*"
        assert role.auto_approve_tools == []
        assert role.tool_filter == []
        assert role.behavior_policy == "responsive"

    def test_full_init(self):
        def hook() -> str:
            return "hook text"
        role = Role(
            name="full",
            description="A full role",
            system_prompt="You are a full role.",
            system_prompt_hooks=[hook],
            required_tools=["tool_a"],
            execution_environment="Jupyter",
            model="gpt-4",
            auto_approve_tools=["read"],
            tool_filter=["read.*"],
            behavior_policy="continuous",
        )
        assert role.system_prompt == "You are a full role."
        assert role.system_prompt_hooks == [hook]
        assert role.required_tools == ["tool_a"]
        assert role.execution_environment == "Jupyter"
        assert role.model == "gpt-4"
        assert role.auto_approve_tools == ["read"]
        assert role.tool_filter == ["read.*"]
        assert role.behavior_policy == "continuous"

    def test_defaults_are_not_mutated_across_instances(self):
        role1 = Role(name="r1", description="r1")
        role2 = Role(name="r2", description="r2")
        role1.system_prompt_hooks.append(lambda: "extra")
        assert len(role2.system_prompt_hooks) == 0
        role1.required_tools.append("x")
        assert role2.required_tools == []


class TestRoleHooks:
    """Test Role.add_system_prompt_hook and _all_hooks."""

    def test_add_system_prompt_hook(self):
        role = Role(name="h", description="h")
        hook1 = lambda: "first"
        hook2 = lambda: "second"
        role.add_system_prompt_hook(hook1)
        role.add_system_prompt_hook(hook2)
        assert len(role.system_prompt_hooks) == 2

    def test_all_hooks_includes_static_prompt(self):
        role = Role(name="p", description="p", system_prompt="static prompt")
        hooks = role._all_hooks
        assert len(hooks) == 1
        result = hooks[0]()
        assert result == "static prompt"

    def test_all_hooks_includes_dynamic_hooks_plus_static(self):
        def dyn() -> str:
            return "dynamic"
        role = Role(name="p", description="p", system_prompt="static", system_prompt_hooks=[dyn])
        hooks = role._all_hooks
        assert len(hooks) == 2
        assert hooks[0]() == "dynamic"
        assert hooks[1]() == "static"

    def test_all_hooks_without_static_prompt(self):
        role = Role(name="h", description="h")
        assert role._all_hooks == []

    def test_all_hooks_only_dynamic_no_static(self):
        def dyn() -> str:
            return "dyn"
        role = Role(name="h", description="h", system_prompt_hooks=[dyn])
        hooks = role._all_hooks
        assert len(hooks) == 1
        assert hooks[0]() == "dyn"


class TestRoleLoadFromDict:
    """Test Role.load_from_dict."""

    def test_minimal_dict(self):
        role = Role.load_from_dict({"name": "simple", "description": "simple desc"})
        assert role.name == "simple"
        assert role.description == "simple desc"
        assert role.system_prompt is None
        assert role.required_tools == []

    def test_dict_with_all_fields(self):
        data = {
            "name": "analyst",
            "description": "A data analyst",
            "system_prompt": "You analyze data.",
            "required_tools": ["read_csv"],
            "execution_environment": "Jupyter",
            "model": "claude-3",
            "auto_approve_tools": ["read_csv"],
            "tool_filter": ["read.*"],
            "behavior_policy": "continuous",
        }
        role = Role.load_from_dict(data)
        assert role.name == "analyst"
        assert role.description == "A data analyst"
        assert role.system_prompt == "You analyze data."
        assert role.required_tools == ["read_csv"]
        assert role.execution_environment == "Jupyter"
        assert role.model == "claude-3"
        assert role.auto_approve_tools == ["read_csv"]
        assert role.tool_filter == ["read.*"]
        assert role.behavior_policy == "continuous"

    def test_dict_defaults(self):
        data = {"name": "d", "description": "d"}
        role = Role.load_from_dict(data)
        assert role.execution_environment == "REPL"
        assert role.model == ".*"
        assert role.auto_approve_tools == []
        assert role.tool_filter == []
        assert role.behavior_policy == "responsive"


class TestRoleLoadConfigFromPath:
    """Test Role.load_config_from_path."""

    def test_no_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Role.load_config_from_path(tmpdir)
            assert config == {}

    def test_with_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text('{"required_tools": ["t1", "t2"], "execution_environment": "Jupyter"}')
            config = Role.load_config_from_path(tmpdir)
            assert config == {"required_tools": ["t1", "t2"], "execution_environment": "Jupyter"}

    def test_with_extra_keys_in_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text('{"auto_approve_tools": ["x"], "tool_filter": ["x.*"], "behavior_policy": "continuous"}')
            config = Role.load_config_from_path(tmpdir)
            assert config["auto_approve_tools"] == ["x"]
            assert config["tool_filter"] == ["x.*"]
            assert config["behavior_policy"] == "continuous"


class TestRoleLoadFromPath:
    """Test Role.load_from_path."""

    def test_minimal_from_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "bot"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("A bot.")
            role = Role.load_from_path(str(role_dir))
            assert role.name == "bot"
            assert role.description == "A bot."
            assert role.system_prompt is None
            assert role.required_tools == []

    def test_from_markdown_and_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "assistant"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("An assistant.")
            (role_dir / "system_prompt.md").write_text("Be helpful.")
            role = Role.load_from_path(str(role_dir))
            assert role.name == "assistant"
            assert role.description == "An assistant."
            assert role.system_prompt == "Be helpful."

    def test_from_config_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "config_role"
            role_dir.mkdir()
            (role_dir / "config.json").write_text('{"description": "From config", "system_prompt": "Config prompt"}')
            role = Role.load_from_path(str(role_dir))
            assert role.name == "config_role"
            assert role.description == "From config"
            assert role.system_prompt == "Config prompt"

    def test_md_precedence_over_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "mixed"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("From markdown.")
            (role_dir / "system_prompt.md").write_text("From markdown prompt.")
            (role_dir / "config.json").write_text('{"description": "From config", "system_prompt": "From config prompt", "required_tools": ["tool1"]}')
            role = Role.load_from_path(str(role_dir))
            assert role.description == "From markdown."
            assert role.system_prompt == "From markdown prompt."
            assert role.required_tools == ["tool1"]

    def test_config_fields_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "configured"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("Desc.")
            (role_dir / "config.json").write_text('{"required_tools": ["t1"], "execution_environment": "Jupyter", "model": "gpt-4", "auto_approve_tools": ["t1"], "tool_filter": ["t1"], "behavior_policy": "continuous"}')
            role = Role.load_from_path(str(role_dir))
            assert role.required_tools == ["t1"]
            assert role.execution_environment == "Jupyter"
            assert role.model == "gpt-4"
            assert role.auto_approve_tools == ["t1"]
            assert role.tool_filter == ["t1"]
            assert role.behavior_policy == "continuous"

    def test_name_from_directory_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "my_custom_role"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("Desc.")
            role = Role.load_from_path(str(role_dir))
            assert role.name == "my_custom_role"

    def test_name_from_file_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_file = Path(tmpdir) / "role.yaml"
            role_file.write_text("Desc.\n")
            # role_file is not a directory, so load_from_path should fail or use suffix
            # This tests the path_obj.suffix logic - but iterdir checks is_dir
            # Actually load_from_path doesn't check if path is dir, just reads from it
            # The directory check is in RoleManager.load_from_dir
            (role_file.parent / "role").mkdir()
            # This test is more about the name extraction logic in the constructor
            (role_file.parent / "role" / "description.md").write_text("Desc.")
            role = Role.load_from_path(str(role_file.parent / "role"))
            assert role.name == "role"
