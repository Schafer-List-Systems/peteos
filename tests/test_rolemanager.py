"""Unit tests for RoleManager."""

import pytest
from pathlib import Path
import tempfile

from peteos.role import Role
from peteos.rolemanager import RoleManager


class TestRoleManagerInit:
    """Test RoleManager initialization."""

    def test_init(self):
        """Test basic initialization with empty registry."""
        manager = RoleManager()
        roles = manager.list_roles()
        assert roles == []


class TestRoleManagerRegisterRole:
    """Test register_role method."""

    def test_register_single_role(self):
        """Test registering a single role."""
        manager = RoleManager()
        role = Role(name="assistant", description="Helpful assistant")
        manager.register_role(role)

        roles = manager.list_roles()
        assert len(roles) == 1
        assert roles[0][0] == "assistant"
        assert roles[0][1].description == "Helpful assistant"

    def test_register_multiple_roles(self):
        """Test registering multiple roles."""
        manager = RoleManager()
        manager.register_role(Role(name="coder", description="Code assistant"))
        manager.register_role(Role(name="writer", description="Writing assistant"))
        manager.register_role(Role(name="analyst", description="Data analyst"))

        roles = manager.list_roles()
        assert len(roles) == 3

    def test_register_duplicate_overwrites(self):
        """Test that registering same role name overwrites."""
        manager = RoleManager()
        manager.register_role(Role(name="test", description="First"))
        manager.register_role(Role(name="test", description="Second"))

        role = manager.get_role("test")
        assert role.description == "Second"


class TestRoleManagerGetRole:
    """Test get_role method."""

    def test_get_existing_role(self):
        """Test getting an existing role."""
        manager = RoleManager()
        role = Role(name="assistant", description="Helpful")
        manager.register_role(role)

        retrieved = manager.get_role("assistant")
        assert retrieved is not None
        assert retrieved.name == "assistant"
        assert retrieved.description == "Helpful"

    def test_get_nonexistent_role(self):
        """Test getting a non-existent role returns None."""
        manager = RoleManager()
        role = manager.get_role("nonexistent")
        assert role is None


class TestRoleManagerLoadFromDir:
    """Test load_from_dir method."""

    def test_load_from_directory(self):
        """Test loading roles from directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create role directories
            assistant_dir = Path(tmpdir) / "assistant"
            assistant_dir.mkdir()
            (assistant_dir / "description.md").write_text("Helpful assistant")

            coder_dir = Path(tmpdir) / "coder"
            coder_dir.mkdir()
            (coder_dir / "description.md").write_text("Code assistant")

            manager = RoleManager()
            loaded = manager.load_from_dir(tmpdir)

            assert len(loaded) == 2
            assert "assistant" in loaded
            assert "coder" in loaded

            # Verify roles registered correctly
            assert manager.get_role("assistant").description == "Helpful assistant"
            assert manager.get_role("coder").description == "Code assistant"

    def test_load_from_directory_with_system_prompt(self):
        """Test loading roles with system prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "test_role"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("Test role")
            (role_dir / "system_prompt.md").write_text("You are a test role.")

            manager = RoleManager()
            manager.load_from_dir(tmpdir)

            role = manager.get_role("test_role")
            assert role.system_prompt == "You are a test role."

    def test_load_from_directory_with_config(self):
        """Test loading roles with config.json."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "configured_role"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("Configured role")

            config = {
                "required_tools": ["tool1", "tool2"],
                "execution_environment": "REPL",
                "model": "model.*"
            }
            (role_dir / "config.json").write_text(json.dumps(config))

            manager = RoleManager()
            manager.load_from_dir(tmpdir)

            role = manager.get_role("configured_role")
            assert role.required_tools == ["tool1", "tool2"]
            assert role.execution_environment == "REPL"
            assert role.model == "model.*"

    def test_load_from_directory_nonexistent(self):
        """Test loading from non-existent directory raises error."""
        manager = RoleManager()
        with pytest.raises(FileNotFoundError):
            manager.load_from_dir("/nonexistent/directory/path")

    def test_load_from_directory_skips_invalid_roles(self):
        """Test that invalid roles are skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid role
            valid_dir = Path(tmpdir) / "valid"
            valid_dir.mkdir()
            (valid_dir / "description.md").write_text("Valid role")

            # Create an invalid directory (no description)
            invalid_dir = Path(tmpdir) / "invalid"
            invalid_dir.mkdir()

            manager = RoleManager()
            loaded = manager.load_from_dir(tmpdir)

            # Should only load valid role
            assert len(loaded) == 1
            assert "valid" in loaded
            assert manager.get_role("valid") is not None

    def test_load_from_directory_markdown_precedence(self):
        """Test that markdown files take precedence over config.json."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "precedence_role"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("From markdown")

            config = {
                "description": "From config",
                "required_tools": ["config_tool"]
            }
            (role_dir / "config.json").write_text(json.dumps(config))

            manager = RoleManager()
            manager.load_from_dir(tmpdir)

            role = manager.get_role("precedence_role")
            # Markdown should take precedence
            assert role.description == "From markdown"
