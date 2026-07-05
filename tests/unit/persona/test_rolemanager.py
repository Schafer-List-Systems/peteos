"""Unit tests for the persona RoleManager class."""

import tempfile
from pathlib import Path

import pytest

from peteos.persona.role import Role
from peteos.persona.rolemanager import RoleManager


class TestRoleManagerInit:
    """Test RoleManager initialization."""

    def test_init_empty_registry(self):
        manager = RoleManager()
        assert manager.list_roles() == []


class TestRoleManagerRegister:
    """Test RoleManager.register_role."""

    def test_register_and_get(self):
        manager = RoleManager()
        role = Role(name="test", description="Test role")
        manager.register_role(role)
        assert manager.get_role("test") is role

    def test_register_overwrites(self):
        manager = RoleManager()
        role1 = Role(name="dup", description="First")
        role2 = Role(name="dup", description="Second")
        manager.register_role(role1)
        manager.register_role(role2)
        assert manager.get_role("dup").description == "Second"

    def test_register_list_roles(self):
        manager = RoleManager()
        r1 = Role(name="a", description="A")
        r2 = Role(name="b", description="B")
        manager.register_role(r1)
        manager.register_role(r2)
        names, roles = zip(*manager.list_roles())
        assert names == ("a", "b")
        assert roles == (r1, r2)


class TestRoleManagerLoadFromDir:
    """Test RoleManager.load_from_dir."""

    def test_load_from_nonexistent_directory(self):
        manager = RoleManager()
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            manager.load_from_dir("/nonexistent/path")

    def test_load_single_role(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "helper"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("A helper.")
            manager = RoleManager()
            loaded = manager.load_from_dir(tmpdir)
            assert loaded == ["helper"]
            assert manager.get_role("helper") is not None
            assert manager.get_role("helper").description == "A helper."

    def test_load_multiple_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, prompt in [("alpha", "Alpha desc"), ("beta", "Beta desc")]:
                role_dir = Path(tmpdir) / name
                role_dir.mkdir()
                (role_dir / "description.md").write_text(f"{name} description.")
            manager = RoleManager()
            loaded = manager.load_from_dir(tmpdir)
            assert sorted(loaded) == ["alpha", "beta"]
            assert manager.get_role("alpha") is not None
            assert manager.get_role("beta") is not None

    def test_load_skips_unloadable_roles(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid role
            good = Path(tmpdir) / "good"
            good.mkdir()
            (good / "description.md").write_text("Good role.")
            # Invalid role (malformed config.json causes load failure)
            bad = Path(tmpdir) / "bad"
            bad.mkdir()
            (bad / "description.md").write_text("Bad role.")
            (bad / "config.json").write_text("{ invalid json")
            manager = RoleManager()
            loaded = manager.load_from_dir(tmpdir)
            assert loaded == ["good"]
            assert manager.get_role("good") is not None
            assert manager.get_role("bad") is None

    def test_load_loads_config_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "configured"
            role_dir.mkdir()
            (role_dir / "description.md").write_text("Desc.")
            (role_dir / "config.json").write_text(
                '{"required_tools": ["t1"], "execution_environment": "Jupyter", "model": "gpt-4"}'
            )
            manager = RoleManager()
            manager.load_from_dir(tmpdir)
            role = manager.get_role("configured")
            assert role.required_tools == ["t1"]
            assert role.execution_environment == "Jupyter"
            assert role.model == "gpt-4"

    def test_load_skips_subdirectories_that_are_not_role_dirs(self):
        """Subdirectories without description.md should be skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty_role"
            empty_dir.mkdir()
            # No description.md and invalid config.json — should be skipped
            good_dir = Path(tmpdir) / "good"
            good_dir.mkdir()
            (good_dir / "description.md").write_text("Good.")
            (empty_dir / "config.json").write_text("{ bad json")
            manager = RoleManager()
            loaded = manager.load_from_dir(tmpdir)
            assert loaded == ["good"]
