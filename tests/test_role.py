from pathlib import Path
import tempfile
import shutil

from peteos.role import Role


def test_role_init():
    """Test Role initialization."""
    role = Role(
        name="test_role",
        description="A test role",
        system_prompt="You are a test assistant."
    )

    assert role.name == "test_role"
    assert role.description == "A test role"
    assert role.system_prompt == "You are a test assistant."


def test_role_init_without_system_prompt():
    """Test Role initialization without system prompt."""
    role = Role(name="simple", description="Simple role")

    assert role.name == "simple"
    assert role.description == "Simple role"
    assert role.system_prompt is None


def test_role_load_from_dict():
    """Test loading Role from dictionary."""
    data = {
        "name": "coder",
        "description": "A software developer role",
        "system_prompt": "You are a coding expert."
    }

    role = Role.load_from_dict(data)

    assert role.name == "coder"
    assert role.description == "A software developer role"
    assert role.system_prompt == "You are a coding expert."


def test_role_load_from_dict_without_system_prompt():
    """Test loading Role from dictionary without system_prompt."""
    data = {
        "name": "helper",
        "description": "A helpful assistant"
    }

    role = Role.load_from_dict(data)

    assert role.name == "helper"
    assert role.description == "A helpful assistant"
    assert role.system_prompt is None


def test_role_load_from_path():
    """Test loading Role from directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "assistant"
        role_dir.mkdir()

        (role_dir / "description.md").write_text("You are a helpful assistant.")
        (role_dir / "system_prompt.md").write_text("Always be polite and helpful.")

        role = Role.load_from_path(str(role_dir))

        assert role.name == "assistant"
        assert role.description == "You are a helpful assistant."
        assert role.system_prompt == "Always be polite and helpful."


def test_role_load_from_path_without_system_prompt():
    """Test loading Role from directory without system_prompt.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "bot"
        role_dir.mkdir()

        (role_dir / "description.md").write_text("A basic bot.")

        role = Role.load_from_path(str(role_dir))

        assert role.name == "bot"
        assert role.description == "A basic bot."
        assert role.system_prompt is None
