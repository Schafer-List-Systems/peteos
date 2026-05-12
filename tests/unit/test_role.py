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
    assert role.required_tools == []
    assert role.execution_environment == "REPL"


def test_role_init_without_system_prompt():
    """Test Role initialization without system prompt."""
    role = Role(name="simple", description="Simple role")

    assert role.name == "simple"
    assert role.description == "Simple role"
    assert role.system_prompt is None
    assert role.required_tools == []
    assert role.execution_environment == "REPL"


def test_role_init_with_config():
    """Test Role initialization with config values."""
    role = Role(
        name="coder",
        description="A coding assistant",
        system_prompt="You are a coding expert.",
        required_tools=["file_read", "file_write"],
        execution_environment="REPL"
    )

    assert role.name == "coder"
    assert role.description == "A coding assistant"
    assert role.system_prompt == "You are a coding expert."
    assert role.required_tools == ["file_read", "file_write"]
    assert role.execution_environment == "REPL"


def test_role_init_with_custom_execution_environment():
    """Test Role initialization with custom execution environment."""
    role = Role(
        name="bot",
        description="A simple bot",
        execution_environment="Jupyter"
    )

    assert role.name == "bot"
    assert role.description == "A simple bot"
    assert role.required_tools == []
    assert role.execution_environment == "Jupyter"


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
    assert role.required_tools == []
    assert role.execution_environment == "REPL"


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
    assert role.required_tools == []
    assert role.execution_environment == "REPL"


def test_role_load_from_dict_with_config():
    """Test loading Role from dictionary with config values."""
    data = {
        "name": "analyst",
        "description": "A data analyst",
        "system_prompt": "You analyze data.",
        "required_tools": ["read_csv", "plot_data"],
        "execution_environment": "Jupyter"
    }

    role = Role.load_from_dict(data)

    assert role.name == "analyst"
    assert role.description == "A data analyst"
    assert role.system_prompt == "You analyze data."
    assert role.required_tools == ["read_csv", "plot_data"]
    assert role.execution_environment == "Jupyter"


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
        assert role.required_tools == []
        assert role.execution_environment == "REPL"


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
        assert role.required_tools == []
        assert role.execution_environment == "REPL"


def test_role_load_from_path_with_config():
    """Test loading Role from directory with config.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "coder"
        role_dir.mkdir()

        (role_dir / "description.md").write_text("A coding assistant.")
        (role_dir / "system_prompt.md").write_text("You are a coding expert.")
        (role_dir / "config.json").write_text(
            '{"required_tools": ["file_read", "file_write"], '
            '"execution_environment": "REPL"}'
        )

        role = Role.load_from_path(str(role_dir))

        assert role.name == "coder"
        assert role.description == "A coding assistant."
        assert role.system_prompt == "You are a coding expert."
        assert role.required_tools == ["file_read", "file_write"]
        assert role.execution_environment == "REPL"


def test_role_load_from_path_with_partial_config():
    """Test loading Role from directory with partial config.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "analyst"
        role_dir.mkdir()

        (role_dir / "description.md").write_text("A data analyst.")
        (role_dir / "config.json").write_text(
            '{"required_tools": ["read_csv"]}'
        )

        role = Role.load_from_path(str(role_dir))

        assert role.name == "analyst"
        assert role.description == "A data analyst."
        assert role.system_prompt is None
        assert role.required_tools == ["read_csv"]
        assert role.execution_environment == "REPL"


def test_role_load_config_from_path_no_config():
    """Test load_config_from_path when config.json doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "simple"
        role_dir.mkdir()

        config = Role.load_config_from_path(str(role_dir))

        assert config == {}


def test_role_load_config_from_path_with_config():
    """Test load_config_from_path when config.json exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "configured"
        role_dir.mkdir()

        (role_dir / "config.json").write_text(
            '{"required_tools": ["tool1", "tool2"], '
            '"execution_environment": "Jupyter"}'
        )

        config = Role.load_config_from_path(str(role_dir))

        assert config == {
            "required_tools": ["tool1", "tool2"],
            "execution_environment": "Jupyter"
        }


def test_role_load_from_path_md_precedence():
    """Test that markdown files have precedence over config.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "role"
        role_dir.mkdir()

        (role_dir / "description.md").write_text("From markdown.")
        (role_dir / "system_prompt.md").write_text("Prompt from markdown.")
        (role_dir / "config.json").write_text(
            '{"description": "From config", '
            '"system_prompt": "Prompt from config", '
            '"required_tools": ["tool1"]}'
        )

        role = Role.load_from_path(str(role_dir))

        # Markdown should take precedence
        assert role.description == "From markdown."
        assert role.system_prompt == "Prompt from markdown."
        assert role.required_tools == ["tool1"]


def test_role_load_from_path_config_fallback():
    """Test that config.json is used as fallback when markdown files missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "role"
        role_dir.mkdir()

        (role_dir / "description.md").write_text("Only description exists.")
        (role_dir / "config.json").write_text(
            '{"description": "From config", '
            '"system_prompt": "Prompt from config"}'
        )

        role = Role.load_from_path(str(role_dir))

        # Description from markdown, system_prompt from config fallback
        assert role.description == "Only description exists."
        assert role.system_prompt == "Prompt from config"


def test_role_load_from_path_config_only():
    """Test loading only from config.json when no markdown files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "role"
        role_dir.mkdir()

        (role_dir / "config.json").write_text(
            '{"description": "Config only", '
            '"system_prompt": "System prompt from config", '
            '"required_tools": ["tool1", "tool2"], '
            '"execution_environment": "Jupyter"}'
        )

        role = Role.load_from_path(str(role_dir))

        assert role.description == "Config only"
        assert role.system_prompt == "System prompt from config"
        assert role.required_tools == ["tool1", "tool2"]
        assert role.execution_environment == "Jupyter"


def test_role_init_with_auto_approve_tools():
    """Test Role initialization with auto_approve_tools."""
    role = Role(
        name="autobot",
        description="Auto-approves some tools",
        auto_approve_tools=["file_read", "calc"]
    )

    assert role.name == "autobot"
    assert role.description == "Auto-approves some tools"
    assert role.auto_approve_tools == ["file_read", "calc"]


def test_role_init_default_auto_approve_tools():
    """Test Role default auto_approve_tools is empty list."""
    role = Role(name="simple", description="Simple role")

    assert role.auto_approve_tools == []


def test_role_load_from_dict_with_auto_approve_tools():
    """Test loading Role from dictionary with auto_approve_tools."""
    data = {
        "name": "autobot",
        "description": "Auto-approves web_fetch",
        "auto_approve_tools": ["web_fetch"]
    }

    role = Role.load_from_dict(data)

    assert role.name == "autobot"
    assert role.description == "Auto-approves web_fetch"
    assert role.auto_approve_tools == ["web_fetch"]


def test_role_load_from_dict_default_auto_approve_tools():
    """Test loading Role from dictionary without auto_approve_tools."""
    data = {
        "name": "simple",
        "description": "Simple role"
    }

    role = Role.load_from_dict(data)

    assert role.auto_approve_tools == []


def test_role_load_from_path_with_auto_approve_tools_config():
    """Test loading Role from directory with auto_approve_tools in config.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "autobot"
        role_dir.mkdir()

        (role_dir / "description.md").write_text("An auto-approver.")
        (role_dir / "config.json").write_text(
            '{"auto_approve_tools": ["file_read", "calc"]}'
        )

        role = Role.load_from_path(str(role_dir))

        assert role.name == "autobot"
        assert role.description == "An auto-approver."
        assert role.auto_approve_tools == ["file_read", "calc"]


def test_role_load_from_path_default_auto_approve_tools():
    """Test loading Role from directory without auto_approve_tools in config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        role_dir = Path(tmpdir) / "simple"
        role_dir.mkdir()

        (role_dir / "description.md").write_text("A simple role.")

        role = Role.load_from_path(str(role_dir))

        assert role.auto_approve_tools == []
