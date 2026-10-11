"""Tests for ConfigManager."""

from pathlib import Path
from unittest.mock import patch

import pytest

from peteos.config import ConfigManager
from peteos.config import _resolve_config_path


def _reset_config_state():
    """Reset singleton state before each test."""
    import os
    ConfigManager._config_file = None
    ConfigManager._config_dir = None
    ConfigManager._variables.clear()
    ConfigManager._on_reloaded_hooks.clear()
    ConfigManager._file_observer = None
    ConfigManager._file_observer_thread = None
    # Clear PETEOS_* env vars so re-init uses test fixture
    for k in list(os.environ):
        if k.startswith("PETEOS_"):
            del os.environ[k]


class TestConfigVariables:
    """Tests for ConfigManager variable management (set_variable, get_variable)."""

    def setup_method(self):
        _reset_config_state()

    def teardown_method(self):
        _reset_config_state()

    def test_set_variable_basic(self):
        """set_variable stores the key-value pair and returns True."""
        result = ConfigManager.set_variable("FOO", "bar")
        assert result is True
        assert ConfigManager._variables["FOO"] == "bar"

    def test_set_variable_no_overwrite_returns_False(self):
        """set_variable returns False when key exists and overwrite is False."""
        ConfigManager.set_variable("FOO", "bar")
        result = ConfigManager.set_variable("FOO", "baz", overwrite=False)
        assert result is False
        assert ConfigManager._variables["FOO"] == "bar"

    def test_set_variable_overwrites_when_allowed(self):
        """set_variable replaces value and returns True when overwrite=True."""
        ConfigManager.set_variable("FOO", "bar")
        result = ConfigManager.set_variable("FOO", "baz", overwrite=True)
        assert result is True
        assert ConfigManager._variables["FOO"] == "baz"

    def test_get_variable_returns_value(self):
        """get_variable returns the stored value."""
        ConfigManager.set_variable("FOO", "bar")
        assert ConfigManager.get_variable("FOO") == "bar"

    def test_get_variable_raises_if_missing(self):
        """get_variable raises KeyError when the key is not set."""
        with pytest.raises(KeyError):
            ConfigManager.get_variable("DOES_NOT_EXIST")


class TestConfigPathResolution:
    """Tests for ConfigManager.resolve_path() template expansion."""

    def setup_method(self):
        _reset_config_state()

    def teardown_method(self):
        _reset_config_state()

    def test_resolve_path_basic(self):
        """A single {VAR} token is replaced with the variable value."""
        ConfigManager.set_variable("FOO", "bar")
        result = ConfigManager.resolve_path("{FOO}")
        assert result == Path("bar")

    def test_resolve_path_returns_Path(self):
        """resolve_path returns a Path object, not a string."""
        ConfigManager.set_variable("BASE", "/home/user")
        result = ConfigManager.resolve_path("{BASE}/projects")
        assert isinstance(result, Path)
        assert result.parts == ("/", "home", "user", "projects")

    def test_resolve_path_multiple_tokens(self):
        """Multiple tokens in one template are all resolved."""
        ConfigManager.set_variable("A", "alpha")
        ConfigManager.set_variable("B", "beta")
        result = ConfigManager.resolve_path("{A}-{B}-{A}")
        assert result == Path("alpha-beta-alpha")

    def test_resolve_path_same_token_twice(self):
        """The same token used twice is resolved independently (no false cycle)."""
        ConfigManager.set_variable("X", "hello")
        result = ConfigManager.resolve_path("{X} {X}")
        assert result == Path("hello hello")

    def test_resolve_path_missing_variable_raises_KeyError(self):
        """A template with an unset variable raises KeyError."""
        ConfigManager.set_variable("FOO", "bar")
        with pytest.raises(KeyError):
            ConfigManager.resolve_path("{MISSING}")

    def test_resolve_path_nested_expansion(self):
        """A value referencing another variable is expanded recursively."""
        ConfigManager.set_variable("A", "{B}")
        ConfigManager.set_variable("B", "final")
        result = ConfigManager.resolve_path("{A}")
        assert result == Path("final")

    def test_resolve_path_deep_nested_expansion(self):
        """Chains of three or more levels resolve correctly."""
        ConfigManager.set_variable("TOP", "{MID}")
        ConfigManager.set_variable("MID", "{BOT}")
        ConfigManager.set_variable("BOT", "deep-value")
        result = ConfigManager.resolve_path("{TOP}")
        assert result == Path("deep-value")

    def test_resolve_path_cycle_raises_RecursionError(self):
        """A->{B}, B->{A} raises RecursionError."""
        ConfigManager.set_variable("A", "{B}")
        ConfigManager.set_variable("B", "{A}")
        with pytest.raises(RecursionError):
            ConfigManager.resolve_path("{A}")

    def test_resolve_path_self_cycle_raises_RecursionError(self):
        """A->{A} raises RecursionError."""
        ConfigManager.set_variable("A", "{A}")
        with pytest.raises(RecursionError):
            ConfigManager.resolve_path("{A}")

    def test_resolve_path_sibling_seen_sets_stay_isolated(self):
        """Two same tokens in one template don't share seen sets across siblings.

        This is the classic bug: if seen is shared across all branches,
        a template like {A}, {A} where A contains {B} would falsely detect a
        cycle. With isolated seen sets per branch, each {A} resolves in its
        own context.
        """
        ConfigManager.set_variable("A", "{B}")
        ConfigManager.set_variable("B", "leaf")
        result = ConfigManager.resolve_path("{A}, {A}")
        assert result == Path("leaf, leaf")

    def test_resolve_path_overrides_take_precedence(self):
        """The overrides dict shadows variables with the same key."""
        ConfigManager.set_variable("X", "shared")
        result = ConfigManager.resolve_path("{X}", overrides={"X": "override"})
        assert result == Path("override")

    def test_resolve_path_override_has_access_to_all_variables(self):
        """An override value can reference variables from the shared dict."""
        ConfigManager.set_variable("BASE", "/home")
        ConfigManager.set_variable("SUB", "projects")
        result = ConfigManager.resolve_path("{BASE}/{SUB}", overrides={"BASE": "/opt"})
        assert result == Path("/opt/projects")

    def test_resolve_path_escaped_braces(self):
        """{{ and }} are preserved literally in the output."""
        ConfigManager.set_variable("X", "value")
        result = ConfigManager.resolve_path("{{ literal }}, {X}")
        assert result == Path("{ literal }, value")


class _MockResponse:
    """Minimal mock HTTP response for httpx.AsyncClient."""

    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def _make_httpx_mock(return_value=None, side_effect=None):
    """Create a mock httpx.AsyncClient that returns pre-set responses."""
    if side_effect is not None:
        responses = list(side_effect)
        _idx = [0]

        class _MockClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, **kwargs):
                idx = _idx[0]
                _idx[0] += 1
                if idx < len(responses):
                    return _MockResponse(responses[idx])
                raise StopAsyncIteration

        return _MockClient()

    class _MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, **kwargs):
            if return_value is not None:
                return _MockResponse(return_value)
            return _MockResponse({})

    return _MockClient


class TestConfigFileResolution:
    """Tests for _resolve_config_path()."""

    def test_env_var_priority(self, tmp_path):
        """PETEOS_CONFIG env var resolves with non-dedicated .peteos subdirectory."""
        env_file = tmp_path / "env_peteos.json"
        env_file.write_text('{"backends": []}')

        with patch("os.environ.get", side_effect=lambda key, default=None: {
            "PETEOS_CONFIG": str(env_file),
        }.get(key, default)):
            resolved = _resolve_config_path()

        assert resolved == (str(env_file), str(env_file.parent / ".peteos"))

    def test_xdg_config(self, tmp_path):
        """$XDG_CONFIG_HOME/peteos/peteos.json is resolved as dedicated (no .peteos subdirectory)."""
        xdg_path = tmp_path / "my_config" / "peteos" / "peteos.json"
        xdg_path.parent.mkdir(parents=True)
        xdg_path.write_text('{"backends": []}')

        xdg_str = str(tmp_path / "my_config")

        def fake_is_file(self):
            return str(self) == str(xdg_path)

        with patch("os.environ.get", side_effect=lambda key, default=None: {
            "XDG_CONFIG_HOME": xdg_str,
        }.get(key, default)):
            with patch("pathlib.Path.is_file", fake_is_file):
                resolved = _resolve_config_path()

        assert resolved == (str(xdg_path), str(xdg_path.parent))

    def test_system_config(self):
        """/etc/peteos/peteos.json is resolved as dedicated (no .peteos subdirectory)."""
        with patch("os.environ.get", side_effect=lambda key, default=None: default):
            def mock_is_file(self):
                return str(self) == "/etc/peteos/peteos.json"

            with patch.object(Path, "is_file", mock_is_file):
                resolved = _resolve_config_path()

        assert resolved == ("/etc/peteos/peteos.json", "/etc/peteos")

    def test_local_cwd(self, tmp_path):
        """./peteos.json is resolved with non-dedicated .peteos subdirectory."""
        local_path = tmp_path / "peteos.json"
        local_path.write_text('{"backends": []}')

        with patch("os.environ.get", side_effect=lambda key, default=None: default):
            def mock_is_file(self):
                return str(self) == str(local_path)

            with patch.object(Path, "is_file", mock_is_file):
                with patch("pathlib.Path.cwd", return_value=tmp_path):
                    resolved = _resolve_config_path()

        assert resolved == (str(local_path), str(local_path.parent / ".peteos"))

    def test_no_file(self):
        """When no env vars set, returns (None, None) if no peteos.json exists."""
        with patch("os.environ.get", side_effect=lambda key, default=None: default):
            with patch("pathlib.Path.is_file", return_value=False):
                resolved = _resolve_config_path()

        assert resolved == (None, None)


class TestConfigInit:
    """Tests for ConfigManager.init()."""

    def _reset(self):
        """Reset singleton state before each test."""
        import os
        ConfigManager._config_file = None
        ConfigManager._config_dir = None
        ConfigManager._variables.clear()
        ConfigManager.reset() if hasattr(ConfigManager, "reset") else None
        from peteos.chatbot import ChatBotManager
        ChatBotManager._backends.clear()
        # Clear PETEOS_* env vars so re-init uses test fixture
        for k in list(os.environ):
            if k.startswith("PETEOS_"):
                del os.environ[k]

    def test_init_no_config_file(self):
        """When no config file exists, returns True and sets no config state."""
        self._reset()
        with patch("peteos.config._resolve_config_path", return_value=(None, None)):
            result = ConfigManager.init()

        assert result is True
        # No config file → CONFIG and CONFIG_DIR not set
        assert "CONFIG" not in ConfigManager._variables
        assert "CONFIG_DIR" not in ConfigManager._variables

    def test_init_sets_config_variable(self, tmp_path):
        """ConfigManager.init sets CONFIG and CONFIG_DIR variables."""
        self._reset()
        config_file = tmp_path / "peteos.json"
        config_file.write_text("{}")
        config_dir = str(tmp_path / ".peteos")

        with patch("peteos.config._resolve_config_path",
                   return_value=(str(config_file), config_dir)):
            result = ConfigManager.init()

        assert result is True
        assert ConfigManager.get_variable("CONFIG") == str(config_file)
        assert ConfigManager.get_variable("CONFIG_DIR") == config_dir

    def test_init_is_idempotent(self, tmp_path):
        """Second init() call returns False and does not overwrite variables."""
        self._reset()
        config_file = tmp_path / "peteos.json"
        config_file.write_text("{}")
        config_dir = str(tmp_path / ".peteos")

        with patch("peteos.config._resolve_config_path",
                   return_value=(str(config_file), config_dir)):
            first = ConfigManager.init()
            second = ConfigManager.init()

        assert first is True
        assert second is False
        assert ConfigManager.get_variable("CONFIG") == str(config_file)

    def test_init_no_roles_directory(self, tmp_path):
        """ConfigManager.init does not fail when roles directory doesn't exist."""
        self._reset()
        config_file = tmp_path / "peteos.json"
        config_file.write_text('{"backends": []}')
        config_dir = str(tmp_path / ".peteos")

        with patch("peteos.config._resolve_config_path",
                   return_value=(str(config_file), config_dir)):
            result = ConfigManager.init()

        assert result is True
        assert ConfigManager.get_variable("CONFIG_DIR") == config_dir

    def test_init_returns_true_on_success(self, tmp_path):
        """init() returns True when configuration is loaded successfully."""
        self._reset()
        config_file = tmp_path / "peteos.json"
        config_file.write_text(
            '{"backends": [{"name": "backend1", "url": "http://b1:8000"}]}'
        )
        config_dir = str(tmp_path / ".peteos")

        with patch("peteos.config._resolve_config_path",
                   return_value=(str(config_file), config_dir)):
            result = ConfigManager.init()

        assert result is True
        assert ConfigManager.get_variable("CONFIG") == str(config_file)
        assert ConfigManager.get_variable("CONFIG_DIR") == config_dir
