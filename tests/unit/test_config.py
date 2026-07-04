"""Tests for ConfigManager."""

from pathlib import Path
from unittest.mock import patch

import pytest

from peteos.config import ConfigManager


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
    """Tests for ConfigManager.resolve_config_path()."""

    def test_env_var_priority(self, tmp_path):
        """PETEOS_CONFIG env var resolves before XDG/system/local."""
        env_file = tmp_path / "env_peteos.json"
        env_file.write_text('{"backends": []}')

        with patch("os.environ.get", side_effect=lambda key, default=None: {
            "PETEOS_CONFIG": str(env_file),
        }.get(key, default)):
            resolved = ConfigManager.resolve_config_path()

        assert resolved == str(env_file)

    def test_xdg_config(self, tmp_path):
        """$XDG_CONFIG_HOME/peteos/peteos.json is resolved."""
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
                resolved = ConfigManager.resolve_config_path()

        assert resolved == str(xdg_path)

    def test_system_config(self):
        """/etc/peteos/peteos.json is resolved when no env or XDG."""
        with patch("os.environ.get", side_effect=lambda key, default=None: default):
            def mock_is_file(self):
                return str(self) == "/etc/peteos/peteos.json"

            with patch.object(Path, "is_file", mock_is_file):
                resolved = ConfigManager.resolve_config_path()

        assert resolved == "/etc/peteos/peteos.json"

    def test_local_cwd(self, tmp_path):
        """./peteos.json is resolved as last resort."""
        local_path = tmp_path / "peteos.json"
        local_path.write_text('{"backends": []}')

        with patch("os.environ.get", side_effect=lambda key, default=None: default):
            def mock_is_file(self):
                return str(self) == str(local_path)

            with patch.object(Path, "is_file", mock_is_file):
                with patch("pathlib.Path.cwd", return_value=tmp_path):
                    resolved = ConfigManager.resolve_config_path()

        assert resolved == str(local_path)

    def test_no_file(self):
        """When no env vars set, returns None if no peteos.json exists."""
        with patch("os.environ.get", side_effect=lambda key, default=None: default):
            with patch("pathlib.Path.is_file", return_value=False):
                resolved = ConfigManager.resolve_config_path()

        assert resolved is None


class TestConfigInit:
    """Tests for ConfigManager.init()."""

    @pytest.mark.asyncio
    async def test_init_no_config_file(self):
        """When no config file exists, returns ConfigResult with None config_dir."""
        with patch.object(ConfigManager, "resolve_config_path", return_value=None):
            result = await ConfigManager.init()

        assert result.config_dir is None
        assert result.loaded_backends == []
        assert result.loaded_roles == []

    @pytest.mark.asyncio
    async def test_init_loads_backends(self, tmp_path):
        """ConfigManager.init loads backends via ChatBotManager.load_from_json."""
        config_file = tmp_path / "peteos.json"
        config_file.write_text(
            '{"backends": [{"name": "test-backend", "url": "http://test:8000"}]}'
        )

        mock_response = {"data": [{"id": "test-model"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with patch.object(ConfigManager, "resolve_config_path", return_value=str(config_file)):
                result = await ConfigManager.init()

        assert result.config_dir == str(tmp_path)
        assert "test-backend" in result.loaded_backends

    @pytest.mark.asyncio
    async def test_init_sets_config_dir(self, tmp_path):
        """Config directory is set to the parent of the loaded peteos.json."""
        config_dir = tmp_path / "peteos_config"
        config_dir.mkdir()
        config_file = config_dir / "peteos.json"
        config_file.write_text(
            '{"backends": [{"name": "dir-backend", "url": "http://dir:8000"}]}'
        )

        mock_response = {"data": [{"id": "dir-model"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with patch.object(ConfigManager, "resolve_config_path", return_value=str(config_file)):
                result = await ConfigManager.init()

        assert result.config_dir == str(config_dir)

    @pytest.mark.asyncio
    async def test_init_no_roles_directory(self, tmp_path):
        """ConfigManager.init does not fail when roles directory doesn't exist."""
        config_file = tmp_path / "peteos.json"
        config_file.write_text('{"backends": []}')

        with patch.object(ConfigManager, "resolve_config_path", return_value=str(config_file)):
            result = await ConfigManager.init()

        assert result.config_dir == str(tmp_path)
        assert result.loaded_roles == []

    @pytest.mark.asyncio
    async def test_init_returns_config_result(self, tmp_path):
        """ConfigManager.init returns a ConfigResult with loaded backends and roles."""
        config_file = tmp_path / "peteos.json"
        config_file.write_text(
            '{"backends": [{"name": "backend1", "url": "http://b1:8000"}]}'
        )

        mock_response = {"data": [{"id": "model1"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with patch.object(ConfigManager, "resolve_config_path", return_value=str(config_file)):
                result = await ConfigManager.init()

        assert result.config_dir == str(tmp_path)
        assert "backend1" in result.loaded_backends


class TestConfigResult:
    """Tests for ConfigResult dataclass."""

    def test_default_values(self):
        """ConfigResult defaults to None for config_dir and empty lists."""
        from peteos.config import ConfigResult

        result = ConfigResult()
        assert result.config_dir is None
        assert result.loaded_backends == []
        assert result.loaded_roles == []

    def test_can_set_values(self):
        """ConfigResult can be initialized with custom values."""
        from peteos.config import ConfigResult

        result = ConfigResult(
            config_dir="/test/path",
            loaded_backends=["backend1"],
            loaded_roles=["role1"],
        )
        assert result.config_dir == "/test/path"
        assert result.loaded_backends == ["backend1"]
        assert result.loaded_roles == ["role1"]
