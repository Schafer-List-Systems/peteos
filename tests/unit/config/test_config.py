"""Tests for ConfigManager."""

from pathlib import Path
from unittest.mock import patch

import pytest

from peteos.config import ConfigManager
from peteos.config import _resolve_config_path


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
        ConfigManager._config_file = None
        ConfigManager._config_dir = None
        ConfigManager.reset() if hasattr(ConfigManager, "reset") else None

    @pytest.mark.asyncio
    async def test_init_no_config_file(self):
        """When no config file exists, returns True and sets no config state."""
        self._reset()
        with patch("peteos.config._resolve_config_path", return_value=(None, None)):
            result = await ConfigManager.init()

        assert result is True
        assert ConfigManager._config_file is None
        assert ConfigManager._config_dir is None

    @pytest.mark.asyncio
    async def test_init_loads_backends(self, tmp_path):
        """ConfigManager.init loads backends and sets _config_file and _config_dir on the singleton."""
        self._reset()
        config_file = tmp_path / "peteos.json"
        config_file.write_text(
            '{"backends": [{"name": "test-backend", "url": "http://test:8000"}]}'
        )
        config_dir = str(tmp_path / ".peteos")

        mock_response = {"data": [{"id": "test-model"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with patch("peteos.config._resolve_config_path",
                       return_value=(str(config_file), config_dir)):
                result = await ConfigManager.init()

        assert result is True
        assert ConfigManager._config_file == str(config_file)
        assert ConfigManager._config_dir == config_dir

    @pytest.mark.asyncio
    async def test_init_is_idempotent(self, tmp_path):
        """Second init() call returns False and does not reload backends."""
        self._reset()
        config_file = tmp_path / "peteos.json"
        config_file.write_text(
            '{"backends": [{"name": "idempotent-backend", "url": "http://id:8000"}]}'
        )
        config_dir = str(tmp_path / ".peteos")

        mock_response = {"data": [{"id": "idempotent-model"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with patch("peteos.config._resolve_config_path",
                       return_value=(str(config_file), config_dir)):
                first = await ConfigManager.init()
                second = await ConfigManager.init()

        assert first is True
        assert second is False
        assert ConfigManager._config_file == str(config_file)

    @pytest.mark.asyncio
    async def test_init_no_roles_directory(self, tmp_path):
        """ConfigManager.init does not fail when roles directory doesn't exist."""
        self._reset()
        config_file = tmp_path / "peteos.json"
        config_file.write_text('{"backends": []}')
        config_dir = str(tmp_path / ".peteos")

        with patch("peteos.config._resolve_config_path",
                   return_value=(str(config_file), config_dir)):
            result = await ConfigManager.init()

        assert result is True
        assert ConfigManager._config_dir == config_dir

    @pytest.mark.asyncio
    async def test_init_returns_true_on_success(self, tmp_path):
        """init() returns True when configuration is loaded successfully."""
        self._reset()
        config_file = tmp_path / "peteos.json"
        config_file.write_text(
            '{"backends": [{"name": "backend1", "url": "http://b1:8000"}]}'
        )
        config_dir = str(tmp_path / ".peteos")

        mock_response = {"data": [{"id": "model1"}]}

        with patch("httpx.AsyncClient", _make_httpx_mock(return_value=mock_response)):
            with patch("peteos.config._resolve_config_path",
                       return_value=(str(config_file), config_dir)):
                result = await ConfigManager.init()

        assert result is True
        assert ConfigManager._config_file == str(config_file)
        assert ConfigManager._config_dir == config_dir
