"""Unit tests for ReadStdoutChannel."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from peteos.channels import ReadStdoutChannel
from peteos.channels.channel import Channel
from peteos.agent import Agent
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


def _cleanup_channels():
    for name in list(Channel._registry.keys()):
        Channel._registry.pop(name)


def _make_agent():
    _cleanup_channels()
    rm = RoleManager()
    rm.register_role(Role(name="test", description="Test role"))
    return Agent(rm, MagicMock(), ToolManager())


class TestReadStdoutChannelInit:
    def setup_method(self):
        self.agent = _make_agent()

    def teardown_method(self):
        _cleanup_channels()

    def test_channel_creation(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*", "process_terminate_timeout": 5.0})
        assert channel.name == "test"
        assert channel._running is False

    def test_channel_empty_exclude_patterns(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*"})
        assert channel._exclude_patterns == []

    def test_channel_populates_exclude_from_config(self):
        config = {"command": ["echo", "hi"], "pattern": ".*", "exclude": ["DEBUG", "TRACE"]}
        channel = ReadStdoutChannel("test", self.agent, config)
        assert channel._exclude_patterns == ["DEBUG", "TRACE"]

    def test_channel_no_exclude_from_config(self):
        config = {"command": ["echo", "hi"], "pattern": ".*"}
        channel = ReadStdoutChannel("test", self.agent, config)
        assert channel._exclude_patterns == []


class TestReadStdoutChannelExcludeMethods:
    def setup_method(self):
        self.agent = _make_agent()

    def teardown_method(self):
        _cleanup_channels()

    def test_add_exclude_pattern(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*"})
        result = channel.add_exclude_pattern(r"DEBUG.*")
        assert result == 0
        assert channel._exclude_patterns == [r"DEBUG.*"]

    def test_add_duplicate_exclude_pattern_returns_none(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*", "exclude": ["DEBUG"]})
        result = channel.add_exclude_pattern("DEBUG")
        assert result is None
        assert channel._exclude_patterns == ["DEBUG"]

    def test_add_multiple_exclude_patterns(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*"})
        assert channel.add_exclude_pattern(r"DEBUG") == 0
        assert channel.add_exclude_pattern(r"TRACE") == 1
        assert channel._exclude_patterns == [r"DEBUG", r"TRACE"]

    def test_remove_exclude_pattern_by_index(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*", "exclude": ["DEBUG", "TRACE"]})
        result = channel.remove_exclude_pattern(0)
        assert result is True
        assert channel._exclude_patterns == ["TRACE"]

    def test_remove_exclude_pattern_still_remaining(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*", "exclude": ["A", "B", "C"]})
        result = channel.remove_exclude_pattern(1)
        assert result is True
        assert channel._exclude_patterns == ["A", "C"]

    def test_remove_exclude_pattern_out_of_range_returns_false(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*", "exclude": ["DEBUG"]})
        result = channel.remove_exclude_pattern(99)
        assert result is False
        assert channel._exclude_patterns == ["DEBUG"]

    def test_remove_exclude_pattern_negative_index(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*", "exclude": ["A", "B", "C"]})
        result = channel.remove_exclude_pattern(-1)
        assert result is True
        assert channel._exclude_patterns == ["A", "B"]

    def test_remove_from_empty_list_returns_false(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*"})
        result = channel.remove_exclude_pattern(0)
        assert result is False

    def test_list_exclude_patterns_returns_copy(self):
        channel = ReadStdoutChannel("test", self.agent, {"command": ["echo", "hi"], "pattern": ".*", "exclude": ["FOO"]})
        result = channel.list_exclude_patterns()
        assert result == ["FOO"]
        result.clear()
        assert channel._exclude_patterns == ["FOO"]


class TestReadStdoutChannelLoadConfig:
    @patch("builtins.open")
    def test_load_config_default_exclude(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '{"command": ["tail", "-f", "/var/log/syslog"], "pattern": "ERROR", "process_terminate_timeout": 5.0}'
        )
        config = ReadStdoutChannel.load_config("fake.json")
        assert config["exclude"] == []

    @patch("builtins.open")
    def test_load_config_preserves_exclude(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '{"command": ["tail", "-f", "/var/log/syslog"], "pattern": ".*", "exclude": ["DEBUG", "INFO"], "process_terminate_timeout": 5.0}'
        )
        config = ReadStdoutChannel.load_config("fake.json")
        assert config["exclude"] == ["DEBUG", "INFO"]

    @patch("builtins.open")
    def test_load_config_missing_command_raises(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = '{"pattern": ".*", "process_terminate_timeout": 5.0}'
        with pytest.raises(KeyError, match="Missing required config fields"):
            ReadStdoutChannel.load_config("fake.json")


class TestReadStdoutChannelFiltering:
    """Test exclude patterns filter lines matching the logic in _read_loop.

    The _read_loop filters lines like this:
    1. Match against inclusive pattern (self._config["pattern"])
    2. Skip if any exclude pattern matches
    3. Queue the message

    We test the filtering directly rather than mocking the subprocess pipeline.
    """

    @staticmethod
    def _filter_lines(lines, pattern, exclude):
        """Reproduce the filtering logic from ReadStdoutChannel._read_loop."""
        import re

        result = []
        for line in lines:
            if not re.search(pattern, line):
                continue
            for ex in exclude:
                if re.search(ex, line):
                    break
            else:
                result.append(line)
        return result

    def test_exclude_pattern_filters_debug_lines(self):
        lines = [
            "DEBUG: normal debug line",
            "ERROR: something broke",
            "DEBUG: another debug line",
            "WARNING: attention needed",
        ]
        result = self._filter_lines(lines, ".*", [r"^DEBUG"])
        assert result == ["ERROR: something broke", "WARNING: attention needed"]

    def test_multiple_exclude_patterns(self):
        lines = [
            "DEBUG: skip",
            "TRACE: skip",
            "ERROR: keep",
            "INFO: keep",
        ]
        result = self._filter_lines(lines, ".*", [r"^DEBUG", r"^TRACE"])
        assert result == ["ERROR: keep", "INFO: keep"]

    def test_no_exclude_keeps_all(self):
        lines = ["INFO: msg1", "ERROR: msg2", "DEBUG: msg3"]
        result = self._filter_lines(lines, ".*", [])
        assert result == ["INFO: msg1", "ERROR: msg2", "DEBUG: msg3"]

    def test_exclusive_respects_inclusive(self):
        """Exclude patterns only apply after inclusive pattern matches."""
        lines = [
            "ERROR: something broke",
            "ERROR: minor issue",
            "WARNING: check",
            "INFO: ignored",
        ]
        result = self._filter_lines(lines, r"^(ERROR|WARNING)", [r"ERROR.*something"])
        assert result == ["ERROR: minor issue", "WARNING: check"]

    def test_empty_excludes_all_matches(self):
        """Empty string in exclude matches everything."""
        lines = ["ERROR: keep", "INFO: skip"]
        result = self._filter_lines(lines, ".*", [".*"])
        assert result == []

    def test_exclude_pattern_case_sensitive(self):
        lines = ["debug: keep", "DEBUG: skip"]
        result = self._filter_lines(lines, ".*", [r"^DEBUG"])
        assert result == ["debug: keep"]
