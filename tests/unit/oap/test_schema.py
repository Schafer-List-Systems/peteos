"""Tests for OAP output schema module."""

from dataclasses import dataclass
from typing import TypedDict

import pytest

from peteos.oap._schema import (
    extract_schema_info,
    format_schema_prompt,
    parse_output,
)


class TestExtractSchemaInfo:
    def test_none_schema(self):
        assert extract_schema_info(None) is None

    def test_dataclass_schema(self):
        @dataclass
        class Report:
            name: str
            count: int

        info = extract_schema_info(Report)
        assert info["type"] == "object"
        assert "name" in info["properties"]
        assert "count" in info["properties"]
        assert info["properties"]["name"]["type"] == "string"
        assert info["properties"]["count"]["type"] == "integer"

    def test_required_fields(self):
        @dataclass
        class Report:
            name: str
            count: int

        info = extract_schema_info(Report)
        assert set(info.get("required", [])) == {"name", "count"}

    def test_typeddict_schema(self):
        class Config(TypedDict):
            host: str
            port: int

        info = extract_schema_info(Config)
        assert "host" in info["properties"]
        assert "port" in info["properties"]


class TestFormatSchemaPrompt:
    def test_empty_for_none(self):
        assert format_schema_prompt(None) == ""

    def test_includes_properties(self):
        info = {"type": "object", "properties": {"name": {"type": "string"}}}
        prompt = format_schema_prompt(info)
        assert "name" in prompt
        assert "string" in prompt
        assert "JSON" in prompt


class TestParseOutput:
    def test_none_schema_returns_raw(self):
        assert parse_output("hello", None) == "hello"

    def test_non_type_returns_raw(self):
        assert parse_output("hello", str) == "hello"

    def test_parse_dataclass(self):
        @dataclass
        class Report:
            name: str
            count: int

        result = parse_output('{"name": "test", "count": 42}', Report)
        assert isinstance(result, Report)
        assert result.name == "test"
        assert result.count == 42
