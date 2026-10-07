"""Unit tests for async function support in SandboxBuilder."""

from __future__ import annotations

import asyncio
import pytest

from peteos.sandbox import SandboxBuilder


class TestExtractFuncNameAndParamsAsync:
    """Tests for async detection in _extract_func_name_and_params."""

    def test_regular_function_is_not_async(self):
        from peteos.sandbox.sandbox_builder import _extract_func_name_and_params

        def regular():
            pass

        _, _, _, is_async = _extract_func_name_and_params(regular)
        assert is_async is False

    def test_async_function_is_async(self):
        from peteos.sandbox.sandbox_builder import _extract_func_name_and_params

        async def async_func():
            pass

        _, _, _, is_async = _extract_func_name_and_params(async_func)
        assert is_async is True

    def test_sync_function_is_not_async(self):
        from peteos.sandbox.sandbox_builder import _extract_func_name_and_params

        def sync_func():
            pass

        _, _, _, is_async = _extract_func_name_and_params(sync_func)
        assert is_async is False

    def test_async_method_is_async(self):
        from peteos.sandbox.sandbox_builder import _extract_func_name_and_params

        class MyClass:
            async def async_method(self):
                pass

        _, _, _, is_async = _extract_func_name_and_params(MyClass.async_method)
        assert is_async is True


class TestAsyncMemberNamesTracking:
    """Tests for _async_member_names tracking."""

    def test_async_function_registered_in_async_names(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code(
            "async def async_func(self) -> int:\n    return 42"
        )
        assert "async_func" in builder._async_member_names

    def test_sync_function_not_in_async_names(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code(
            "def sync_func(self) -> int:\n    return 42"
        )
        assert "sync_func" not in builder._async_member_names

    def test_remove_clears_async_name(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code(
            "async def async_func(self) -> int:\n    return 42"
        )
        assert "async_func" in builder._async_member_names
        builder.remove_source_code("async_func")
        assert "async_func" not in builder._async_member_names

    def test_async_names_are_per_builder_not_propagated_to_parents(self):
        cls_builder = SandboxBuilder("class")
        inst_builder = SandboxBuilder("instance", base=cls_builder)
        sess_builder = SandboxBuilder("session", base=inst_builder)

        cls_builder.add_safe_builtins()
        cls_builder.add_source_code(
            "async def cls_async(self) -> str:\n    return 'class'"
        )
        inst_builder.add_source_code(
            "async def inst_async(self) -> str:\n    return 'instance'"
        )
        sess_builder.add_source_code(
            "def sess_sync(self) -> str:\n    return 'session'"
        )

        assert "cls_async" in cls_builder._async_member_names
        assert "cls_async" not in inst_builder._async_member_names
        assert "cls_async" not in sess_builder._async_member_names

        assert "inst_async" in inst_builder._async_member_names
        assert "inst_async" not in sess_builder._async_member_names

        assert "sess_sync" not in sess_builder._async_member_names

    def test_build_description_walks_chain_for_async_names(self):
        cls_builder = SandboxBuilder("class")
        inst_builder = SandboxBuilder("instance", base=cls_builder)
        sess_builder = SandboxBuilder("session", base=inst_builder)

        cls_builder.add_safe_builtins()
        cls_builder.add_source_code(
            "async def cls_async(self) -> str:\n    return 'class'"
        )
        inst_builder.add_source_code(
            "async def inst_async(self) -> str:\n    return 'instance'"
        )
        sess_builder.add_source_code(
            "def sess_sync(self) -> str:\n    return 'session'"
        )

        desc = sess_builder.build_sandbox_description()
        assert "cls_async" in desc
        assert "inst_async" in desc

        inst_desc = inst_builder.build_sandbox_description()
        assert "cls_async" in inst_desc
        assert "inst_async" in inst_desc

        cls_desc = cls_builder.build_sandbox_description()
        assert "cls_async" in cls_desc


class TestBuildSandboxDescription:
    """Tests for async functions in build_sandbox_description."""

    def test_description_includes_async_functions(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code(
            "async def async_job(self) -> int:\n    return 42"
        )
        builder.add_source_code(
            "def sync_job(self) -> int:\n    return 42"
        )

        desc = builder.build_sandbox_description()
        assert "Async functions (await the result): async_job" in desc
        assert "sync_job" not in desc or "async_job" in desc

    def test_description_lists_multiple_async_functions(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code(
            "async def job_a(self) -> int:\n    return 1\n\n"
            "async def job_b(self) -> str:\n    return 'b'"
        )

        desc = builder.build_sandbox_description()
        assert "job_a" in desc
        assert "job_b" in desc

    def test_description_no_async_section_when_no_async(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code(
            "def sync_only(self) -> int:\n    return 42"
        )

        desc = builder.build_sandbox_description()
        assert "Async functions" not in desc

    def test_description_inherits_async_from_base(self):
        cls_builder = SandboxBuilder("class")
        inst_builder = SandboxBuilder("instance", base=cls_builder)

        cls_builder.add_safe_builtins()
        cls_builder.add_source_code(
            "async def base_async(self) -> int:\n    return 1"
        )

        desc = inst_builder.build_sandbox_description()
        assert "base_async" in desc


class TestAsyncProxyExecution:
    """Tests for running async functions via sandbox proxy."""

    @pytest.mark.asyncio
    async def test_async_proxy_returns_awaitable(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code(
            "async def async_compute(self, x: int) -> int:\n    return x * 2"
        )

        sandbox = builder.get_sandbox()
        result = sandbox.async_compute(21)

        assert asyncio.iscoroutinefunction(sandbox.async_compute) or asyncio.iscoroutine(result)
        final = await result
        assert final == 42

    @pytest.mark.asyncio
    async def test_mixed_sync_async_execution(self):
        builder = SandboxBuilder("session")
        builder.add_safe_builtins()
        builder.add_source_code(
            "def sync_add(self, a: int, b: int) -> int:\n    return a + b\n\n"
            "async def async_mult(self, a: int, b: int) -> int:\n    return a * b"
        )

        sandbox = builder.get_sandbox()
        sync_result = sandbox.sync_add(3, 4)
        assert sync_result == 7

        async_result = await sandbox.async_mult(3, 4)
        assert async_result == 12

    @pytest.mark.asyncio
    async def test_async_in_chain_levels(self):
        cls_builder = SandboxBuilder("class")
        inst_builder = SandboxBuilder("instance", base=cls_builder)
        sess_builder = SandboxBuilder("session", base=inst_builder)

        cls_builder.add_safe_builtins()
        cls_builder.add_source_code(
            "async def cls_async(self) -> str:\n    return 'from_class'"
        )
        inst_builder.add_source_code(
            "def inst_method(self) -> str:\n    return 'from_instance'"
        )

        sandbox = sess_builder.get_sandbox()
        cls_result = await sandbox.cls_async()
        assert cls_result == "from_class"
        inst_result = sandbox.inst_method()
        assert inst_result == "from_instance"

    @pytest.mark.asyncio
    async def test_async_calling_sync_in_chain(self):
        cls_builder = SandboxBuilder("class")
        inst_builder = SandboxBuilder("instance", base=cls_builder)
        sess_builder = SandboxBuilder("session", base=inst_builder)

        cls_builder.add_safe_builtins()
        cls_builder.add_source_code(
            "def get_greeting(self) -> str:\n    return 'Hello'"
        )
        inst_builder.add_source_code(
            "async def async_greet(self) -> str:\n"
            "    return self.get_greeting() + ' World'"
        )

        sandbox = sess_builder.get_sandbox()
        result = await sandbox.async_greet()
        assert result == "Hello World"
