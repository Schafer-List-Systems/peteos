"""Tests for OAP sandbox."""

import pytest

from peteos.oap.sandbox import create_sandbox_globals


class TestSandbox:
    def test_safe_builtins(self):
        sandbox = create_sandbox_globals(object())
        assert sandbox["__builtins__"] is not {}
        assert sandbox["__builtins__"]["print"] is print
        assert sandbox["__builtins__"]["str"] is str
        assert sandbox["__builtins__"]["list"] is list
        # File access should still be blocked
        with pytest.raises((NameError, ImportError)):
            exec("open('/dev/null')", sandbox)

    def test_self_is_object(self):
        obj = object()
        sandbox = create_sandbox_globals(obj)
        assert sandbox["this"] is obj

    def test_imports_injected(self):
        import os

        sandbox = create_sandbox_globals(object(), imports=[os])
        assert "os" in sandbox
        assert sandbox["os"] is os

    def test_import_not_available(self):
        sandbox = create_sandbox_globals(object())
        with pytest.raises(ImportError):
            exec("import sys", sandbox)

    def test_no_file_access(self):
        sandbox = create_sandbox_globals(object())
        with pytest.raises((NameError, ImportError)):
            exec("f = open('/etc/passwd')", sandbox)
