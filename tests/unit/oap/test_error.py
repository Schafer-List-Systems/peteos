"""Tests for OAP Error class."""

from peteos.oap.error import Error


class TestError:
    def test_error_message(self):
        e = Error("something went wrong")
        assert e.message == "something went wrong"

    def test_error_repr(self):
        e = Error("test error")
        assert repr(e) == "Error('test error')"

    def test_error_equality(self):
        e1 = Error("same message")
        e2 = Error("same message")
        e3 = Error("different message")
        assert e1 == e2
        assert e1 != e3

    def test_error_not_equal_to_non_error(self):
        e = Error("test")
        assert e != "test"
        assert e != 42
        assert e != None  # noqa: E711
