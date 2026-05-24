"""Tests for AgenticState string key-value store."""

from peteos.session import AgenticState


class TestAgenticState:

    def test_create_and_read(self):
        state = AgenticState()
        state.create("foo", "bar")
        assert state.get("foo") == "bar"

    def test_get_returns_none_for_missing(self):
        state = AgenticState()
        assert state.get("missing") is None

    def test_create_existing_raises_valueerror(self):
        state = AgenticState()
        state.create("foo", "bar")
        try:
            state.create("foo", "baz")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "foo" in str(e)

    def test_create_empty_value_raises_valueerror(self):
        state = AgenticState()
        try:
            state.create("foo", "")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "non-empty" in str(e)

    def test_update_matching_old_value(self):
        state = AgenticState()
        state.create("count", "1")
        state.update("count", "1", "2")
        assert state.get("count") == "2"

    def test_update_nonmatching_old_value_raises_valueerror(self):
        state = AgenticState()
        state.create("count", "1")
        try:
            state.update("count", "5", "3")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "count" in str(e)

    def test_update_nonexistent_raises_keyerror(self):
        state = AgenticState()
        try:
            state.update("missing", "old", "new")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "missing" in str(e)

    def test_update_old_value_none_raises_valueerror(self):
        state = AgenticState()
        try:
            state.update("foo", None, "bar")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_update_new_value_none_raises_valueerror(self):
        state = AgenticState()
        state.create("foo", "bar")
        try:
            state.update("foo", "bar", None)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_delete_existing(self):
        state = AgenticState()
        state.create("foo", "bar")
        state.delete("foo")
        assert state.get("foo") is None

    def test_delete_nonexistent_raises_keyerror(self):
        state = AgenticState()
        try:
            state.delete("missing")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "missing" in str(e)

    def test_list_returns_empty_for_no_variables(self):
        state = AgenticState()
        assert state.list() == []

    def test_list_returns_names(self):
        state = AgenticState()
        state.create("a", "1")
        state.create("b", "2")
        state.create("c", "3")
        names = state.list()
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_list_after_delete(self):
        state = AgenticState()
        state.create("a", "1")
        state.create("b", "2")
        state.delete("a")
        assert state.list() == ["b"]

    def test_multiple_variables_independent(self):
        state = AgenticState()
        state.create("x", "1")
        state.create("y", "2")
        state.update("x", "1", "10")
        assert state.get("x") == "10"
        assert state.get("y") == "2"