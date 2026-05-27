"""Tests for OAP thread store."""

from peteos.oap._thread_store import ThreadStore


class TestThreadStore:
    def setup_method(self):
        ThreadStore.clear()

    def test_get_nonexistent(self):
        assert ThreadStore.get_session("nonexistent") is None

    def test_set_and_get(self):
        session = object()
        ThreadStore.set_session("t1", session, "agent")
        assert ThreadStore.get_session("t1") is session

    def test_destroy(self):
        ThreadStore.set_session("t1", object(), "agent")
        ThreadStore.destroy("t1")
        assert ThreadStore.get_session("t1") is None

    def test_list_ids(self):
        ThreadStore.set_session("t1", object(), "agent")
        ThreadStore.set_session("t2", object(), "agent")
        assert set(ThreadStore.list_ids()) == {"t1", "t2"}

    def test_clear(self):
        ThreadStore.set_session("t1", object(), "agent")
        ThreadStore.clear()
        assert ThreadStore.list_ids() == []
