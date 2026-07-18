from __future__ import annotations

from app.convcache import ConversationCache
from app.openai_models import ChatMessage


def msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


def test_store_then_hit():
    cache = ConversationCache()
    history = [msg("system", "s"), msg("user", "q1")]
    cache.store(history, "a1", "agy-conv-1")

    nxt = history + [msg("assistant", "a1"), msg("user", "q2")]
    hit = cache.lookup(nxt)
    assert hit is not None
    assert hit.conversation_id == "agy-conv-1"
    assert hit.new_user_text == "q2"


def test_miss_on_edited_history():
    cache = ConversationCache()
    history = [msg("user", "q1")]
    cache.store(history, "a1", "agy-conv-1")

    edited = [msg("user", "q1-EDITED"), msg("assistant", "a1"), msg("user", "q2")]
    assert cache.lookup(edited) is None


def test_miss_when_trailing_not_user():
    cache = ConversationCache()
    history = [msg("user", "q1")]
    cache.store(history, "a1", "c1")
    assert cache.lookup(history + [msg("assistant", "a1")]) is None


def test_ttl_expiry(monkeypatch):
    cache = ConversationCache(ttl_s=10)
    history = [msg("user", "q1")]
    cache.store(history, "a1", "c1")

    import app.convcache as cc
    real = cc.time.monotonic
    monkeypatch.setattr(cc.time, "monotonic", lambda: real() + 11)
    assert cache.lookup(history + [msg("assistant", "a1"), msg("user", "q2")]) is None


def test_lru_eviction():
    cache = ConversationCache(max_entries=2)
    for i in range(3):
        cache.store([msg("user", f"q{i}")], f"a{i}", f"c{i}")
    assert len(cache) == 2
    # Oldest (q0) evicted.
    assert cache.lookup([msg("user", "q0"), msg("assistant", "a0"), msg("user", "x")]) is None
    assert cache.lookup([msg("user", "q2"), msg("assistant", "a2"), msg("user", "x")]) is not None


def test_no_store_without_conv_id():
    cache = ConversationCache()
    cache.store([msg("user", "q")], "a", "")
    assert len(cache) == 0
