"""History-prefix → agy conversation-id cache.

agy stores conversation history server-side; ``--conversation <id>`` resumes it.
OpenAI clients (hermes) resend the whole message list every request, so we map
"the history we have already played into an agy conversation" to its id:

* After a completed turn we store ``hash(request messages + [assistant reply])
  → agy conversation id``.
* The next request whose messages equal that stored transcript **plus one new
  trailing user message** hits the cache: only the new message is sent, with
  ``--conversation <id>``.
* Any mismatch (edited/compressed history, expired entry) misses and the caller
  folds the full transcript into a fresh conversation — always correct, just
  costlier.

Keys hash the normalized (role, text) sequence, so formatting-preserving changes
to unrelated fields don't break matching, while any content edit does.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from app.openai_models import ChatMessage
from app.translate import message_text


def _normalize(messages: list[ChatMessage]) -> list[tuple[str, str]]:
    return [(m.role, message_text(m)) for m in messages]


def history_key(messages: list[ChatMessage]) -> str:
    payload = json.dumps(_normalize(messages), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheHit:
    conversation_id: str
    new_user_text: str


class ConversationCache:
    """LRU + TTL map of history-prefix hash → agy conversation id."""

    def __init__(self, max_entries: int = 256, ttl_s: int = 3600) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._entries: "OrderedDict[str, tuple[str, float]]" = OrderedDict()

    def lookup(self, messages: list[ChatMessage]) -> Optional[CacheHit]:
        """Match ``messages`` = stored transcript + one trailing user message."""
        if len(messages) < 2 or messages[-1].role != "user":
            return None
        key = history_key(messages[:-1])
        entry = self._entries.get(key)
        if entry is None:
            return None
        conv_id, stored_at = entry
        if time.monotonic() - stored_at > self.ttl_s:
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        new_text = message_text(messages[-1])
        if not new_text.strip():
            return None
        return CacheHit(conversation_id=conv_id, new_user_text=new_text)

    def store(
        self,
        request_messages: list[ChatMessage],
        assistant_text: str,
        conversation_id: str,
    ) -> None:
        """Record the transcript after a completed turn."""
        if not conversation_id:
            return
        transcript = list(request_messages) + [
            ChatMessage(role="assistant", content=assistant_text)
        ]
        key = history_key(transcript)
        self._entries[key] = (conversation_id, time.monotonic())
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
