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

Keys hash full structured messages, including image content and tool metadata.
Lookup consumes the conversation atomically; a completed turn must store it again.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from app.openai_models import ChatMessage
from app.translate import message_text


def history_key(messages: list[ChatMessage], namespace: str = "") -> str:
    """Fingerprint all parsed fields; large image strings contribute SHA-256 digests.

    Exact image encodings are intentional: uncertain equivalence should miss.
    """
    def digest_images(value):
        if isinstance(value, dict):
            return {k: ({"sha256": hashlib.sha256(v.encode("utf-8")).hexdigest()}
                        if k == "url" and isinstance(v, str) else digest_images(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [digest_images(v) for v in value]
        return value

    payload = json.dumps([namespace, [digest_images(m.model_dump(mode="json"))
                                     for m in messages]],
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_transcript(messages: list[ChatMessage]) -> bool:
    # Remote images can change behind an identical URL. Unknown parts and tool
    # transcripts cannot be faithfully resumed through the plain-text interface.
    for m in messages:
        if m.tool_calls or m.tool_call_id or m.role not in {"system", "developer", "user", "assistant"}:
            return False
        if isinstance(m.content, list):
            for part in m.content:
                if not isinstance(part, dict):
                    return False
                if part.get("type") == "text":
                    if not isinstance(part.get("text"), str):
                        return False
                elif part.get("type") == "image_url":
                    image = part.get("image_url")
                    if not isinstance(image, dict) or not str(image.get("url", "")).startswith("data:"):
                        return False
                else:
                    return False
    return True


@dataclass
class CacheHit:
    conversation_id: str
    new_user_text: str


class ConversationCache:
    """LRU + TTL map of history-prefix hash → agy conversation id."""

    def __init__(self, max_entries: int = 256, ttl_s: int = 3600) -> None:
        self._lock = threading.Lock()
        self.max_entries = max(0, max_entries)
        self.ttl_s = ttl_s
        self._entries: "OrderedDict[str, tuple[str, float]]" = OrderedDict()

    def _invalidate(self, conversation_id: str) -> None:
        for key in [k for k, (cid, _) in self._entries.items() if cid == conversation_id]:
            del self._entries[key]

    def lookup(self, messages: list[ChatMessage], namespace: str = "") -> Optional[CacheHit]:
        """Atomically claim a transcript plus one plain trailing user message.

        A hit removes every entry for that mutable upstream conversation.
        Failed/cancelled continuations therefore cannot leave reusable prefixes.
        """
        if len(messages) < 2 or messages[-1].role != "user" or not _safe_transcript(messages):
            return None
        last = messages[-1]
        if last.name or (isinstance(last.content, list) and
                         any(p.get("type") != "text" for p in last.content)):
            return None
        new_text = message_text(last)
        if not new_text.strip():
            return None
        key = history_key(messages[:-1], namespace)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            conv_id, stored_at = entry
            self._invalidate(conv_id)
            if time.monotonic() - stored_at > self.ttl_s:
                return None
            return CacheHit(conversation_id=conv_id, new_user_text=new_text)

    def store(
        self,
        request_messages: list[ChatMessage],
        assistant_text: str,
        conversation_id: str,
        namespace: str = "",
    ) -> None:
        """Replace every previous prefix for this conversation after completion."""
        if not conversation_id:
            return
        transcript = list(request_messages) + [ChatMessage(role="assistant", content=assistant_text)]
        key = history_key(transcript, namespace) if _safe_transcript(transcript) else None
        with self._lock:
            self._invalidate(conversation_id)
            if key is None:
                return
            self._entries[key] = (conversation_id, time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
