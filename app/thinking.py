"""Thinking passthrough — surface agy's reasoning as ``reasoning_content``.

agy's ``--output-format stream-json`` does NOT emit the model's thinking text
(verified against 1.1.4: thinking steps surface as an empty ``unknown`` step;
only ``thinking_tokens`` shows up in usage). But the CLI writes the FULL
thinking text to its internal conversation transcript,

    ~/.gemini/antigravity-cli/brain/<conversation_id>/
        .system_generated/logs/transcript_full.jsonl

as a ``"thinking"`` field on the model-response entries — flushed exactly when
the response starts streaming (measured live: same tick as the first
``text_delta``). So the turn loop polls that file and emits each new thinking
block as an OpenAI-style ``delta.reasoning_content`` chunk right before the
answer text. There is no incremental thinking stream — blocks arrive whole,
one per model step (multi-step tool turns yield one per step).

Everything here is best-effort on an UNDOCUMENTED internal file: a missing
directory, a renamed field, or a format change must degrade to "no reasoning
surfaced", never to an error.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agi.thinking")


def default_brain_path() -> Path:
    return Path("~/.gemini/antigravity-cli/brain").expanduser()


class ThinkingReader:
    """Incrementally read new ``"thinking"`` blocks from one conversation's
    ``transcript_full.jsonl``.

    ``skip_existing=True`` snapshots the current file size so a resumed
    conversation only surfaces thinking produced from now on; a fresh
    conversation starts at 0 (the file does not exist yet anyway).
    """

    def __init__(self, brain_path: Path, conversation_id: str, *, skip_existing: bool) -> None:
        self.path = (
            brain_path / conversation_id / ".system_generated" / "logs" / "transcript_full.jsonl"
        )
        self._offset = 0
        self._seen: set[str] = set()
        if skip_existing:
            try:
                self._offset = self.path.stat().st_size
            except OSError:
                self._offset = 0

    def poll(self) -> list[str]:
        """Return thinking blocks appended since the last poll (possibly [])."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._offset:
            # File was rewritten shorter — re-read from the start; the seen-set
            # keeps already-emitted blocks from repeating.
            self._offset = 0
        if size == self._offset:
            return []
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read(size - self._offset)
        except OSError:
            return []
        # Consume only complete lines; a partially-written tail stays for later.
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self._offset += end + 1
        out: list[str] = []
        for raw in chunk[: end + 1].splitlines():
            try:
                v = json.loads(raw.decode("utf-8", "replace"))
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(v, dict):
                continue
            thinking = v.get("thinking")
            if not isinstance(thinking, str) or not thinking.strip():
                continue
            key = hashlib.sha1(thinking.encode("utf-8")).hexdigest()
            if key in self._seen:
                continue
            self._seen.add(key)
            out.append(thinking)
        return out


class TurnThinking:
    """Per-turn reader lifecycle: created before spawn, bound to the agy
    conversation either upfront (resume — the id is already known) or from the
    ``init`` stream event (fresh conversation)."""

    def __init__(self, brain_path: Path, conversation_id: Optional[str] = None) -> None:
        self._brain = brain_path
        self._reader: Optional[ThinkingReader] = None
        if conversation_id:
            # Resumed conversation: the transcript already holds earlier turns'
            # thinking — surface only what this turn produces.
            self._reader = ThinkingReader(brain_path, conversation_id, skip_existing=True)

    def on_init(self, conversation_id: Optional[str]) -> None:
        if self._reader is None and conversation_id:
            self._reader = ThinkingReader(self._brain, conversation_id, skip_existing=False)

    def poll(self) -> list[str]:
        if self._reader is None:
            return []
        return self._reader.poll()
