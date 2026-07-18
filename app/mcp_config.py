"""Manage our per-conversation entries in the shared Antigravity MCP config.

The ONLY MCP config the `agy` CLI honors (verified against 1.1.4) is the shared
``~/.gemini/config/mcp_config.json`` — the same file the Antigravity IDE
manages, containing the user's own servers (and secrets). agy has no
``--mcp-config`` flag and reads this file at conversation start, so tool-turn
conversations each get their own entry added before spawn and removed at close:

    "hermes":  {"serverUrl": "http://127.0.0.1:<port>/mcp/<conv_id>"}
    "hermes2": {"serverUrl": ...}                      (concurrent conversations)

Rules:

* **Never touch foreign keys.** Read-modify-write preserves every entry we did
  not create; writes are atomic (temp file + replace).
* All mutations run under one asyncio lock (we are the only writer on our keys;
  the IDE may rewrite the file concurrently, which the re-read-before-modify
  tolerates).
* ``cleanup_stale`` removes leftover ``hermes*`` entries pointing at our port
  from a previous crashed run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agi.mcpconfig")

_SLOT_RE = re.compile(r"^hermes(\d*)$")


def default_config_path() -> Path:
    return Path("~/.gemini/config/mcp_config.json").expanduser()


class McpConfigManager:
    """Owns the ``hermes*`` server-slot entries in the shared config file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or default_config_path()
        self._lock = asyncio.Lock()
        self._owned: dict[str, str] = {}  # slot name -> conv_id

    # ── file primitives ─────────────────────────────────────────────────────

    def _read(self) -> dict:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"mcpServers": {}}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("shared MCP config %s is not valid JSON; refusing to touch it", self.path)
            raise
        if not isinstance(data, dict):
            raise ValueError(f"unexpected shared MCP config shape in {self.path}")
        data.setdefault("mcpServers", {})
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(self.path))
        except BaseException:
            with_contextlib_suppress_unlink(tmp)
            raise

    # ── slot management ─────────────────────────────────────────────────────

    def _next_slot(self, servers: dict) -> str:
        n = 1
        while True:
            name = "hermes" if n == 1 else f"hermes{n}"
            if name not in servers and name not in self._owned:
                return name
            n += 1

    async def add_conversation(self, conv_id: str, url: str) -> str:
        """Add a server entry for ``conv_id`` → returns the slot name."""
        async with self._lock:
            data = self._read()
            servers = data["mcpServers"]
            slot = self._next_slot(servers)
            servers[slot] = {"serverUrl": url}
            self._write(data)
            self._owned[slot] = conv_id
            logger.info("mcp-config: added slot %s -> %s", slot, url)
            return slot

    async def remove_conversation(self, slot: str) -> None:
        """Remove our entry for ``slot`` (foreign keys untouched)."""
        async with self._lock:
            if slot not in self._owned:
                return
            try:
                data = self._read()
            except (json.JSONDecodeError, ValueError):
                self._owned.pop(slot, None)
                return
            if slot in data["mcpServers"]:
                data["mcpServers"].pop(slot)
                self._write(data)
            self._owned.pop(slot, None)
            logger.info("mcp-config: removed slot %s", slot)

    async def cleanup_stale(self, port: int) -> int:
        """Drop leftover ``hermes*`` entries pointing at ``127.0.0.1:<port>``
        (a previous run that crashed before cleanup). Returns count removed."""
        marker = f"127.0.0.1:{port}/"
        async with self._lock:
            try:
                data = self._read()
            except (json.JSONDecodeError, ValueError):
                return 0
            servers = data["mcpServers"]
            stale = [
                name for name, spec in servers.items()
                if _SLOT_RE.match(name)
                and isinstance(spec, dict)
                and marker in str(spec.get("serverUrl") or spec.get("url") or "")
                and name not in self._owned
            ]
            for name in stale:
                servers.pop(name)
            if stale:
                self._write(data)
                logger.info("mcp-config: cleaned %d stale slot(s): %s", len(stale), stale)
            return len(stale)

    async def remove_all(self) -> None:
        """Shutdown: remove every slot we own."""
        for slot in list(self._owned):
            await self.remove_conversation(slot)


def with_contextlib_suppress_unlink(tmp: str) -> None:
    try:
        os.unlink(tmp)
    except OSError:
        pass
