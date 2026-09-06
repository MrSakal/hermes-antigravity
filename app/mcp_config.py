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
* Mutations use an asyncio lock and a POSIX advisory sidecar lock. Cooperating
  processes serialize allocation and read-modify-replace; noncooperating IDE
  writers can still race these transactions.
* ``cleanup_stale`` removes leftover ``hermes*`` entries pointing at our port
  from a previous crashed run.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import socket
from urllib.parse import urlsplit
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

    def __init__(self, path: Optional[Path] = None, *, lock_timeout_s: float = 5.0) -> None:
        self.path = path or default_config_path()
        self.lock_timeout_s = lock_timeout_s
        self._lock = asyncio.Lock()
        self._owned: dict[str, str] = {}  # slot name -> exact owner URL
        self._leases: dict[str, int] = {}

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
        if not isinstance(data["mcpServers"], dict):
            raise ValueError(f"unexpected mcpServers shape in {self.path}")
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

    @contextlib.asynccontextmanager
    async def _transaction(self):
        # Lock a stable sidecar inode: replacing the JSON must not replace the lock.
        async with asyncio.timeout(self.lock_timeout_s), self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        await asyncio.sleep(0.01)
                # No suspension points inside the read/modify/replace transaction.
                yield
            finally:
                os.close(fd)

    def _lease(self, slot: str) -> Optional[int]:
        fd = os.open(str(self.path) + f".{slot}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        return fd

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
        async with self._transaction():
            data = self._read()
            servers = data["mcpServers"]
            slot = self._next_slot(servers)
            lease = self._lease(slot)
            while lease is None:
                # Another owner may have had its entry externally removed.
                servers = dict(servers)
                servers[slot] = None
                slot = self._next_slot(servers)
                lease = self._lease(slot)
            servers = data["mcpServers"]
            servers[slot] = {"serverUrl": url}
            try:
                self._write(data)
            except BaseException:
                os.close(lease)
                raise
            self._leases[slot] = lease
            self._owned[slot] = url
            logger.info("mcp-config: added slot %s -> %s", slot, url)
            return slot

    async def remove_conversation(self, slot: str) -> None:
        """Delete only a slot whose current URL still matches our owner URL."""
        async with self._transaction():
            if slot not in self._owned:
                return
            data = self._read()  # Refuse unreadable/corrupt data; retain ownership for retry.
            spec = data["mcpServers"].get(slot)
            if isinstance(spec, dict) and spec.get("serverUrl") == self._owned[slot]:
                del data["mcpServers"][slot]
                self._write(data)
            self._owned.pop(slot)
            os.close(self._leases.pop(slot))

    async def cleanup_stale(self, port: int) -> int:
        """Remove unleased leftovers at this exact loopback MCP endpoint.

        A listening service makes legacy entries ambiguous, so preserve them.
        Cooperating owners hold per-slot advisory leases until removal or exit.
        """
        async with self._transaction():
            data = self._read()
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                    return 0
            except OSError:
                pass
            servers = data["mcpServers"]
            stale = []
            leases = []
            try:
                for name, spec in servers.items():
                    if not _SLOT_RE.fullmatch(name) or not isinstance(spec, dict):
                        continue
                    try:
                        url = urlsplit(spec.get("serverUrl") or spec.get("url") or "")
                        matches = (url.scheme == "http" and url.hostname == "127.0.0.1"
                                   and url.port == port and url.path.startswith("/mcp/")
                                   and url.username is None and url.password is None)
                    except (ValueError, TypeError):
                        continue
                    if matches and name not in self._owned:
                        lease = self._lease(name)
                        if lease is not None:
                            leases.append(lease)
                            stale.append(name)
                for name in stale:
                    del servers[name]
                if stale:
                    self._write(data)
                return len(stale)
            finally:
                for lease in leases:
                    os.close(lease)

    async def remove_all(self) -> None:
        """Shutdown: remove every slot we own."""
        for slot in list(self._owned):
            await self.remove_conversation(slot)


def with_contextlib_suppress_unlink(tmp: str) -> None:
    try:
        os.unlink(tmp)
    except OSError:
        pass
