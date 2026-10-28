"""One-time cleanup of the shared Antigravity MCP config (migration only).

Older builds registered every tool conversation as a ``hermes``/``hermes2``/…
entry in the SHARED ``~/.gemini/config/mcp_config.json`` — the same file the
Antigravity IDE and every other agy process read. That leaked our per-request
servers into unrelated sessions and raced the IDE's own writes. The server now
declares its MCP server inside its own agent file instead (strict, per spawn —
see :mod:`app.agent_config`) and never writes this file during operation.

What is left is a startup migration: drop leftover entries that are provably
ours. The match is deliberately narrow, because the user may well have their
own server called ``hermes``: the name must be ``hermes``/``hermesN``, the URL
host exactly ``127.0.0.1`` (what old builds wrote; any port — a crashed old run
may have used another one) and the path exactly ``/mcp/<conv id>`` in one of
the two id formats old builds minted (``conv{N}-{unix ts}``,
``conv{N}-{uuid hex}``). Before the first rewrite a one-time backup of the
original file is written next to it. Foreign entries are never touched, an
unparsable file is left alone, and the write is atomic (temp file + replace)
through the file's REAL path, so a symlinked ``mcp_config.json`` (dotfiles
repo) stays a symlink. The file is re-read right before the replace and the
rewrite is abandoned if it changed since it was read (an IDE edit in between
wins; the next startup retries).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger("agi.mcpconfig")

_SLOT_RE = re.compile(r"^hermes\d*$")
_PATH_RE = re.compile(r"^/mcp/conv\d+-[0-9a-f]{6,}$|^/mcp/conv\d+-\d{9,}$")
BACKUP_SUFFIX = ".hermes-antigravity.bak"


def default_config_path() -> Path:
    return Path("~/.gemini/config/mcp_config.json").expanduser()


def _is_ours(name: str, spec: Any) -> bool:
    if not _SLOT_RE.match(name) or not isinstance(spec, dict):
        return False
    url = str(spec.get("serverUrl") or spec.get("url") or "")
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError:
        return False
    return (
        parts.scheme == "http"
        and parts.hostname == "127.0.0.1"
        and bool(_PATH_RE.match(parts.path))
        and not parts.query
    )


def legacy_entries_remaining(path: Optional[Path] = None) -> Optional[bool]:
    """False once the shared config holds none of our legacy entries (or does
    not exist) — the migration is complete and its backup may go; True while
    some remain; None when the file cannot be read or parsed (unknown)."""
    path = Path(path).expanduser() if path else default_config_path()
    try:
        raw = Path(os.path.realpath(path)).read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        return False
    return any(_is_ours(name, spec) for name, spec in servers.items())


def cleanup_legacy_mcp_entries(path: Optional[Path] = None) -> int:
    """Remove our legacy slot entries from the shared config. Returns the count
    removed; never raises."""
    path = Path(path).expanduser() if path else default_config_path()
    # Write through symlinks: replacing the link itself would turn it into a
    # regular file and silently fork the user's config.
    real = Path(os.path.realpath(path))
    try:
        raw = real.read_text(encoding="utf-8")
    except OSError:
        return 0
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("shared MCP config %s is not valid JSON; leaving it alone", path)
        return 0
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        return 0
    stale = [name for name, spec in servers.items() if _is_ours(name, spec)]
    if not stale:
        return 0
    # This file is shared with the IDE and holds the user's own servers (and
    # their secrets): keep the original once, so a wrong match is recoverable.
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        try:
            backup.write_text(raw, encoding="utf-8")
        except OSError:
            logger.warning("could not back up shared MCP config to %s; not rewriting it",
                           backup, exc_info=True)
            return 0
    for name in stale:
        servers.pop(name)
    try:
        fd, tmp = tempfile.mkstemp(dir=str(real.parent), prefix=".agi-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Narrow the race with the IDE: if the file changed since it was
            # read, our rewrite would drop that change — leave it alone.
            if real.read_text(encoding="utf-8") != raw:
                os.unlink(tmp)
                logger.warning("shared MCP config %s changed while cleaning it; "
                               "leaving it alone (retried next start)", path)
                return 0
            os.replace(tmp, str(real))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        logger.warning("could not rewrite shared MCP config %s", path, exc_info=True)
        return 0
    logger.info("mcp-config: removed %d legacy hermes slot(s) from %s: %s (backup: %s)",
                len(stale), path, stale, backup)
    return len(stale)
