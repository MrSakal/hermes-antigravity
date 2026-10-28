"""Zero-footprint housekeeping: an ownership ledger for everything the server
leaves on disk, and the deletion of it as soon as it can no longer be used.

agy has no retention of its own: every process persists, forever, under
``~/.gemini/antigravity-cli`` (``acli``) — ``conversations/<uuid>.db``
(+ ``-wal``/``-shm``), ``presence/<uuid>.lock``, ``brain/<uuid>/``,
``annotations/<uuid>.pbtxt``, ``implicit/<uuid>.pb``, one ``log/cli-*.log`` per
process — and lists the conversation in the shared ``cache/
conversation_metadata.json`` / ``cache/last_conversations.json``. The same
directory holds the USER's own agy / IDE conversations, which are
indistinguishable by name or age. So nothing is ever deleted by pattern or
age: only what the ledger proves we created.

**Ledger.** A small SQLite file in our own state dir (Linux
``$XDG_STATE_HOME/hermes-antigravity`` or ``~/.local/state/hermes-antigravity``;
Windows ``%LOCALAPPDATA%\\hermes-antigravity``; ``AGI_STATE_DIR`` overrides)
records every agy conversation uuid one of our processes announced in its
``init`` line (warm-pool boots included — the line is printed at boot), every
spawned process (pid + OS start time + its ``--log-file``) and every workdir
used. All ledger I/O and file deletion runs on ONE worker thread, never on the
event loop. Rows carry the id of the server instance that wrote them; each
instance holds an OS lock on ``run/<instance>.lock`` for its lifetime, so a
second server on the same state dir never touches the first one's rows, and a
crashed instance is recognisable (its lock can be taken).

**Per-process logs.** Every spawn (and every ``agy models`` refresh) runs with
``--log-file <state>/logs/<spawn>.log`` (verified on 1.4.8: honored, nothing is
written to ``acli/log``), deleted when the process ends — kept only for a
failed process when ``AGI_KEEP_AGY_LOGS=true`` (then swept after 7 days).

**Prompt deletion.** A ledger conversation is deleted once it can no longer
be used: no live process of ours has it open, and the conversation manager
does not reference it (a live, parked or closing process). The server never
resumes an exited process' conversation (a follow-up starts fresh with the
full history), so it goes right after its process exits — idle TTL, crash,
eviction, an unadopted warm-pool boot's discard, shutdown — and, at the next
start, the leftovers of a crashed run (its surviving agy processes are killed first —
pid + start time, so a recycled pid is never hit). Deleted per uuid, exactly:
``conversations/<uuid>.db{,-wal,-shm}``, ``presence/<uuid>.lock``,
``brain/<uuid>/``, ``annotations/<uuid>.pbtxt``, ``implicit/<uuid>.pb``, plus
the uuid's entries in the two shared cache JSON files — rewritten atomically,
other entries untouched, and skipped when the file does not parse, has an
unexpected shape, is held open (Windows) or changed while we rewrote it.
``conversation_summaries.db`` (agy's shared SQLite) is never touched. A
``PermissionError`` (Windows: a file still held for a moment) is retried
briefly and otherwise left for the next sweep; the ledger row stays until the
files are gone.

**Sweep.** :meth:`Housekeeper.sweep` (startup + every
``AGI_HOUSEKEEPING_INTERVAL_S``) re-runs the above for ledger entries whose
files still exist, recovers crashed instances, and removes our other cruft:
orphan per-spawn agent files and ``*.tmp`` in the managed plugin dir, our MCP
schema caches (``acli/mcp/hermes-<8hex>/``, legacy ``acli/mcp/hermes/``)
unused for ``AGI_MCP_CACHE_MAX_AGE_S``, crash-left ``.agi-*`` temp files in
attachment dirs and next to the shared MCP config, stale logs, the legacy
``mcp_config.json.hermes.lock`` and — once the legacy MCP-config migration
has completed — its ``.hermes-antigravity.bak``. ``python -m app.purge``
removes the whole footprint (see app/purge.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("agi.housekeeping")

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
MCP_CACHE_RE = re.compile(r"^hermes-[0-9a-f]{8}$")
LEGACY_MCP_CACHE = "hermes"
_LOG_RE = re.compile(r"^(?:spawn|models)-[0-9a-f]{12}\.log$")
_ATTACHMENT_RE = re.compile(r"^img-[0-9a-f]{16}\.(png|jpg|gif|webp)$")
TEMP_PREFIX = ".agi-"

LEDGER_NAME = "ledger.sqlite3"
MIGRATION_DONE_KEY = "mcp_migration_done"
# Temp files younger than this may belong to a write in progress.
_TEMP_MIN_AGE_S = 3600.0
# Logs nobody owns any more (crash between spawn and exit, a models refresh
# that died with the server) older than this are removed.
_ORPHAN_LOG_AGE_S = 600.0
_KEPT_LOG_AGE_S = 7 * 86400.0
# Shared-cache rewrites refused this many times (agy keeps the file busy /
# rewrites it) are given up on, so a ledger row cannot live forever.
_CACHE_ATTEMPTS = 5
_DELETE_RETRIES = 3
_DELETE_BACKOFF_S = 0.2
_STOP_ATTEMPTS = 5
_STOP_BACKOFF_S = 1.0


# ── locations ───────────────────────────────────────────────────────────────


def default_state_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "hermes-antigravity"
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "hermes-antigravity"


def default_agy_root() -> Path:
    return Path("~/.gemini/antigravity-cli").expanduser()


def resolve_state_dir(settings: Any) -> Path:
    p = getattr(settings, "state_dir", None)
    return Path(p).expanduser() if p else default_state_dir()


def resolve_agy_root(settings: Any) -> Path:
    p = getattr(settings, "agy_state_root", None)
    return Path(p).expanduser() if p else default_agy_root()


# ── processes (pid + start time: never kill a recycled pid) ────────────────


def process_start_time(pid: int) -> Optional[str]:
    """An opaque token for ``pid``'s start time, or None when it is not
    running / cannot be read. Linux: /proc starttime; Windows: creation
    FILETIME (ctypes)."""
    if pid <= 0:
        return None
    if os.name == "nt":
        return _win_start_time(pid)
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            fields = f.read().rsplit(b")", 1)[1].split()
    except (OSError, IndexError):
        return None
    # After "(comm)": state(3) ppid pgrp ... starttime is field 22 overall.
    if len(fields) < 20 or fields[0] == b"Z":
        return None
    return fields[19].decode()


def _win_start_time(pid: int) -> Optional[str]:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
            return None  # STILL_ACTIVE
        ft = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(handle, *[ctypes.byref(x) for x in ft]):
            return None
        return str((ft[0].dwHighDateTime << 32) | ft[0].dwLowDateTime)
    finally:
        k32.CloseHandle(handle)


def kill_process_tree(pid: int) -> None:
    """Kill ``pid`` and its descendants (the caller verified pid + start time).
    POSIX: our agy processes lead their own process group."""
    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return
    import signal

    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def kill_if_same(pid: int, start: Optional[str], wait_s: float = 5.0) -> bool:
    """Kill ``pid`` only if it is still the process that started at ``start``.
    Returns True when that process is gone afterwards."""
    if not start or process_start_time(pid) != start:
        return True  # gone (or a different process now owns the number)
    logger.warning("housekeeping: killing leftover agy process %d from a crashed run", pid)
    kill_process_tree(pid)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if process_start_time(pid) != start:
            return True
        time.sleep(0.1)
    return False


# ── instance locks ──────────────────────────────────────────────────────────


def _lock_fd(fd: int) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)


class InstanceLock:
    """``<state>/run/<instance>.lock``, OS-locked while the server runs."""

    def __init__(self, run_dir: Path, instance: str) -> None:
        self.path = run_dir / f"{instance}.lock"
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        if not _lock_fd(fd):
            os.close(fd)
            raise OSError(f"cannot lock {self.path}")
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        _unlock_fd(self._fd)
        with contextlib.suppress(OSError):
            os.close(self._fd)
        self._fd = None
        with contextlib.suppress(OSError):
            self.path.unlink()


def instance_alive(run_dir: Path, instance: str) -> bool:
    """True while a server holds ``instance``'s lock. A missing lock file, or
    one that can be locked, means that instance is gone."""
    path = run_dir / f"{instance}.lock"
    try:
        fd = os.open(str(path), os.O_RDWR)
    except FileNotFoundError:
        return False
    except OSError:
        return True  # cannot tell: assume alive (never touch its rows)
    try:
        if _lock_fd(fd):
            _unlock_fd(fd)
            return False
        return True
    finally:
        os.close(fd)


def live_instances(run_dir: Path) -> list[str]:
    """Instances whose lock is held right now (stale lock files are removed)."""
    out = []
    try:
        names = [p.name for p in run_dir.iterdir() if p.name.endswith(".lock")]
    except OSError:
        return out
    for name in names:
        inst = name[:-5]
        if instance_alive(run_dir, inst):
            out.append(inst)
        else:
            with contextlib.suppress(OSError):
                (run_dir / name).unlink()
    return out


# ── ledger ──────────────────────────────────────────────────────────────────


class Ledger:
    """SQLite ledger. Not thread-safe by itself: the server confines it to one
    worker thread; app.purge uses it synchronously."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._con: Optional[sqlite3.Connection] = None

    def open(self) -> "Ledger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.path), timeout=10, isolation_level=None,
                              check_same_thread=False)
        con.execute("PRAGMA journal_mode=DELETE")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS convs (
                uuid TEXT PRIMARY KEY, instance TEXT NOT NULL, root TEXT NOT NULL,
                created REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS procs (
                spawn TEXT PRIMARY KEY, instance TEXT NOT NULL, pid INTEGER,
                pstart TEXT, log TEXT, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS workdirs (
                path TEXT PRIMARY KEY, last_used REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        self._con = con
        return self

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    @property
    def con(self) -> sqlite3.Connection:
        assert self._con is not None, "ledger not open"
        return self._con

    # conversations
    def add_conv(self, uuid: str, instance: str, root: str) -> None:
        self.con.execute("INSERT OR IGNORE INTO convs(uuid, instance, root, created) "
                         "VALUES (?,?,?,?)", (uuid, instance, root, time.time()))

    def convs(self, instance: Optional[str] = None) -> list[tuple[str, str, str, int]]:
        q = "SELECT uuid, instance, root, attempts FROM convs"
        rows = self.con.execute(q + (" WHERE instance=?" if instance else ""),
                                (instance,) if instance else ()).fetchall()
        return [tuple(r) for r in rows]

    def remove_conv(self, uuid: str) -> None:
        self.con.execute("DELETE FROM convs WHERE uuid=?", (uuid,))

    def bump_conv(self, uuid: str) -> int:
        self.con.execute("UPDATE convs SET attempts=attempts+1 WHERE uuid=?", (uuid,))
        row = self.con.execute("SELECT attempts FROM convs WHERE uuid=?", (uuid,)).fetchone()
        return int(row[0]) if row else 0

    def adopt(self, from_instance: str, to_instance: str) -> None:
        self.con.execute("UPDATE convs SET instance=? WHERE instance=?", (to_instance, from_instance))

    # processes
    def add_proc(self, spawn: str, instance: str, pid: Optional[int], pstart: Optional[str],
                 log: Optional[str]) -> None:
        self.con.execute("INSERT OR REPLACE INTO procs(spawn, instance, pid, pstart, log, created) "
                         "VALUES (?,?,?,?,?,?)", (spawn, instance, pid, pstart, log, time.time()))

    def procs(self, instance: Optional[str] = None) -> list[tuple[str, str, Optional[int], Optional[str], Optional[str]]]:
        q = "SELECT spawn, instance, pid, pstart, log FROM procs"
        rows = self.con.execute(q + (" WHERE instance=?" if instance else ""),
                                (instance,) if instance else ()).fetchall()
        return [tuple(r) for r in rows]

    def remove_proc(self, spawn: str) -> None:
        self.con.execute("DELETE FROM procs WHERE spawn=?", (spawn,))

    def instances(self) -> set[str]:
        rows = self.con.execute("SELECT instance FROM convs UNION SELECT instance FROM procs").fetchall()
        return {r[0] for r in rows}

    # workdirs
    def add_workdir(self, path: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO workdirs(path, last_used) VALUES (?,?)",
                         (path, time.time()))

    def workdirs(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT path FROM workdirs").fetchall()]

    # meta
    def get_meta(self, key: str) -> Optional[str]:
        row = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))


# ── file helpers ────────────────────────────────────────────────────────────


def _unlink(p: Path, retries: int = _DELETE_RETRIES) -> bool:
    """True once ``p`` is gone. PermissionError (Windows: still held open) is
    retried briefly; any other failure leaves it for the next sweep."""
    for attempt in range(max(1, retries)):
        try:
            p.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt + 1 < retries:
                time.sleep(_DELETE_BACKOFF_S)
        except OSError:
            return False
    return False


def _rmtree(p: Path, retries: int = _DELETE_RETRIES) -> bool:
    """Remove a real directory tree (a symlink is unlinked, never followed)."""
    if p.is_symlink():
        return _unlink(p, retries)
    if not p.exists():
        return True
    for attempt in range(max(1, retries)):
        shutil.rmtree(p, ignore_errors=True)
        if not p.exists():
            return True
        if attempt + 1 < retries:
            time.sleep(_DELETE_BACKOFF_S)
    return False


def _newest_mtime(p: Path) -> float:
    try:
        newest = p.lstat().st_mtime
    except OSError:
        return 0.0
    if p.is_dir() and not p.is_symlink():
        for child in p.rglob("*"):
            with contextlib.suppress(OSError):
                newest = max(newest, child.lstat().st_mtime)
    return newest


def _age(p: Path, now: Optional[float] = None) -> float:
    try:
        return (now or time.time()) - p.lstat().st_mtime
    except OSError:
        return 0.0


def conversation_files(root: Path, uuid: str) -> tuple[list[Path], Path]:
    """(files, brain dir) agy keeps for conversation ``uuid``."""
    conv = root / "conversations"
    files = [conv / f"{uuid}.db", conv / f"{uuid}.db-wal", conv / f"{uuid}.db-shm",
             root / "presence" / f"{uuid}.lock", root / "annotations" / f"{uuid}.pbtxt",
             root / "implicit" / f"{uuid}.pb"]
    return files, root / "brain" / uuid


def conversation_exists(root: Path, uuid: str) -> bool:
    files, brain = conversation_files(root, uuid)
    return any(os.path.lexists(p) for p in files) or os.path.lexists(brain)


def delete_conversation_state(root: Path, uuid: str, retries: int = _DELETE_RETRIES) -> bool:
    """Delete agy's per-conversation files of ``uuid``. True once all are
    gone. Refuses anything that is not a canonical uuid (no path games)."""
    if not UUID_RE.match(uuid):
        raise ValueError(f"not a conversation uuid: {uuid!r}")
    files, brain = conversation_files(root, uuid)
    ok = True
    for p in files:
        ok &= _unlink(p, retries)
    ok &= _rmtree(brain, retries)
    return ok


def delete_crash_markers(root: Path, pid: int, before: Optional[float] = None,
                         after: Optional[float] = None) -> int:
    """agy creates ``crashes/crash_<pid>_<uuid>.log`` at every start (empty
    unless it crashed; verified on 1.4.8) and leaves it behind even on a clean
    exit — the next agy start deletes the ones of dead pids. Called once OUR
    process ``pid`` has been reaped; only markers written between ``after``
    (just before our spawn) and ``before`` (the reap) count, so the stale
    marker of an older, unrelated process with the same pid number, or one of
    a newer process that reused it, is never touched."""
    pat = re.compile(rf"^crash_{int(pid)}_[0-9a-f-]{{36}}\.log$")
    removed = 0
    try:
        entries = list((root / "crashes").iterdir())
    except OSError:
        return 0
    for p in entries:
        if not pat.match(p.name) or p.is_symlink():
            continue
        if before is not None and _age(p, before) < 0:
            continue  # written after our process was gone: not ours
        if after is not None and _age(p, after) > 0:
            continue  # older than our spawn: not ours
        if _unlink(p, 1):
            removed += 1
    return removed


def _rewrite_json(path: Path, transform: Callable[[Any], Any]) -> bool:
    """Atomically rewrite a shared JSON file through ``transform`` (returns the
    new object, or None for "leave it alone"). True = done / nothing to do /
    permanently skipped (unparsable, unexpected shape); False = transient
    failure (held open, changed under us) — retry on the next sweep."""
    real = Path(os.path.realpath(path))
    try:
        raw = real.read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning("housekeeping: %s does not parse; leaving it alone", path)
        return True
    new = transform(data)
    if new is None:
        return True
    text = json.dumps(new, ensure_ascii=False, indent=2) + ("\n" if raw.endswith("\n") else "")
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(real.parent), prefix=TEMP_PREFIX, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        if real.read_text(encoding="utf-8") != raw:
            return False  # agy rewrote it meanwhile: its write wins
        os.replace(tmp, str(real))
        tmp = None
        return True
    except OSError:
        return False
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def scrub_shared_caches(root: Path, uuids: Iterable[str]) -> bool:
    """Remove ``uuids`` from agy's ``cache/last_conversations.json`` (cwd →
    uuid) and ``cache/conversation_metadata.json`` (``conversations`` map).
    Every other entry is kept as is. True when both are done."""
    ids = {u.lower() for u in uuids if UUID_RE.match(u)}
    if not ids:
        return True

    def last(data: Any) -> Any:
        if not isinstance(data, dict):
            return None
        keep = {k: v for k, v in data.items() if not (isinstance(v, str) and v.lower() in ids)}
        return keep if len(keep) != len(data) else None

    def meta(data: Any) -> Any:
        convs = data.get("conversations") if isinstance(data, dict) else None
        if not isinstance(convs, dict):
            return None
        keep = {k: v for k, v in convs.items() if k.lower() not in ids}
        if len(keep) == len(convs):
            return None
        return {**data, "conversations": keep}

    cache = root / "cache"
    ok = _rewrite_json(cache / "last_conversations.json", last)
    ok &= _rewrite_json(cache / "conversation_metadata.json", meta)
    return ok


def user_mcp_server_names(mcp_config: Optional[Path]) -> set[str]:
    """Server names in the user's shared MCP config (their caches are never
    ours). Unreadable → {"hermes"} so the legacy name stays protected."""
    if mcp_config is None:
        return set()
    try:
        data = json.loads(Path(mcp_config).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except (OSError, ValueError):
        return {LEGACY_MCP_CACHE}
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    return set(servers) if isinstance(servers, dict) else set()


def sweep_mcp_caches(root: Path, *, max_age_s: float, in_use: Iterable[str] = (),
                     user_servers: Iterable[str] = (), everything: bool = False,
                     dry_run: bool = False) -> int:
    """Our per-toolset MCP schema caches (``hermes-<8hex>``, legacy
    ``hermes``) unused for ``max_age_s`` (``everything``: regardless of age).
    Names the user configured themselves, or a live conversation uses, are
    never touched. A live process whose cache vanished simply re-lists."""
    busy = set(in_use) | set(user_servers)
    removed = 0
    horizon = time.time() - max_age_s
    try:
        entries = list((root / "mcp").iterdir())
    except OSError:
        return 0
    for d in entries:
        name = d.name
        if not (MCP_CACHE_RE.match(name) or name == LEGACY_MCP_CACHE) or name in busy:
            continue
        if d.is_symlink() or not d.is_dir():
            continue
        if not everything and _newest_mtime(d) > horizon:
            continue
        if dry_run or _rmtree(d):
            removed += 1
    return removed


def sweep_temp_files(directory: Path, *, pattern: Callable[[str], bool],
                     min_age_s: float = _TEMP_MIN_AGE_S) -> int:
    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    now = time.time()
    for p in entries:
        if not pattern(p.name) or p.is_symlink() or not p.is_file():
            continue
        if _age(p, now) < min_age_s:
            continue
        if _unlink(p, 1):
            removed += 1
    return removed


def _is_agi_temp(name: str) -> bool:
    return name.startswith(TEMP_PREFIX)


def purge_attachments(workdir: Path, dry_run: bool = False) -> int:
    """Remove our attachment files (and crash-left temps) under
    ``workdir/attachments`` and the dir itself when empty."""
    d = Path(workdir) / "attachments"
    if d.is_symlink() or not d.is_dir():
        return 0
    removed = 0
    for p in list(d.iterdir()):
        if (_ATTACHMENT_RE.match(p.name) or _is_agi_temp(p.name)) and p.is_file() \
                and not p.is_symlink():
            removed += dry_run or _unlink(p)
    if not dry_run:
        with contextlib.suppress(OSError):
            d.rmdir()
    return removed


# ── the server-side housekeeper ─────────────────────────────────────────────


class Housekeeper:
    """Owns the ledger and the deletion policy (see module docstring).

    AgySession calls :meth:`log_file_for` / :meth:`on_spawn` /
    :meth:`on_init` / :meth:`on_exit`; the conversation manager registers
    :attr:`references` (agy ids its processes still use) and pokes after it drops
    one. Everything that touches disk runs on :attr:`_ex` (one thread)."""

    def __init__(self, settings: Any, *, plugin_path: Optional[Path] = None,
                 agent_name: str = "hermes") -> None:
        self.settings = settings
        self.enabled = bool(getattr(settings, "housekeeping", True))
        self.state_dir = resolve_state_dir(settings)
        self.root = resolve_agy_root(settings)
        self.logs_dir = self.state_dir / "logs"
        self.run_dir = self.state_dir / "run"
        self.keep_logs = bool(getattr(settings, "keep_agy_logs", False))
        self.plugin_path = Path(plugin_path) if plugin_path else None
        self.agent_name = agent_name
        mcp = getattr(settings, "mcp_config_path", None)
        self.mcp_config_path = Path(mcp).expanduser() if mcp else \
            Path("~/.gemini/config/mcp_config.json").expanduser()
        self.instance = f"{os.getpid()}-{secrets.token_hex(4)}"
        self._ledger = Ledger(self.state_dir / LEDGER_NAME)
        self._lock = InstanceLock(self.run_dir, self.instance)
        self._ex: Optional[ThreadPoolExecutor] = None
        self.active = False
        # uuid -> sessions of ours that have it open (until their exit).
        self._live: dict[str, set[Any]] = {}
        # uuids this instance owns per the ledger (mirror, event-loop side).
        self._owned: set[str] = set()
        # agy ids the conversation manager's processes still use.
        self.references: Optional[Callable[[], Iterable[str]]] = None
        # MCP server names live conversations use (never swept).
        self.servers_in_use: Optional[Callable[[], Iterable[str]]] = None
        # Workdirs to add to the attachment store's GC set.
        self.on_workdirs: Optional[Callable[[list[Path]], None]] = None
        self._workdirs_seen: set[str] = set()
        self._reconcile_lock = asyncio.Lock()
        self._poke_task: Optional[asyncio.Task] = None
        self._pending: set[asyncio.Future] = set()
        self.stats = {"deleted": 0, "killed": 0, "logs_deleted": 0, "swept": 0}

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self.enabled or self.active:
            return
        self._ex = ThreadPoolExecutor(1, thread_name_prefix="agi-housekeeping")
        try:
            await self._run(self._start_sync)
        except Exception:
            logger.warning("housekeeping disabled: cannot use state dir %s", self.state_dir,
                           exc_info=True)
            self._ex.shutdown(wait=False)
            self._ex = None
            return
        self.active = True
        await self.sweep()

    def _start_sync(self) -> None:
        for d in (self.state_dir, self.logs_dir, self.run_dir):
            d.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(d, 0o700)
        self._lock.acquire()
        try:
            self._ledger.open()
        except BaseException:
            self._lock.release()
            raise

    async def stop(self) -> None:
        """Shutdown (after the pool and every conversation closed): delete
        every conversation this instance owns, then release the state lock.
        What could not be deleted stays in the ledger for the next start."""
        if not self.active:
            return
        try:
            for fut in list(self._pending):
                with contextlib.suppress(Exception):
                    await fut
            if self._poke_task is not None:
                self._poke_task.cancel()
                with contextlib.suppress(BaseException):
                    await self._poke_task
            # Windows: a just-exited agy's files may stay held for a moment.
            for attempt in range(_STOP_ATTEMPTS):
                await self.reconcile(final=True)
                if not any(not self._live.get(u) for u in self._owned):
                    break
                await asyncio.sleep(_STOP_BACKOFF_S)
            if self._owned:
                logger.warning("housekeeping: %d conversation(s) still held at shutdown; "
                               "deleted at the next start", len(self._owned))
            await self._run(self._sweep_logs_sync)
        finally:
            self.active = False
            ex = self._ex
            if ex is not None:
                with contextlib.suppress(Exception):
                    await self._run(self._close_sync)
                self._ex = None
                ex.shutdown(wait=True)

    def _close_sync(self) -> None:
        self._ledger.close()
        self._lock.release()

    async def _run(self, fn: Callable, *args: Any) -> Any:
        assert self._ex is not None
        return await asyncio.get_running_loop().run_in_executor(self._ex, fn, *args)

    def _submit(self, fn: Callable, *args: Any) -> None:
        """Fire-and-forget ledger write (ordered: one worker thread)."""
        if self._ex is None:
            return
        fut = asyncio.wrap_future(self._ex.submit(self._guard, fn, *args))
        self._pending.add(fut)
        fut.add_done_callback(self._pending.discard)

    @staticmethod
    def _guard(fn: Callable, *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception:
            logger.warning("housekeeping: %s failed", getattr(fn, "__name__", fn), exc_info=True)
            return None

    async def flush(self) -> None:
        """Wait for queued ledger writes (tests / shutdown)."""
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    # ── session hooks (event loop; never block) ─────────────────────────────

    def log_file_for(self, session: Any) -> Optional[str]:
        if not self.active:
            return None
        spawn = f"spawn-{secrets.token_hex(6)}"
        session._hk_spawn = spawn
        session._hk_uuids = set()
        session._hk_spawned_wall = time.time() - 2.0  # (coarse mtime resolution)
        return str(self.logs_dir / f"{spawn}.log")

    def on_spawn(self, session: Any, pid: Optional[int], log_file: Optional[str]) -> None:
        spawn = getattr(session, "_hk_spawn", None)
        if not self.active or spawn is None:
            return
        self._submit(self._spawn_sync, spawn, pid, log_file)
        self.note_workdir(getattr(session, "workdir", None))

    def _spawn_sync(self, spawn: str, pid: Optional[int], log_file: Optional[str]) -> None:
        pstart = process_start_time(pid) if pid else None
        self._ledger.add_proc(spawn, self.instance, pid, pstart, log_file)

    def on_init(self, session: Any, uuid: Optional[str]) -> None:
        if not self.active or not uuid:
            return
        uuid = uuid.lower()
        if not UUID_RE.match(uuid):
            return
        uuids = getattr(session, "_hk_uuids", None)
        if uuids is None:
            uuids = session._hk_uuids = set()
        uuids.add(uuid)
        self._live.setdefault(uuid, set()).add(session)
        if uuid not in self._owned:
            self._owned.add(uuid)
            self._submit(self._ledger.add_conv, uuid, self.instance, str(self.root))

    def on_exit(self, session: Any, *, failed: bool = False) -> None:
        """The process is gone (reaped): its conversation may now be deleted
        (if unreferenced) and its log is removed."""
        for uuid in getattr(session, "_hk_uuids", ()) or ():
            holders = self._live.get(uuid)
            if holders is not None:
                holders.discard(session)
                if not holders:
                    self._live.pop(uuid, None)
        spawn = getattr(session, "_hk_spawn", None)
        if spawn is None or not self.active:
            return
        log_file = str(self.logs_dir / f"{spawn}.log")
        keep = self.keep_logs and failed
        self._submit(self._exit_sync, spawn, log_file, keep, getattr(session, "pid", None),
                     getattr(session, "_hk_spawned_wall", None), time.time())
        self.poke()

    def _exit_sync(self, spawn: str, log_file: str, keep: bool, pid: Optional[int],
                   spawned_at: Optional[float], exited_at: float) -> None:
        if keep:
            logger.info("housekeeping: kept agy log of a failed process: %s", log_file)
        else:
            if _unlink(Path(log_file)):
                self.stats["logs_deleted"] += 1
            if pid:
                delete_crash_markers(self.root, pid, exited_at, spawned_at)
        self._ledger.remove_proc(spawn)

    # `agy models` refreshes

    def models_log_file(self) -> Optional[str]:
        if not self.active:
            return None
        return str(self.logs_dir / f"models-{secrets.token_hex(6)}.log")

    def release_log(self, path: Optional[str], *, failed: bool = False,
                    pid: Optional[int] = None, started_at: Optional[float] = None) -> None:
        if not path or (self.keep_logs and failed):
            return

        exited_at = time.time()

        def release() -> None:
            _unlink(Path(path))
            if pid:
                delete_crash_markers(self.root, pid, exited_at,
                                     None if started_at is None else started_at - 2.0)
        if self._ex is not None:
            self._submit(release)
        else:
            release()

    def note_workdir(self, workdir: Any) -> None:
        if not self.active or not workdir:
            return
        path = str(Path(workdir).resolve())
        if path in self._workdirs_seen:
            return
        self._workdirs_seen.add(path)
        self._submit(self._ledger.add_workdir, path)

    # ── deletion ────────────────────────────────────────────────────────────

    def poke(self, delay: float = 0.2) -> None:
        """Schedule a reconcile soon (coalesced)."""
        if not self.active:
            return
        if self._poke_task is not None and not self._poke_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._poke_task = loop.create_task(self._delayed_reconcile(delay),
                                           name="agi-housekeeping-reconcile")

    async def _delayed_reconcile(self, delay: float) -> None:
        await asyncio.sleep(delay)
        try:
            await self.reconcile()
        except Exception:  # pragma: no cover - defensive
            logger.warning("housekeeping: reconcile failed", exc_info=True)

    def _referenced(self) -> set[str]:
        refs: set[str] = set()
        if self.references is not None:
            with contextlib.suppress(Exception):
                refs = {r.lower() for r in self.references() if r}
        return refs

    async def reconcile(self, *, final: bool = False) -> int:
        """Delete every owned conversation no live process has open and (unless
        ``final``) the manager no longer references. Returns the count done."""
        if not self.active:
            return 0
        async with self._reconcile_lock:
            await self.flush()  # ledger rows for these uuids are written first
            refs = set() if final else self._referenced()
            candidates = sorted(u for u in self._owned if u not in refs and not self._live.get(u))
            if not candidates:
                return 0
            done = await self._run(self._delete_sync, candidates)
            self._owned.difference_update(done)
            return len(done)

    def _delete_sync(self, uuids: list[str], root: Optional[Path] = None) -> list[str]:
        root = root or self.root
        gone = [u for u in uuids if delete_conversation_state(root, u)]
        if not gone:
            return []
        caches_ok = scrub_shared_caches(root, gone)
        finished = []
        for u in gone:
            if caches_ok or self._ledger.bump_conv(u) >= _CACHE_ATTEMPTS:
                self._ledger.remove_conv(u)
                finished.append(u)
        if not caches_ok:
            logger.info("housekeeping: agy's shared cache files are busy; their entries for "
                        "%d conversation(s) are retried on the next sweep", len(gone))
        self.stats["deleted"] += len(finished)
        logger.info("housekeeping: deleted agy state of %d conversation(s)", len(finished))
        return finished

    # ── sweep ───────────────────────────────────────────────────────────────

    async def sweep(self) -> None:
        """Startup + periodic: crashed instances, ledger entries whose files
        survived, and every other piece of our cruft."""
        if not self.active:
            return
        await self.flush()
        recovered, owned, workdirs = await self._run(self._recover_sync)
        self._owned |= owned
        if workdirs and self.on_workdirs is not None:
            self.on_workdirs([Path(w) for w in workdirs])
        in_use: set[str] = set()
        if self.servers_in_use is not None:
            with contextlib.suppress(Exception):
                in_use = {s for s in self.servers_in_use() if s}
        await self.reconcile()
        swept = await self._run(self._sweep_sync, in_use, workdirs)
        self.stats["swept"] += swept
        if recovered or swept:
            logger.info("housekeeping: sweep recovered %d conversation(s) of crashed runs, "
                        "removed %d stale file(s)", recovered, swept)

    def _recover_sync(self) -> tuple[int, set[str], list[str]]:
        """Crashed instances: kill their surviving agy processes (pid + start
        time), then adopt their ledger rows — which the reconcile right after
        deletes (nothing references them)."""
        recovered = 0
        for inst in self._ledger.instances():
            if inst == self.instance or instance_alive(self.run_dir, inst):
                continue
            all_dead = True
            for spawn, _, pid, pstart, log in self._ledger.procs(inst):
                ours = bool(pid and pstart and process_start_time(pid) == pstart)
                if ours:
                    if not kill_if_same(pid, pstart):
                        all_dead = False
                        continue
                    self.stats["killed"] += 1
                if log and not self.keep_logs:
                    _unlink(Path(log))
                self._ledger.remove_proc(spawn)
            if not all_dead:
                continue  # its conversations wait until those processes are gone
            rows = self._ledger.convs(inst)
            self._ledger.adopt(inst, self.instance)
            recovered += len(rows)
            with contextlib.suppress(OSError):
                (self.run_dir / f"{inst}.lock").unlink()
        owned = set()
        for uuid, _, root, _ in self._ledger.convs(self.instance):
            if Path(root) == self.root:
                owned.add(uuid)
            elif uuid not in self._live:
                # A crashed run that used another agy root (tests, a bench
                # server): delete there; it is never live in this instance.
                if delete_conversation_state(Path(root), uuid) and \
                        scrub_shared_caches(Path(root), [uuid]):
                    self._ledger.remove_conv(uuid)
        return recovered, owned, self._ledger.workdirs()

    def _sweep_sync(self, in_use: set[str], workdirs: list[str]) -> int:
        s = self.settings
        removed = 0
        # Managed plugin dir: every *.tmp there is ours (atomic writes).
        if self.plugin_path is not None:
            for d in (self.plugin_path, self.plugin_path / "agents"):
                removed += sweep_temp_files(d, pattern=lambda n: n.endswith(".tmp"),
                                            min_age_s=_TEMP_MIN_AGE_S)
        # MCP schema caches.
        removed += sweep_mcp_caches(
            self.root, max_age_s=float(getattr(s, "mcp_cache_max_age_s", 7 * 86400)),
            in_use=in_use, user_servers=user_mcp_server_names(self.mcp_config_path))
        # Crash-left temp files of attachment writes.
        dirs = {Path(w) for w in workdirs}
        default = getattr(s, "default_workdir", None)
        if default:
            dirs.add(Path(default))
        for w in dirs:
            removed += sweep_temp_files(w / "attachments", pattern=_is_agi_temp)
        # Shared MCP config neighbours: our temp files, legacy lock, backup.
        if getattr(s, "manage_mcp_config", True):
            cfg = self.mcp_config_path
            removed += sweep_temp_files(cfg.parent, pattern=lambda n: n.startswith(TEMP_PREFIX)
                                        and n.endswith(".tmp"))
            legacy_lock = cfg.with_name(cfg.name + ".hermes.lock")
            if os.path.lexists(legacy_lock) and not legacy_lock.is_dir() and _unlink(legacy_lock):
                removed += 1
            if self._ledger.get_meta(MIGRATION_DONE_KEY):
                bak = cfg.with_name(cfg.name + ".hermes-antigravity.bak")
                if bak.exists() and _unlink(bak):
                    removed += 1
                    logger.info("housekeeping: removed the MCP-config migration backup %s", bak)
        removed += self._sweep_logs_sync()
        return removed

    def _sweep_logs_sync(self) -> int:
        """Logs no live process owns: orphans (a crash) after 10 min, kept
        error logs after 7 days."""
        owned = {Path(log).name for _, inst, _, _, log in self._ledger.procs()
                 if log and (inst == self.instance or instance_alive(self.run_dir, inst))}
        removed = 0
        now = time.time()
        try:
            entries = list(self.logs_dir.iterdir())
        except OSError:
            return 0
        for p in entries:
            if not _LOG_RE.match(p.name) or p.name in owned:
                continue
            limit = _KEPT_LOG_AGE_S if self.keep_logs else _ORPHAN_LOG_AGE_S
            if _age(p, now) > limit and _unlink(p, 1):
                removed += 1
        return removed

    def record_migration_done(self) -> None:
        if self.active:
            self._submit(self._ledger.set_meta, MIGRATION_DONE_KEY, str(time.time()))

    # ── introspection (tests / metrics) ─────────────────────────────────────

    @property
    def owned(self) -> set[str]:
        return set(self._owned)

    @property
    def live(self) -> dict[str, int]:
        return {u: len(s) for u, s in self._live.items()}

    async def ledger_snapshot(self) -> dict[str, Any]:
        if not self.active:
            return {}
        await self.flush()

        def snap() -> dict[str, Any]:
            return {"convs": self._ledger.convs(), "procs": self._ledger.procs(),
                    "workdirs": self._ledger.workdirs()}
        return await self._run(snap)
