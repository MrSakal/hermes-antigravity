#!/usr/bin/env python3
"""Prune what the ``agy`` CLI leaves behind in ``~/.gemini/antigravity-cli``.

agy has no retention of its own (``agy help``: no prune/clean flag) and every
invocation persists, forever:

* ``conversations/<id>.db`` (+ ``-wal``/``-shm``) — SQLite, WAL mode. The CLI
  does not checkpoint on exit, so the WAL often outgrows the DB. Measured on
  this host: 359 MB for 501 conversations, 255 MB of it WAL.
* ``brain/<id>/`` — full transcript JSONL, ~180 KB per conversation.
* ``log/cli-<timestamp>.log`` — one file per process start, ~28 KB.
* ``presence/<id>.lock`` — empty lock files, never removed.
* ``mcp/hermes-<8hex>/`` — the tool-schema cache agy keeps per MCP server
  name; the bridge names its server after the toolset hash, so every distinct
  toolset leaves one directory. Only ``hermes-<8hex>`` is touched — other
  names belong to the user's own servers.

The bridge never resumes a conversation older than
``AGI_CONVERSATION_CACHE_TTL_S`` (default 1 h), and agy answers a resume of a
pruned id by silently starting a fresh conversation (the bridge detects the id
mismatch and refolds the history — see ``app/conversation.py``), so deleting
old conversations is safe for the bridge. Interactive ``agy`` sessions you want
to keep are NOT distinguishable here: raise ``--keep-days`` if you use agy
interactively on this host.

Safety:
* Only touches the five directories above, only names matching agy's (or the
  bridge's) own scheme, never symlinks, never anything newer than
  ``--keep-days``.
* A conversation is pruned as a unit (db + wal + shm + brain dir + presence
  lock) and only when none of its files is newer than the cutoff AND no live
  process references it: the bridge keeps persistent agy processes (and a warm
  pool) running around the clock, so "no agy running" never happens — instead
  every live process's argv is scanned for ``--conversation <id>`` (Linux:
  ``/proc``; Windows: Toolhelp32 + the process command line), and those ids
  are skipped. A conversation a live process created itself is being written,
  hence newer than the cutoff.
* ``--checkpoint`` folds each *kept*, not-in-use conversation's WAL back into
  its DB (``PRAGMA wal_checkpoint(TRUNCATE)``), then removes the empty
  WAL/SHM. Skipped for DBs that fail ``PRAGMA quick_check``.

Exit codes: 0 success; 2 refused because live processes cannot be listed on
this platform (``--force`` prunes anyway, by age alone); 1 unexpected error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

CLI_DIR = Path("~/.gemini/antigravity-cli").expanduser()
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
CONV_RE = re.compile(rf"^({UUID})\.db(-wal|-shm)?$")
BRAIN_RE = re.compile(rf"^{UUID}$")
LOCK_RE = re.compile(rf"^{UUID}\.lock$")
LOG_RE = re.compile(r"^cli-\d{8}_\d{6}\.log$")
MCP_CACHE_RE = re.compile(r"^hermes-[0-9a-f]{8}$")
_ID_RE = re.compile(rf"^{UUID}$", re.IGNORECASE)


# ── live processes ───────────────────────────────────────────────────────────


def conversation_ids(argv: list[str]) -> set[str]:
    """``--conversation <id>`` / ``--conversation=<id>`` values in one argv."""
    out: set[str] = set()
    for i, tok in enumerate(argv):
        val = None
        if tok == "--conversation" and i + 1 < len(argv):
            val = argv[i + 1]
        elif tok.startswith("--conversation="):
            val = tok.split("=", 1)[1]
        if val and _ID_RE.match(val):
            out.add(val.lower())
    return out


def _linux_argvs() -> tuple[list[list[str]], int]:
    argvs: list[list[str]] = []
    for cmd in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            raw = cmd.read_bytes()
        except OSError:
            continue  # exited meanwhile (or hidepid): such a process is not ours
        if raw:
            argvs.append([os.fsdecode(a) for a in raw.rstrip(b"\0").split(b"\0")])
    return argvs, 0


def _windows_argvs() -> tuple[list[list[str]], int]:  # pragma: no cover - Windows only
    """Every readable process command line (Toolhelp32 snapshot, then
    NtQueryInformationProcess(ProcessCommandLineInformation) per pid — no WMI,
    no psutil). Returns (argvs, agy.exe processes whose command line could not
    be read)."""
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    class UNICODE_STRING(ctypes.Structure):
        _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                    ("Buffer", ctypes.c_void_p)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    ntdll.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p,
                                                ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
    ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    procs: list[tuple[int, str]] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            procs.append((entry.th32ProcessID, entry.szExeFile))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    def command_line(pid: int) -> Optional[str]:
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            size = ctypes.c_ulong(0)
            ntdll.NtQueryInformationProcess(handle, 60, None, 0, ctypes.byref(size))
            if not size.value:
                return None
            buf = ctypes.create_string_buffer(size.value)
            if ntdll.NtQueryInformationProcess(handle, 60, buf, size, ctypes.byref(size)) != 0:
                return None
            us = UNICODE_STRING.from_buffer(buf)
            if not us.Buffer or not us.Length:
                return None
            return ctypes.wstring_at(us.Buffer, us.Length // 2)
        finally:
            kernel32.CloseHandle(handle)

    def split(cmdline: str) -> list[str]:
        argc = ctypes.c_int(0)
        argv = shell32.CommandLineToArgvW(cmdline, ctypes.byref(argc))
        if not argv:
            return cmdline.split()
        try:
            return [argv[i] for i in range(argc.value)]
        finally:
            kernel32.LocalFree(argv)

    argvs: list[list[str]] = []
    unreadable_agy = 0
    for pid, name in procs:
        if pid in (0, 4):
            continue
        cmdline = command_line(pid)
        if cmdline is None:
            unreadable_agy += name.lower() == "agy.exe"
            continue
        argvs.append(split(cmdline))
    return argvs, unreadable_agy


def live_argvs() -> Optional[tuple[list[list[str]], int]]:
    """(argv of every visible process, agy processes that could not be read),
    or None when this platform offers no process listing."""
    if os.name == "nt":
        return _windows_argvs()
    if Path("/proc/self/cmdline").exists():
        return _linux_argvs()
    return None


def in_use_conversations() -> Optional[tuple[set[str], int]]:
    listed = live_argvs()
    if listed is None:
        return None
    argvs, unreadable = listed
    ids: set[str] = set()
    for argv in argvs:
        ids |= conversation_ids(argv)
    return ids, unreadable


def _regular(p: Path) -> bool:
    return p.is_file() and not p.is_symlink()


def _mtime(p: Path) -> float:
    try:
        return p.lstat().st_mtime
    except FileNotFoundError:
        return 0.0


def _size(p: Path) -> int:
    try:
        return p.lstat().st_size
    except FileNotFoundError:
        return 0


def _rm(p: Path, stats: dict, dry: bool) -> None:
    if p.is_dir() and not p.is_symlink():
        for child in sorted(p.rglob("*"), reverse=True):
            _rm(child, stats, dry)
        if not dry:
            p.rmdir()
        stats["dirs"] += 1
        return
    stats["bytes"] += _size(p)
    stats["files"] += 1
    if not dry:
        p.unlink(missing_ok=True)


def conversations(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in (root / "conversations").iterdir() if (root / "conversations").is_dir() else []:
        m = CONV_RE.match(p.name)
        if m and _regular(p):
            groups.setdefault(m.group(1), []).append(p)
    return groups


def checkpoint(db: Path, stats: dict, dry: bool) -> None:
    wal, shm = db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")
    if not _regular(wal):
        return
    before = _size(wal)
    if dry:
        stats["wal_bytes"] += before
        stats["checkpointed"] += 1
        return
    try:
        con = sqlite3.connect(f"file:{db}?mode=rw", uri=True, timeout=5)
        try:
            if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                stats["checkpoint_skipped"] += 1
                return
            busy, _, _ = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy:
                stats["checkpoint_skipped"] += 1
                return
        finally:
            con.close()
    except sqlite3.Error:
        stats["checkpoint_skipped"] += 1
        return
    stats["wal_bytes"] += before - _size(wal)
    stats["checkpointed"] += 1
    for side in (wal, shm):
        if _regular(side) and _size(side) == 0:
            side.unlink(missing_ok=True)



def _newest(p: Path) -> float:
    """Newest mtime of ``p`` and, for a directory, everything below it."""
    newest = _mtime(p)
    if p.is_dir() and not p.is_symlink():
        for child in p.rglob("*"):
            newest = max(newest, _mtime(child))
    return newest


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=CLI_DIR)
    ap.add_argument("--keep-days", type=float, default=7.0)
    ap.add_argument("--checkpoint", action="store_true", help="fold kept conversations' WAL into the DB")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="prune by age alone when live processes cannot be listed on this platform")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    listed = in_use_conversations()
    if listed is None:
        if not args.force:
            print("refusing: cannot list live processes on this platform, so conversations "
                  "in use cannot be told apart; --force prunes by age alone", file=sys.stderr)
            return 2
        in_use, unreadable = set(), 0
    else:
        in_use, unreadable = listed
    if unreadable:
        print(f"warning: {unreadable} agy process(es) whose command line is unreadable; "
              "their conversations are protected by --keep-days only", file=sys.stderr)

    root = args.root.expanduser()
    cutoff = time.time() - args.keep_days * 86400
    stats = {k: 0 for k in ("conversations", "files", "dirs", "bytes", "logs", "locks",
                            "mcp_caches", "in_use_skipped", "checkpointed",
                            "checkpoint_skipped", "wal_bytes")}

    for conv_id, files in conversations(root).items():
        brain = root / "brain" / conv_id
        lock = root / "presence" / f"{conv_id}.lock"
        if conv_id in in_use:
            # A live process (persistent session / resume) holds it open.
            stats["in_use_skipped"] += 1
            continue
        members = list(files)
        if brain.is_dir() and not brain.is_symlink():
            members.append(brain)
        if _regular(lock):
            members.append(lock)
        newest = max((_mtime(p) for p in files + ([brain] if brain.exists() else [])), default=0.0)
        if newest < cutoff:
            for p in members:
                _rm(p, stats, args.dry_run)
            stats["conversations"] += 1
        elif args.checkpoint:
            db = root / "conversations" / f"{conv_id}.db"
            if _regular(db):
                checkpoint(db, stats, args.dry_run)

    # Brain dirs / locks whose conversation DB is already gone.
    for d in (root / "brain").iterdir() if (root / "brain").is_dir() else []:
        if BRAIN_RE.match(d.name) and d.name not in in_use and d.is_dir() and not d.is_symlink() \
                and not (root / "conversations" / f"{d.name}.db").exists() and _mtime(d) < cutoff:
            _rm(d, stats, args.dry_run)
    for p in (root / "presence").iterdir() if (root / "presence").is_dir() else []:
        if LOCK_RE.match(p.name) and p.stem not in in_use and _regular(p) \
                and not (root / "conversations" / f"{p.stem}.db").exists() and _mtime(p) < cutoff:
            _rm(p, stats, args.dry_run)
            stats["locks"] += 1
    for p in (root / "log").iterdir() if (root / "log").is_dir() else []:
        if LOG_RE.match(p.name) and _regular(p) and _mtime(p) < cutoff:
            _rm(p, stats, args.dry_run)
            stats["logs"] += 1
    # The bridge's per-toolset MCP schema caches (never other server names).
    # A live process re-lists its tools if its cache vanishes, so age is enough.
    for d in (root / "mcp").iterdir() if (root / "mcp").is_dir() else []:
        if MCP_CACHE_RE.match(d.name) and d.is_dir() and not d.is_symlink() \
                and _newest(d) < cutoff:
            _rm(d, stats, args.dry_run)
            stats["mcp_caches"] += 1

    stats["dry_run"] = args.dry_run
    stats["keep_days"] = args.keep_days
    stats["live_conversations"] = len(in_use)
    if args.json:
        print(json.dumps(stats))
    else:
        verb = "would remove" if args.dry_run else "removed"
        print(f"{verb} {stats['conversations']} conversations, {stats['logs']} logs, "
              f"{stats['locks']} orphan locks, {stats['mcp_caches']} MCP schema caches "
              f"({stats['bytes'] / 1e6:.1f} MB); {stats['in_use_skipped']} in use; "
              f"checkpointed {stats['checkpointed']} WALs "
              f"(-{stats['wal_bytes'] / 1e6:.1f} MB, {stats['checkpoint_skipped']} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
