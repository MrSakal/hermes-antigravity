#!/usr/bin/env python3
"""Prune what the ``agy`` CLI leaves behind in ``~/.gemini/antigravity-cli``.

agy 1.1.27 has no retention of its own (``agy help``: no prune/clean flag) and
every invocation persists, forever:

* ``conversations/<id>.db`` (+ ``-wal``/``-shm``) — SQLite, WAL mode. The CLI
  does not checkpoint on exit, so the WAL often outgrows the DB. Measured on
  this host: 359 MB for 501 conversations, 255 MB of it WAL.
* ``brain/<id>/`` — full transcript JSONL, ~180 KB per conversation.
* ``log/cli-<timestamp>.log`` — one file per process start, ~28 KB.
* ``presence/<id>.lock`` — empty lock files, never removed.

The bridge never resumes a conversation older than
``AGI_CONVERSATION_CACHE_TTL_S`` (default 1 h), and agy answers a resume of a
pruned id by silently starting a fresh conversation (the bridge detects the id
mismatch and refolds the history — see ``app/routes/chat.py``), so deleting
old conversations is safe for the bridge. Interactive ``agy`` sessions you want
to keep are NOT distinguishable here: raise ``--keep-days`` if you use agy
interactively on this host.

Safety:
* Only touches the four directories above, only files matching agy's own
  naming, never symlinks, never anything newer than ``--keep-days``.
* A conversation is pruned as a unit (db + wal + shm + brain dir + presence
  lock) and only when none of its files is newer than the cutoff.
* Refuses to run while an ``agy --print`` process is alive (a live turn may
  hold any of these files open) unless ``--force``.
* ``--checkpoint`` folds each *kept* conversation's WAL back into its DB
  (``PRAGMA wal_checkpoint(TRUNCATE)``), then removes the empty WAL/SHM.
  Skipped for DBs that fail ``PRAGMA quick_check``.

Exit code 0 on success, 2 when refusing due to a live agy process.
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

CLI_DIR = Path("~/.gemini/antigravity-cli").expanduser()
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
CONV_RE = re.compile(rf"^({UUID})\.db(-wal|-shm)?$")
BRAIN_RE = re.compile(rf"^{UUID}$")
LOCK_RE = re.compile(rf"^{UUID}\.lock$")
LOG_RE = re.compile(r"^cli-\d{8}_\d{6}\.log$")


def agy_print_running() -> list[int]:
    pids = []
    for cmd in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = cmd.read_bytes().split(b"\0")
            if argv and os.path.basename(os.fsdecode(argv[0])) == "agy" and b"--print" in argv:
                pids.append(int(cmd.parent.name))
        except (OSError, ValueError):
            pass
    return pids


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=CLI_DIR)
    ap.add_argument("--keep-days", type=float, default=7.0)
    ap.add_argument("--checkpoint", action="store_true", help="fold kept conversations' WAL into the DB")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if an agy --print process is alive")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.force:
        live = agy_print_running()
        if live:
            print(f"refusing: agy --print running (pids {live}); retry later or --force", file=sys.stderr)
            return 2

    root = args.root.expanduser()
    cutoff = time.time() - args.keep_days * 86400
    stats = {k: 0 for k in ("conversations", "files", "dirs", "bytes", "logs", "locks",
                            "checkpointed", "checkpoint_skipped", "wal_bytes")}

    for conv_id, files in conversations(root).items():
        brain = root / "brain" / conv_id
        lock = root / "presence" / f"{conv_id}.lock"
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
        if BRAIN_RE.match(d.name) and d.is_dir() and not d.is_symlink() \
                and not (root / "conversations" / f"{d.name}.db").exists() and _mtime(d) < cutoff:
            _rm(d, stats, args.dry_run)
    for p in (root / "presence").iterdir() if (root / "presence").is_dir() else []:
        if LOCK_RE.match(p.name) and _regular(p) \
                and not (root / "conversations" / f"{p.stem}.db").exists() and _mtime(p) < cutoff:
            _rm(p, stats, args.dry_run)
            stats["locks"] += 1
    for p in (root / "log").iterdir() if (root / "log").is_dir() else []:
        if LOG_RE.match(p.name) and _regular(p) and _mtime(p) < cutoff:
            _rm(p, stats, args.dry_run)
            stats["logs"] += 1

    stats["dry_run"] = args.dry_run
    stats["keep_days"] = args.keep_days
    if args.json:
        print(json.dumps(stats))
    else:
        verb = "would remove" if args.dry_run else "removed"
        print(f"{verb} {stats['conversations']} conversations, {stats['logs']} logs, {stats['locks']} orphan locks "
              f"({stats['bytes'] / 1e6:.1f} MB); checkpointed {stats['checkpointed']} WALs "
              f"(-{stats['wal_bytes'] / 1e6:.1f} MB, {stats['checkpoint_skipped']} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
