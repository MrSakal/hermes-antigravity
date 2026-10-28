"""``python -m app.purge`` — remove the server's ENTIRE on-disk footprint.

Run it while the server is stopped (it stops nothing itself; a running server
holds its instance lock in the state dir and the purge refuses with exit 2).
Removes, using the same settings / environment as the server (``AGI_*``):

* the agy state of every conversation in the ledger (``conversations/``,
  ``presence/``, ``brain/``, ``annotations/``, ``implicit/`` + our entries in
  agy's two shared cache JSON files) — after killing any agy process a crashed
  run left behind (pid + start time verified);
* the managed plugin dir (``~/.gemini/config/plugins/hermes-bridge``);
* our MCP tool-schema caches (``acli/mcp/hermes-<8hex>/``, legacy
  ``acli/mcp/hermes/`` unless the user configured a server of that name);
* attachments in every ledger workdir and the default workdir (the default
  workdir itself too when it is then empty);
* the shared MCP config's migration backup (``.hermes-antigravity.bak``),
  legacy ``.hermes.lock`` and our crash-left ``.agi-*.tmp`` beside it;
* the state dir itself (ledger, logs, locks).

Never touched: any agy conversation that is not in the ledger (the user's own
agy / IDE data), ``conversation_summaries.db``, other plugins, other MCP
caches, the shared ``mcp_config.json`` itself.

Exit codes: 0 done (or nothing to do); 1 something could not be removed (a
file still held — rerun); 2 refused, a server is running on this state dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from app.agent_config import default_plugin_path
from app.config import Settings
from app.housekeeping import (
    LEDGER_NAME,
    UUID_RE,
    Ledger,
    _rmtree,
    _unlink,
    delete_conversation_state,
    kill_if_same,
    live_instances,
    purge_attachments,
    resolve_agy_root,
    resolve_state_dir,
    scrub_shared_caches,
    sweep_mcp_caches,
    sweep_temp_files,
    user_mcp_server_names,
)
from app.mcp_config import BACKUP_SUFFIX, default_config_path

logger = logging.getLogger("agi.purge")


def purge(settings: Settings, *, dry_run: bool = False) -> tuple[int, dict]:
    state = resolve_state_dir(settings)
    root = resolve_agy_root(settings)
    report: dict = {"state_dir": str(state), "conversations": 0, "killed": 0,
                    "attachments": 0, "mcp_caches": 0, "removed": [], "left": []}
    running = live_instances(state / "run") if state.is_dir() else []
    if running:
        report["refused"] = f"server instance(s) running: {', '.join(running)}"
        return 2, report

    def gone(path: Path, ok: bool) -> None:
        (report["removed"] if ok else report["left"]).append(str(path))

    workdirs: set[Path] = {Path(settings.default_workdir).expanduser()}
    ledger_path = state / LEDGER_NAME
    if ledger_path.exists():
        ledger = Ledger(ledger_path).open()
        try:
            for spawn, _, pid, pstart, log in ledger.procs():
                if pid and pstart and not dry_run:
                    if kill_if_same(pid, pstart):
                        report["killed"] += 1
                    else:
                        report["left"].append(f"pid {pid}")
            rows = ledger.convs()
            by_root: dict[Path, list[str]] = {}
            for uuid, _, conv_root, _ in rows:
                if UUID_RE.match(uuid):
                    by_root.setdefault(Path(conv_root), []).append(uuid)
            for conv_root, uuids in by_root.items():
                deleted = []
                for uuid in uuids:
                    if dry_run or delete_conversation_state(conv_root, uuid):
                        deleted.append(uuid)
                    else:
                        report["left"].append(f"conversation {uuid}")
                if deleted and not dry_run and not scrub_shared_caches(conv_root, deleted):
                    report["left"].append(f"{conv_root / 'cache'} entries (agy busy)")
                report["conversations"] += len(deleted)
            workdirs.update(Path(w) for w in ledger.workdirs())
        finally:
            ledger.close()

    # Managed plugin dir (ours entirely).
    plugin = Path(settings.agent_plugin_path).expanduser() if settings.agent_plugin_path \
        else default_plugin_path()
    if plugin.exists():
        gone(plugin, dry_run or _rmtree(plugin))

    # MCP schema caches (never a name the user configured).
    cfg = Path(settings.mcp_config_path).expanduser() if settings.mcp_config_path \
        else default_config_path()
    report["mcp_caches"] = sweep_mcp_caches(
        root, max_age_s=0, everything=True, user_servers=user_mcp_server_names(cfg),
        dry_run=dry_run)

    # Attachments; the default workdir too when nothing else is in it.
    for w in sorted(workdirs):
        report["attachments"] += purge_attachments(w, dry_run=dry_run)
    default = Path(settings.default_workdir).expanduser()
    if not dry_run and default.is_dir() and not default.is_symlink():
        try:
            default.rmdir()
            gone(default, True)
        except OSError:
            pass  # not empty: the user's (or the model's) files stay

    # Shared MCP config neighbours (the config itself is the user's).
    if settings.manage_mcp_config:
        for p in (cfg.with_name(cfg.name + BACKUP_SUFFIX), cfg.with_name(cfg.name + ".hermes.lock")):
            if p.exists():
                gone(p, dry_run or _unlink(p))
        if not dry_run:
            sweep_temp_files(cfg.parent, pattern=lambda n: n.startswith(".agi-") and n.endswith(".tmp"),
                             min_age_s=0)

    # Our state dir last (the ledger is only dropped once the rest is done).
    if state.exists():
        if report["left"]:
            report["left"].append(f"{state} (kept: it still lists what is left)")
        else:
            gone(state, dry_run or _rmtree(state))
    return (1 if report["left"] else 0), report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.purge",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only report what would be removed")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    code, report = purge(Settings(), dry_run=args.dry_run)
    if args.json:
        print(json.dumps(report, indent=2))
    elif code == 2:
        print(f"refusing: {report['refused']} — stop the server first", file=sys.stderr)
    else:
        verb = "would remove" if args.dry_run else "removed"
        print(f"{verb}: {report['conversations']} agy conversation(s), "
              f"{report['mcp_caches']} MCP schema cache(s), {report['attachments']} attachment(s), "
              f"killed {report['killed']} leftover process(es)")
        for p in report["removed"]:
            print(f"  - {p}")
        for p in report["left"]:
            print(f"  ! left: {p}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
