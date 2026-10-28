# Deployment units (systemd --user)

| Unit | Purpose |
|---|---|
| `agi-server.service` | The bridge itself on `127.0.0.1:8788`, venv python, `.env` via `EnvironmentFile`, sandboxed (`ProtectHome=read-only` + explicit `ReadWritePaths`: the workspace, `~/.gemini`, and the server's state dir `~/.local/state/hermes-antigravity`). |

There is no separate pruning timer any more: the server cleans up after itself
(see "Footprint & cleanup" in the top-level README). It records every agy
conversation, process and workdir it creates in its ledger
(`~/.local/state/hermes-antigravity/ledger.sqlite3`), deletes an agy
conversation as soon as its process exits, sweeps leftovers every
`AGI_HOUSEKEEPING_INTERVAL_S` (10 min) and, at the next start, kills and
deletes what a crashed run left. It never deletes an agy conversation that is
not in that ledger, i.e. never your own agy / IDE data.

```bash
install -m 0644 deploy/agi-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now agi-server.service
loginctl enable-linger "$USER"   # keep running after logout
```

Upgrading from a build that had `agy-prune.timer`: that timer deleted agy
conversations by age alone, the user's own included. Remove it:

```bash
systemctl --user disable --now agy-prune.timer
rm ~/.config/systemd/user/agy-prune.{service,timer}
systemctl --user daemon-reload
```

Stop gracefully (`systemctl --user stop agi-server`, SIGTERM): shutdown deletes
every conversation the run created and releases the state lock. To remove the
server's entire footprint (ledger conversations, the `hermes-bridge` plugin,
MCP schema caches, attachments, the state dir) after stopping it:

```bash
.venv/bin/python -m app.purge          # exit 2 = a server is still running
```

The live units are the source of truth; copy them back here after editing
(`cp ~/.config/systemd/user/agi-server.service deploy/`). No `--reload`: restart
the service after a code change.
