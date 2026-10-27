# Deployment units (systemd --user)

| Unit | Purpose |
|---|---|
| `agi-server.service` | The bridge itself on `127.0.0.1:8788`, venv python, `.env` via `EnvironmentFile`, sandboxed (`ProtectHome=read-only` + explicit `ReadWritePaths`). |
| `agy-prune.service` + `.timer` | Daily 04:15 run of `scripts/prune_agy_state.py --keep-days 7 --checkpoint`: agy keeps every conversation/transcript/log (and one `mcp/hermes-<8hex>/` schema cache per toolset) forever and never checkpoints its SQLite WALs; this bounds the growth. It runs while the server's persistent agy processes and warm pool are up: conversations a live process references (`--conversation <id>` in its argv, read from `/proc`) or touched within `--keep-days` are skipped. Exit 0 = done, 2 = live processes cannot be listed (refused; `--force` prunes by age alone), 1 = error — the unit no longer treats 2 as success. |

```bash
install -m 0644 deploy/*.service deploy/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now agi-server.service agy-prune.timer
loginctl enable-linger "$USER"   # keep running after logout
```

The live units are the source of truth; copy them back here after editing
(`cp ~/.config/systemd/user/agi-server.service deploy/`). No `--reload`: restart
the service after a code change.
