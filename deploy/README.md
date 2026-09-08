# Deployment units (systemd --user)

| Unit | Purpose |
|---|---|
| `agi-server.service` | The bridge itself on `127.0.0.1:8788`, venv python, `.env` via `EnvironmentFile`, sandboxed (`ProtectHome=read-only` + explicit `ReadWritePaths`). |
| `agy-prune.service` + `.timer` | Daily 04:15 run of `scripts/prune_agy_state.py --keep-days 7 --checkpoint`: agy keeps every conversation/transcript/log forever and never checkpoints its SQLite WALs; this bounds the growth. |

```bash
install -m 0644 deploy/*.service deploy/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now agi-server.service agy-prune.timer
loginctl enable-linger "$USER"   # keep running after logout
```

The live units are the source of truth; copy them back here after editing
(`cp ~/.config/systemd/user/agi-server.service deploy/`). No `--reload`: restart
the service after a code change.
