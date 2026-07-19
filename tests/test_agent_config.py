from __future__ import annotations

import json
import os
import time

from app.agent_config import AgentPluginManager, slug_for


async def test_ensure_agent_creates_plugin_and_file(tmp_path):
    mgr = AgentPluginManager(tmp_path / "hermes-bridge")
    slug = await mgr.ensure_agent("You are a pirate.")

    assert slug == slug_for("You are a pirate.")
    manifest = json.loads((tmp_path / "hermes-bridge" / "plugin.json").read_text())
    assert manifest["name"] == "hermes-bridge"
    body = (tmp_path / "hermes-bridge" / "agents" / f"{slug}.md").read_text()
    assert body.startswith("---\n")
    assert f"name: {slug}" in body
    assert "You are a pirate." in body


async def test_same_prompt_same_slug_idempotent(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p")
    a = await mgr.ensure_agent("prompt A")
    b = await mgr.ensure_agent("prompt A")
    c = await mgr.ensure_agent("prompt B")
    assert a == b
    assert a != c
    files = list((tmp_path / "p" / "agents").glob("hermes-*.md"))
    assert len(files) == 2


async def test_prune_drops_oldest_owned_only(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p", max_agents=2)
    # A foreign agent file must never be touched.
    (tmp_path / "p" / "agents").mkdir(parents=True)
    foreign = tmp_path / "p" / "agents" / "my-own-agent.md"
    foreign.write_text("hands off")

    # Stagger mtimes BEFORE the prune-triggering third write, so LRU order is
    # deterministic even on coarse filesystem clocks.
    slugs = [await mgr.ensure_agent("one")]
    os.utime(tmp_path / "p" / "agents" / f"{slugs[0]}.md",
             (time.time() - 100, time.time() - 100))
    slugs.append(await mgr.ensure_agent("two"))
    os.utime(tmp_path / "p" / "agents" / f"{slugs[1]}.md",
             (time.time() - 50, time.time() - 50))
    slugs.append(await mgr.ensure_agent("three"))  # prune runs here

    remaining = {p.name for p in (tmp_path / "p" / "agents").iterdir()}
    assert foreign.name in remaining
    assert f"{slugs[0]}.md" not in remaining  # oldest pruned
    assert f"{slugs[1]}.md" in remaining
    assert f"{slugs[2]}.md" in remaining


async def test_reuse_bumps_mtime_for_lru(tmp_path):
    mgr = AgentPluginManager(tmp_path / "p", max_agents=2)
    a = await mgr.ensure_agent("alpha")
    b = await mgr.ensure_agent("beta")
    ta = tmp_path / "p" / "agents" / f"{a}.md"
    tb = tmp_path / "p" / "agents" / f"{b}.md"
    past = time.time() - 1000
    os.utime(ta, (past, past))
    os.utime(tb, (past + 1, past + 1))

    # Re-using "alpha" must protect it; the next new agent evicts "beta".
    await mgr.ensure_agent("alpha")
    await mgr.ensure_agent("gamma")
    remaining = {p.name for p in (tmp_path / "p" / "agents").iterdir()}
    assert f"{a}.md" in remaining
    assert f"{b}.md" not in remaining


async def test_unwritable_path_returns_none(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the plugin dir should be")
    mgr = AgentPluginManager(blocker)
    assert await mgr.ensure_agent("whatever") is None
