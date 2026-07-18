"""Suspend/resume state-machine tests with a fake hanging agy subprocess.

The real MCP HTTP transport was verified live (spike: agy blocks 90s+ inside
call_tool); here the bridge's ``dispatch`` is driven directly, which is exactly
what the MCP ``call_tool`` handler does.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import Settings
from app.conversation import (
    SUSPENDED,
    ConversationManager,
    DoneChunk,
    ErrorChunk,
    ExpiredContinuation,
    TextChunk,
    ToolCallsChunk,
)
from app.mcp_bridge import McpBridge, ToolResultError
from app.mcp_config import McpConfigManager
from app.openai_models import ChatCompletionRequest


def make_request(**kw) -> ChatCompletionRequest:
    payload = {
        "messages": [{"role": "user", "content": "what's the weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }}],
    }
    payload.update(kw)
    return ChatCompletionRequest(**payload)


@pytest.fixture
def mgr(tmp_path, fake_agy):
    settings = Settings(
        _env_file=None,
        default_workdir=tmp_path / "ws",
        mcp_config_path=tmp_path / "mcp_config.json",
        request_timeout_s=15,
        suspended_ttl_s=300,
    )
    (tmp_path / "ws").mkdir()
    return ConversationManager(McpBridge(), McpConfigManager(settings.mcp_config_path), settings)


async def collect_turn(mgr, conv):
    chunks = []
    async for ch in mgr.run_turn(conv):
        chunks.append(ch)
    return chunks


async def test_full_tool_cycle(mgr, fake_agy, tmp_path):
    fake_agy("hang")
    req = make_request()
    conv = await mgr.create(req, model="Gemini 3.5 Flash (Medium)",
                            workdir=mgr.settings.default_workdir)

    # Config entry exists while the conversation lives.
    cfg = json.loads((tmp_path / "mcp_config.json").read_text(encoding="utf-8"))
    assert conv.slot in cfg["mcpServers"]

    # Simulate agy calling our MCP bridge (what call_tool does) — like the real
    # CLI, the call arrives after the text delta has already streamed.
    async def dispatch_later():
        await asyncio.sleep(1.0)
        return await conv.bridge.dispatch("get_weather", {"city": "Budapest"})

    dispatch_task = asyncio.create_task(dispatch_later())

    chunks = await collect_turn(mgr, conv)
    texts = [c for c in chunks if isinstance(c, TextChunk)]
    calls = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert texts and texts[0].text == "Working "
    assert len(calls) == 1 and len(calls[0].calls) == 1
    pc = calls[0].calls[0]
    assert pc.name == "get_weather"
    assert pc.arguments == {"city": "Budapest"}
    assert conv.state == SUSPENDED
    assert not dispatch_task.done()

    # Continuation: hermes returns the tool result.
    cont = ChatCompletionRequest(**{
        "messages": [
            {"role": "user", "content": "what's the weather?"},
            {"role": "assistant", "tool_calls": [pc.openai_tool_call()]},
            {"role": "tool", "tool_call_id": pc.id, "content": "sunny, 28C"},
        ],
        "tools": [],
    })
    assert ConversationManager.is_continuation(cont)
    resumed = await mgr.resume(cont)
    assert resumed is conv
    assert await asyncio.wait_for(dispatch_task, timeout=5) == "sunny, 28C"

    await mgr.close_all()
    cfg = json.loads((tmp_path / "mcp_config.json").read_text(encoding="utf-8"))
    assert conv.slot not in cfg["mcpServers"]


async def test_parallel_batch_collected(mgr, fake_agy):
    fake_agy("hang")
    conv = await mgr.create(make_request(), model=None,
                            workdir=mgr.settings.default_workdir)
    t1 = asyncio.create_task(conv.bridge.dispatch("f1", {"a": 1}))
    t2 = asyncio.create_task(conv.bridge.dispatch("f2", {"b": 2}))
    chunks = await collect_turn(mgr, conv)
    calls = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(calls) == 1 and len(calls[0].calls) == 2
    conv.bridge.fail_all("test done")
    for t in (t1, t2):
        with pytest.raises(ToolResultError):
            await t
    await mgr.close_all()


async def test_resume_unknown_ids_raises(mgr):
    cont = ChatCompletionRequest(**{
        "messages": [{"role": "tool", "tool_call_id": "call_nope", "content": "x"}],
    })
    with pytest.raises(ExpiredContinuation):
        await mgr.resume(cont)


async def test_clean_completion_closes(mgr, fake_agy, tmp_path):
    fake_agy("simple")
    conv = await mgr.create(make_request(), model=None,
                            workdir=mgr.settings.default_workdir)
    chunks = await collect_turn(mgr, conv)
    assert isinstance(chunks[-1], DoneChunk)
    assert conv.state == "closed"
    cfg = json.loads((tmp_path / "mcp_config.json").read_text(encoding="utf-8"))
    assert conv.slot not in cfg["mcpServers"]


async def test_upstream_error_surfaces(mgr, fake_agy):
    fake_agy("error")
    conv = await mgr.create(make_request(), model=None,
                            workdir=mgr.settings.default_workdir)
    chunks = await collect_turn(mgr, conv)
    errs = [c for c in chunks if isinstance(c, ErrorChunk)]
    assert errs and "not recognized" in errs[0].message


async def test_gc_reaps_suspended(mgr, fake_agy, tmp_path):
    fake_agy("hang")
    conv = await mgr.create(make_request(), model=None,
                            workdir=mgr.settings.default_workdir)
    dispatch_task = asyncio.create_task(conv.bridge.dispatch("f", {}))
    await collect_turn(mgr, conv)
    assert conv.state == SUSPENDED

    conv.last_activity -= mgr.settings.suspended_ttl_s + 1
    assert await mgr.gc_once() == 1
    with pytest.raises(ToolResultError):
        await asyncio.wait_for(dispatch_task, timeout=5)
    cfg = json.loads((tmp_path / "mcp_config.json").read_text(encoding="utf-8"))
    assert conv.slot not in cfg["mcpServers"]


async def test_double_continuation_rejected(mgr, fake_agy):
    fake_agy("hang")
    conv = await mgr.create(make_request(), model=None,
                            workdir=mgr.settings.default_workdir)
    dispatch_task = asyncio.create_task(conv.bridge.dispatch("f", {}))
    chunks = await collect_turn(mgr, conv)
    pc = next(c for c in chunks if isinstance(c, ToolCallsChunk)).calls[0]

    cont_payload = {
        "messages": [{"role": "tool", "tool_call_id": pc.id, "content": "r"}],
    }
    await mgr.resume(ChatCompletionRequest(**cont_payload))
    await asyncio.wait_for(dispatch_task, timeout=5)
    # Second identical continuation: the conversation is RUNNING now.
    with pytest.raises(ExpiredContinuation):
        await mgr.resume(ChatCompletionRequest(**cont_payload))
    await mgr.close_all()
