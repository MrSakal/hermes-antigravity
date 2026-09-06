from __future__ import annotations

import json

import pytest

from app.agy_session import STREAM_CLOSED, AgySession
from app.events import Error, Init, TextDelta, ToolStep, TurnDone


async def drain(sess: AgySession) -> list:
    events = []
    while True:
        ev = await sess.next_event(timeout=15)
        assert ev is not None, "timed out waiting for fake agy"
        if ev is STREAM_CLOSED:
            break
        events.append(ev)
    return events


def make_session(tmp_path, **kw) -> AgySession:
    defaults = dict(
        agy_bin="agy-unused",
        model="Gemini 3.7 Flash (Medium)",
        workdir=tmp_path,
        prompt="hello",
        print_timeout_s=30,
    )
    defaults.update(kw)
    return AgySession(**defaults)


async def test_simple_turn(fake_agy, tmp_path):
    fake_agy("simple")
    sess = make_session(tmp_path)
    await sess.start()
    events = await drain(sess)
    await sess.aclose()

    assert isinstance(events[0], Init)
    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert texts == ["Hello ", "world"]
    done = [e for e in events if isinstance(e, TurnDone)]
    assert len(done) == 1
    assert done[0].response_text == "Hello world"
    assert done[0].conversation_id == "conv-fake-1"


async def test_tool_steps_surface(fake_agy, tmp_path):
    fake_agy("tool")
    sess = make_session(tmp_path)
    await sess.start()
    events = await drain(sess)
    await sess.aclose()

    steps = [e for e in events if isinstance(e, ToolStep)]
    assert [s.state for s in steps] == ["ACTIVE", "DONE"]
    assert steps[0].tool_name == "list_dir"
    assert steps[1].output == "a.txt\nb.txt"


async def test_error_result(fake_agy, tmp_path):
    fake_agy("error")
    sess = make_session(tmp_path)
    await sess.start()
    events = await drain(sess)
    await sess.aclose()

    errs = [e for e in events if isinstance(e, Error)]
    assert len(errs) == 1
    assert "not recognized" in errs[0].message


async def test_noise_keeps_diagnostics(fake_agy, tmp_path):
    fake_agy("noise")
    sess = make_session(tmp_path)
    await sess.start()
    events = await drain(sess)
    await sess.aclose()

    assert events == []  # nothing parseable
    diag = sess.diagnostics()
    assert "no output produced" in diag
    assert "some stderr detail" in diag


async def test_arg_construction(fake_agy, tmp_path):
    fake_agy("echoargs")
    sess = make_session(tmp_path, conversation_id="prev-42", prompt="the prompt")
    await sess.start()
    events = await drain(sess)
    await sess.aclose()

    # fake_agy echoargs mode emits {"argv": [...], "prompt": ...} which
    # parse_line ignores (no "event") — so it lands in raw_tail.
    argv_line = next((l for l in sess.raw_tail if l.startswith('{"argv"')), None)
    assert argv_line is not None
    echoed = json.loads(argv_line)
    argv = echoed["argv"]
    # stdin transport: print mode with an EMPTY argv prompt, stream-json in.
    assert argv[:4] == ["--print", "", "--input-format", "stream-json"]
    assert "-p" not in argv
    assert echoed["prompt"] == "the prompt"
    assert "--output-format" in argv and "stream-json" in argv
    assert "--dangerously-skip-permissions" in argv
    assert "--disable-slash-commands" in argv
    assert "--print-timeout" in argv and "30s" in argv
    assert "--model" in argv and "Gemini 3.7 Flash (Medium)" in argv
    assert "--conversation" in argv and "prev-42" in argv
    assert "--add-dir" in argv
    assert "--agent" not in argv


async def test_prompt_not_in_argv_survives_argv_limits(fake_agy, tmp_path):
    """A prompt far beyond MAX_ARG_STRLEN (128 KiB) must reach agy intact
    over stdin — the whole point of the stream-json transport."""
    fake_agy("echoprompt")
    big = "ő" * 300_000 + "THE_END!"  # 600 KB+ of UTF-8
    sess = make_session(tmp_path, prompt=big)
    await sess.start()
    events = await drain(sess)
    await sess.aclose()
    done = [e for e in events if isinstance(e, TurnDone)]
    assert done and done[0].response_text == f"len={len(big)};tail=THE_END!"


async def test_slash_commands_can_be_enabled(fake_agy, tmp_path):
    fake_agy("echoargs")
    sess = make_session(tmp_path, disable_slash_commands=False)
    await sess.start()
    await drain(sess)
    await sess.aclose()
    argv = json.loads(next(l for l in sess.raw_tail if l.startswith('{"argv"')))["argv"]
    assert "--disable-slash-commands" not in argv


async def test_arg_construction_with_agent(fake_agy, tmp_path):
    fake_agy("echoargs")
    sess = make_session(tmp_path, agent="hermes-abc123def456")
    await sess.start()
    await drain(sess)
    await sess.aclose()

    argv_line = next((l for l in sess.raw_tail if l.startswith('{"argv"')), None)
    assert argv_line is not None
    argv = json.loads(argv_line)["argv"]
    assert "--agent" in argv and "hermes-abc123def456" in argv


async def test_missing_binary_raises(tmp_path):
    sess = AgySession(
        agy_bin="definitely-not-a-real-binary-xyz",
        model=None, workdir=tmp_path, prompt="x",
    )
    with pytest.raises(RuntimeError, match="agy CLI not found"):
        await sess.start()
