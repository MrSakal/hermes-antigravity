from __future__ import annotations

import asyncio
import json

import pytest

from app.agy_session import STREAM_CLOSED, AgySession
from app.events import Error, Init, TextDelta, ToolStep, TurnDone


async def drain(sess: AgySession) -> list:
    """Events of ONE turn: up to its result line (the process persists), or
    up to EOF when the process dies without one."""
    events = []
    while True:
        ev = await sess.next_event(timeout=15)
        assert ev is not None, "timed out waiting for fake agy"
        if ev is STREAM_CLOSED:
            break
        events.append(ev)
        if isinstance(ev, (TurnDone, Error)):
            break
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


async def test_process_survives_turn_and_takes_the_next(fake_agy, tmp_path):
    """stdin stays open after a turn: the same process answers the next one
    on the same agy conversation, and init is printed once, at boot."""
    fake_agy("echo")
    sess = make_session(tmp_path, prompt=None)
    await sess.start()
    assert await sess.wait_started(10)  # init at boot, before any message
    first = await sess.next_event(timeout=5)
    assert isinstance(first, Init)
    sess.send("one")
    t1 = await drain(sess)
    sess.send("two")
    t2 = await drain(sess)
    assert [e.response_text for e in t1 if isinstance(e, TurnDone)] == ["echo: one"]
    assert [e.response_text for e in t2 if isinstance(e, TurnDone)] == ["echo: two"]
    assert not any(isinstance(e, Init) for e in t1 + t2)
    assert sess.running and sess.turns_sent == 2
    await sess.aclose(graceful=True)
    assert sess._proc.returncode == 0  # stdin EOF -> clean exit, not a kill


async def test_persistent_session_disables_print_timeout(fake_agy, tmp_path):
    fake_agy("echoargs")
    sess = AgySession(agy_bin="agy", model=None, workdir=tmp_path, prompt="x")
    await sess.start()
    await drain(sess)
    await sess.aclose()
    argv = json.loads(next(l for l in sess.raw_tail if l.startswith('{"argv"')))["argv"]
    assert argv[argv.index("--print-timeout") + 1] == "0s"


async def test_kill_now_is_synchronous_and_closed_event_fires(fake_agy, tmp_path):
    fake_agy("hang", FAKE_AGY_HANG_S="30")
    sess = make_session(tmp_path)
    await sess.start()
    assert await sess.wait_started(10)
    sess.kill_now()  # no await: must already be on its way down
    await sess.aclose()
    assert sess.closed_event.is_set() and not sess.running
    assert sess._proc.returncode is not None


async def test_injected_items_share_the_event_stream(fake_agy, tmp_path):
    fake_agy("simple")
    sess = make_session(tmp_path, prompt=None)
    await sess.start()
    marker = object()
    waiter = asyncio.create_task(sess.next_event(timeout=10))
    first = await waiter  # boot init
    assert isinstance(first, Init)
    sess.inject(marker)
    assert await sess.next_event(timeout=1) is marker
    await sess.aclose()


@pytest.mark.skipif(not hasattr(__import__("os"), "killpg"), reason="POSIX process groups")
async def test_no_group_kill_after_leader_already_reaped(fake_agy, tmp_path, monkeypatch):
    """Once the leader exited (and was reaped) before teardown began, its pgid
    may belong to an unrelated process group: never killpg it."""
    import os

    fake_agy("simple")
    sess = AgySession(agy_bin="agy", model="m", workdir=tmp_path, prompt=None)
    await sess.start()
    assert await sess.wait_started(10)
    sess._proc.kill()
    await sess._proc.wait()
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    await sess.aclose()
    assert calls == []


@pytest.mark.skipif(not hasattr(__import__("os"), "killpg"), reason="POSIX process groups")
async def test_group_kill_when_leader_alive_at_teardown(fake_agy, tmp_path, monkeypatch):
    import os

    fake_agy("simple")
    sess = AgySession(agy_bin="agy", model="m", workdir=tmp_path, prompt=None)
    await sess.start()
    assert await sess.wait_started(10)
    real = os.killpg
    calls = []

    def spy(pgid, sig):
        calls.append(sig)
        real(pgid, sig)
    monkeypatch.setattr(os, "killpg", spy)
    await sess.aclose()
    assert calls  # SIGKILL to the group while the leader was alive


async def test_boot_eof_sets_boot_failed_with_stderr(fake_agy, tmp_path):
    fake_agy("bootfail")
    sess = AgySession(agy_bin="agy", model="m", workdir=tmp_path, prompt=None)
    await sess.start()
    assert await sess.wait_started(10)  # EOF releases the waiter...
    assert sess.boot_failed  # ...but it is a failed boot
    assert "not signed in" in await sess.exit_diagnostics()
    await sess.aclose()


@pytest.mark.skipif(not __import__("os").path.isdir("/proc"), reason="Linux /proc")
def test_recycled_pgid_is_never_considered_ours(tmp_path):
    """A live process whose PID equals our (reaped) leader's pgid means the
    number was recycled: never treat that group as ours."""
    import os

    sess = AgySession(agy_bin="agy", model="m", workdir=tmp_path, prompt=None)
    sess._pgid = os.getpid()  # a live process holds that PID now
    assert sess._orphaned_group_alive() is False
    sess._pgid = None
    assert sess._orphaned_group_alive() is False
