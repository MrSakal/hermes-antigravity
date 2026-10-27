"""Real subprocess lifecycle regressions.

The shared-config slot-lease tests that lived here went away with the slot
manager itself: the server no longer writes ~/.gemini/config/mcp_config.json
(see tests/test_mcp_config.py for the legacy migration)."""
import asyncio
import sys

import pytest

from app.agy_session import AgySession, STREAM_CLOSED
from app.events import Error
from conftest import kill_pid, pid_alive


def session(tmp_path, monkeypatch, code, **kwargs):
    monkeypatch.setattr(AgySession, '_build_args', lambda self: [sys.executable, '-c', code])
    return AgySession(agy_bin=sys.executable, model=None, workdir=tmp_path,
                      prompt=kwargs.pop('prompt', 'fixture'), **kwargs)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize('exit_parent', [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        sys.platform == 'win32',
        reason='needs POSIX process groups: Windows taskkill /T walks the tree '
               'from a live leader, an orphan of an exited leader is unreachable')),
])
async def test_descendants_die_even_after_leader_exit(tmp_path, monkeypatch, exit_parent):
    code = ('import subprocess,sys,time\n'
            'p=subprocess.Popen([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"],'
            'stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n'
            'print(p.pid,flush=True)\n' + ('' if exit_parent else 'time.sleep(60)\n'))
    sess = session(tmp_path, monkeypatch, code)
    child = None
    try:
        await sess.start()
        await until(lambda: sess.raw_tail)
        child = int(sess.raw_tail[-1])
        if exit_parent:
            await until(lambda: sess._proc.returncode is not None)
        await sess.aclose()
        await until(lambda: not pid_alive(child))
        assert sess._proc.returncode is not None
    finally:
        await sess.aclose()
        if child:
            kill_pid(child)


async def test_full_queue_close_and_eof_do_not_block(tmp_path, monkeypatch):
    code = 'import json\nfor i in range(1000): print(json.dumps({"event":"result","result":{"status":"SUCCESS","response":"x"}}),flush=True)'
    sess = session(tmp_path, monkeypatch, code, event_queue_size=1)
    try:
        await sess.start()
        await until(lambda: sess._queue.full())
        assert sess._queue.qsize() == 1
        await asyncio.wait_for(sess.aclose(), 2)
        await sess.next_event()
        assert await sess.next_event() is STREAM_CLOSED
    finally:
        await sess.aclose()


async def test_stdin_deadline_surfaces_error_and_reaps(tmp_path, monkeypatch):
    sess = session(tmp_path, monkeypatch, 'import time;time.sleep(60)',
                   prompt='x' * 2_000_000, stdin_timeout_s=0.05)
    try:
        await sess.start()
        event = await sess.next_event(timeout=2)
        assert isinstance(event, Error)
        assert 'TimeoutError' in event.message
        await sess.aclose()
        assert sess._proc.returncode is not None
    finally:
        await sess.aclose()


async def test_repeated_cancellation_finishes_one_close(tmp_path, monkeypatch):
    sess = session(tmp_path, monkeypatch, 'import time;time.sleep(60)')
    await sess.start()
    original = sess._close_impl
    entered = asyncio.Event()
    release = asyncio.Event()
    async def delayed():
        entered.set()
        await release.wait()
        await original()
    monkeypatch.setattr(sess, '_close_impl', delayed)
    task = asyncio.create_task(sess.aclose())
    try:
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await sess.aclose()
        assert sess._proc.returncode is not None
    finally:
        release.set()
        await sess.aclose()
        await asyncio.gather(task, return_exceptions=True)


async def test_eof_preserves_full_queue_and_repeats_sentinel(tmp_path, monkeypatch):
    sess = session(tmp_path, monkeypatch,
                   'print(\'{"event":"result","result":{"status":"SUCCESS","response":"last"}}\',flush=True)',
                   event_queue_size=1)
    try:
        await sess.start()
        await until(lambda: sess._eof)
        assert sess._queue.full()
        assert (await sess.next_event()).response_text == 'last'
        assert await sess.next_event() is STREAM_CLOSED
        assert await sess.next_event() is STREAM_CLOSED
    finally:
        await sess.aclose()


async def test_broken_stdin_surfaces_error(tmp_path, monkeypatch):
    sess = session(tmp_path, monkeypatch, 'import os,time;os.close(0);time.sleep(60)',
                   prompt='x' * 2_000_000)
    try:
        await sess.start()
        event = await sess.next_event(timeout=2)
        assert isinstance(event, Error)
        assert 'stdin write failed' in event.message
    finally:
        await sess.aclose()
