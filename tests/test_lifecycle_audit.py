"""Real subprocess and isolated shared-config lifecycle regressions."""
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import sys

import pytest

from app.agy_session import AgySession, STREAM_CLOSED
from app.events import Error
from app.mcp_config import McpConfigManager


def session(tmp_path, monkeypatch, code, **kwargs):
    monkeypatch.setattr(AgySession, '_build_args', lambda self: [sys.executable, '-c', code])
    return AgySession(agy_bin=sys.executable, model=None, workdir=tmp_path,
                      prompt=kwargs.pop('prompt', 'fixture'), **kwargs)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize('exit_parent', [False, True])
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
        stat = Path(f'/proc/{child}/stat')
        await until(lambda: not stat.exists() or stat.read_text().split()[2] == 'Z')
        assert sess._proc.returncode is not None
    finally:
        await sess.aclose()
        if child:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


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


async def test_replaced_slot_survives_cleanup(tmp_path):
    path = tmp_path / 'mcp.json'
    mgr = McpConfigManager(path)
    slot = await mgr.add_conversation('mine', 'http://127.0.0.1:1234/mcp/mine')
    replacement = {'serverUrl': 'http://foreign.example/mcp/new'}
    path.write_text(json.dumps({'mcpServers': {slot: replacement}}))
    await mgr.remove_all()
    assert json.loads(path.read_text())['mcpServers'][slot] == replacement


async def test_other_manager_lease_and_live_listener_preserved(tmp_path):
    path = tmp_path / 'mcp.json'
    owner, other = McpConfigManager(path), McpConfigManager(path)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        slot = await owner.add_conversation('active', f'http://127.0.0.1:{port}/mcp/active')
        try:
            assert await other.cleanup_stale(port) == 0
            data = json.loads(path.read_text())
            data['mcpServers']['hermes9'] = {'serverUrl': f'http://127.0.0.1:{port}/mcp/legacy'}
            path.write_text(json.dumps(data))
            sock.listen()
            assert await other.cleanup_stale(port) == 0
            assert slot in json.loads(path.read_text())['mcpServers']
        finally:
            await owner.remove_all()


@pytest.mark.parametrize('raw', ['{bad', '[]', '{"mcpServers":[]}'])
async def test_corrupt_config_never_overwritten(tmp_path, raw):
    path = tmp_path / 'mcp.json'
    path.write_text(raw)
    mgr = McpConfigManager(path)
    with pytest.raises(ValueError):
        await mgr.add_conversation('x', 'http://x')
    assert path.read_text() == raw


async def test_processes_allocate_without_lost_updates(tmp_path):
    path = tmp_path / 'mcp.json'
    code = '''import asyncio,sys,time
from pathlib import Path
from app.mcp_config import McpConfigManager
async def main():
 m=McpConfigManager(Path(sys.argv[1]))
 for i in range(15): await m.add_conversation(str(i), 'http://fixture/'+sys.argv[2]+'/'+str(i))
asyncio.run(main())
'''
    procs = []
    try:
        for n in range(4):
            procs.append(await asyncio.create_subprocess_exec(sys.executable, '-c', code, str(path), str(n)))
        assert await asyncio.gather(*(p.wait() for p in procs)) == [0]*4
        servers = json.loads(path.read_text())['mcpServers']
        assert len(servers) == 60
        assert len({s['serverUrl'] for s in servers.values()}) == 60
    finally:
        for proc in procs:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()


async def test_cancelled_lock_wait_leaves_config_and_lock_usable(tmp_path):
    import fcntl
    path = tmp_path / 'mcp.json'
    path.write_text('{"mcpServers":{}}')
    mgr = McpConfigManager(path)
    with open(str(path) + '.lock', 'w') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        task = asyncio.create_task(mgr.add_conversation('cancelled', 'http://fixture/cancelled'))
        await asyncio.sleep(0.03)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert json.loads(path.read_text()) == {'mcpServers': {}}
    await mgr.add_conversation('next', 'http://fixture/next')
    await mgr.remove_all()
    assert json.loads(path.read_text()) == {'mcpServers': {}}


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


async def test_unreadable_config_is_not_replaced(tmp_path, monkeypatch):
    path = tmp_path / 'mcp.json'
    mgr = McpConfigManager(path)
    slot = await mgr.add_conversation('mine', 'http://fixture/mine')
    saved = path.read_bytes()
    original = Path.read_text
    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError('fixture denied')
        return original(self, *args, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'read_text', denied)
            with pytest.raises(PermissionError):
                await mgr.remove_conversation(slot)
        assert path.read_bytes() == saved
        assert slot in mgr._owned
    finally:
        await mgr.remove_all()
