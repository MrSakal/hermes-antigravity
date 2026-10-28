"""Async ``agy`` CLI driver — one PERSISTENT subprocess per conversation.

agy (>= 1.1.15; verified on 1.4.8) has a stdin mode equivalent to ``claude -p
--input-format stream-json``: ``--print "" --input-format stream-json
--output-format stream-json``. Each stdin line
``{"event":"user","message":{"role":"user","content":<str>}}`` runs ONE turn,
which ends with exactly one ``result`` line; the process then waits for the
next line on the same agy conversation (context kept). stdin EOF → exit 0.
Verified facts this module relies on:

* The ``init`` line is printed at BOOT (≈1–3 s after spawn), before any stdin
  message, and the ``--agent`` file is read before it. :meth:`wait_started`
  is the message-free proof a spawn got that far (see app.agent_config).
* Content must be TEXT only — any other block errors the turn AND exits the
  process (rc=1). :meth:`send` therefore only takes a string; images travel
  as workdir files referenced in the text.
* No per-turn cancel (``control_request`` is fatal): aborting a turn means
  killing the process. Model / effort / agent / ``--add-dir`` are fixed per
  process.
* ``--print-timeout`` counts from the first prompt; a persistent process is
  spawned with ``0s`` (disabled) and our own per-turn deadlines bound it.

The prompt travels over stdin, so there is no argv ceiling (Linux
``MAX_ARG_STRLEN`` 128 KiB, Windows ~32 K chars). ``prompt=None`` spawns and
lets the process boot without a turn (the warm pool); :meth:`send` delivers
turns later, as many as the conversation needs.

A reader task parses stdout JSONL lines into :data:`~app.events.ChatEvent`
values on a bounded :class:`asyncio.Queue`. The MCP bridge pushes its
:class:`~app.mcp_bridge.PendingCall` items into the SAME stream via
:meth:`inject`, so a turn loop awaits one source instead of racing two tasks
per event. :data:`STREAM_CLOSED` marks stdout EOF, which in stdin mode means
the process DIED (a clean turn end is a ``result`` line, never EOF).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import time

from collections import deque
from pathlib import Path
from typing import Any, Optional, Union

import anyio

from app.events import ChatEvent, Error, Init, parse_line
from app.timing import log_spawn

logger = logging.getLogger("agi.session")

# Sentinel returned once the CLI's stdout reached EOF — in stdin mode that
# means the process exited (crash, kill, or stdin closed).
STREAM_CLOSED = object()

# StreamReader line buffer. Tool outputs embedded in step_update lines can be
# sizeable; keep the cap generous and skip — rather than crash on — any
# pathologically long line.
_READ_LIMIT = 64 * 1024 * 1024

# How long a graceful close (IDLE process) waits for agy to exit on stdin EOF
# before it is killed.
_GRACEFUL_EXIT_S = 2.0


def stdin_payload(prompt: str) -> bytes:
    """One NDJSON user line (``--input-format stream-json``) = one turn.

    ``errors="replace"``: client JSON may carry a lone surrogate that strict
    UTF-8 refuses — raising here would fail an already claimed turn, while one
    ``?`` in the prompt is harmless."""
    return (
        json.dumps(
            {"event": "user", "message": {"role": "user", "content": prompt}},
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8", "replace")


class AgySession:
    """Drives one persistent ``agy`` stdin-mode subprocess."""

    def __init__(
        self,
        *,
        agy_bin: str,
        model: Optional[str],
        workdir: Union[str, Path],
        prompt: Optional[str] = None,
        agent: Optional[str] = None,
        print_timeout_s: int = 0,
        disable_slash_commands: bool = True,
        timing_log: bool = False,
        timing_label: str = "session",
        event_queue_size: int = 256,
        stdin_timeout_s: float = 30,
        observer: Optional[Any] = None,
    ) -> None:
        if event_queue_size < 1 or stdin_timeout_s <= 0:
            raise ValueError("event_queue_size and stdin_timeout_s must be positive")
        self.event_queue_size = event_queue_size
        self.stdin_timeout_s = stdin_timeout_s
        self.agy_bin = agy_bin
        self.model = model
        self.workdir = str(workdir)
        self.prompt = prompt
        # Learned from the init / result lines (see ConversationManager.run_turn).
        self.conversation_id: Optional[str] = None
        self.agent = agent
        # 0 = disabled: a persistent process idles between turns and blocks
        # inside our MCP call_tool; the turn loop enforces its own deadline.
        self.print_timeout_s = print_timeout_s
        self.disable_slash_commands = disable_slash_commands
        self.timing_log = timing_log
        self.timing_label = timing_label
        # app.housekeeping.Housekeeper (or None): per-process --log-file,
        # ownership ledger, deletion of the agy state once the process is gone.
        self.observer = observer
        self.log_file: Optional[str] = None
        self._saw_error = False

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._queue: asyncio.Queue[Union[ChatEvent, object]] = asyncio.Queue(maxsize=event_queue_size)
        # Items pushed by the MCP bridge (PendingCall). Unbounded here: the
        # bridge enforces max_pending_tool_calls before it injects.
        self._injected: deque[object] = deque()
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._stdin_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._eof = False
        self.turns_sent = 0
        self.spawned_at: Optional[float] = None
        self._available = asyncio.Event()
        # Set on the first stdout line (or EOF): agy has read its --agent file
        # by then (see app.agent_config), so its transient file may be deleted.
        self._started = asyncio.Event()
        # True once a real stdout line arrived. EOF without one = agy died at
        # boot (logged out, bad --model, rejected agent frontmatter): see
        # :attr:`boot_failed`.
        self._got_line = False
        self._writer_error: Optional[Error] = None
        self._close_task: Optional[asyncio.Task[None]] = None
        # Teardown mode read by _close_impl: graceful = close stdin and let agy
        # exit on its own first (an IDLE process); otherwise straight to the kill.
        self._graceful_close = False
        # Windows: the ``taskkill /T /F`` started by kill_now().
        self._killer: Optional[subprocess.Popen] = None
        self._pgid: Optional[int] = None
        self._pushback: deque[Union[ChatEvent, object]] = deque()
        # Set once teardown finished and the process is gone.
        self.closed_event = asyncio.Event()
        # Diagnostics: last raw stdout lines the parser skipped + stderr tail.
        self.raw_tail: deque[str] = deque(maxlen=5)
        self.stderr_tail: deque[str] = deque(maxlen=5)

    # ── command construction ────────────────────────────────────────────────

    def _build_args(self) -> list[str]:
        # `--print ""` selects print mode with no argv prompt; the real prompt
        # arrives on stdin as stream-json. Order matters: agy's flag parser
        # treats the token after -p/--print as the prompt value.
        args = [
            self.agy_bin,
            "--print", "",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--print-timeout", f"{max(0, int(self.print_timeout_s))}s",
            "--add-dir", self.workdir,
        ]
        if self.disable_slash_commands:
            # User content is data, never a slash command / skill invocation.
            args.append("--disable-slash-commands")
        if self.model:
            args += ["--model", self.model]
        if self.agent:
            args += ["--agent", self.agent]
        if self.log_file:
            # Our own per-process log instead of a new acli/log/cli-*.log
            # (verified on 1.4.8); deleted when the process ends.
            args += ["--log-file", self.log_file]
        return args

    # ── lifecycle ─────────────────────────────────────────────────────────—

    async def start(self) -> None:
        """Spawn the subprocess (idempotent) and launch reader/stderr tasks.

        Feeds the prompt immediately when one was given at construction;
        otherwise the process idles until :meth:`send`."""
        if self._closed:
            raise RuntimeError("session is closed")
        if self._proc is not None:
            return
        if self.observer is not None and self.log_file is None:
            self.log_file = self.observer.log_file_for(self)
        args = self._build_args()
        logger.info(
            "spawning agy: model=%s workdir=%s agent=%s deferred=%s",
            self.model, self.workdir, self.agent, self.prompt is None,
        )
        spawn_t0 = time.monotonic()
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workdir,
                limit=_READ_LIMIT,
                # POSIX: own process group, so one killpg reaches agy's
                # children even after the leader exited.
                start_new_session=(os.name == "posix"),
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "agy CLI not found on PATH — install the Antigravity CLI "
                "(https://antigravity.google/docs/cli-install) and sign in"
            ) from e
        if os.name == "posix":
            self._pgid = self._proc.pid
        if self.observer is not None:
            self.observer.on_spawn(self, self._proc.pid, self.log_file)
        self.spawned_at = time.monotonic()
        log_spawn(self.timing_log, self.timing_label, (time.monotonic() - spawn_t0) * 1000)
        self._reader_task = asyncio.create_task(self._read_stdout(), name="agy-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="agy-stderr")
        if self.prompt is not None:
            self.send(self.prompt)

    def send(self, prompt: str) -> None:
        """Start one turn: write a user line to stdin. stdin stays OPEN, so the
        process survives the turn and takes the next one on the same agy
        conversation.

        The write runs in the background: before boot agy does not drain stdin,
        and a payload larger than the pipe buffer would otherwise block the
        caller (and the HTTP response) for that long. Successive writes are
        chained, never interleaved."""
        if self._proc is None:
            raise RuntimeError("session not started")
        if not isinstance(prompt, str):
            raise TypeError("agy stdin mode accepts text content only")
        if not self.running:
            raise RuntimeError(f"agy process is not running ({self.diagnostics()})")
        self.prompt = prompt
        self.turns_sent += 1
        payload = stdin_payload(prompt)
        logger.debug("agy stdin: %d bytes (turn %d)", len(payload), self.turns_sent)
        self._stdin_task = asyncio.create_task(
            self._feed_stdin(payload, self._stdin_task), name="agy-stdin")

    async def wait_started(self, timeout: float) -> bool:
        """Wait until the process printed its first stdout line (or exited).

        Returns False on timeout. EOF also returns True (the process can no
        longer read anything, so its agent file may be deleted) — callers
        must then check :attr:`boot_failed`, which tells a dead boot from a live
        one. Already-queued events count as started so test doubles that fill
        the queue without a real process pass through."""
        if self._started.is_set() or self._eof or not self._queue.empty() or self._pushback:
            return True
        if self._proc is None:
            return True  # nothing was spawned, so nothing can read the agent file
        try:
            async with asyncio.timeout(timeout):
                await self._started.wait()
        except TimeoutError:
            return False
        return True

    @property
    def boot_failed(self) -> bool:
        """stdout reached EOF before agy printed a single line: the process
        died during boot and never served anything."""
        return self._eof and not self._got_line and self._proc is not None

    async def exit_diagnostics(self, timeout: float = 1.0) -> str:
        """:meth:`diagnostics` once the dead process' stderr is drained and its
        exit code is known (bounded by ``timeout``) — the stderr tail is what
        says WHY agy refused to boot (not signed in, unknown model, bad agent)."""
        proc, stderr_task = self._proc, self._stderr_task
        waiter = None
        try:
            waits: list[asyncio.Future] = []
            if stderr_task is not None and not stderr_task.done():
                waits.append(stderr_task)
            if proc is not None and proc.returncode is None:
                waiter = asyncio.ensure_future(proc.wait())
                waits.append(waiter)
            if waits:
                await asyncio.wait(waits, timeout=timeout)
        finally:
            if waiter is not None and not waiter.done():
                waiter.cancel()
        return self.diagnostics()

    @property
    def running(self) -> bool:
        return (self._proc is not None and self._proc.returncode is None
                and not self._closed and not self._eof)

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @property
    def idle_s(self) -> float:
        """Seconds since spawn; meaningful for pooled, not-yet-prompted procs."""
        return 0.0 if self.spawned_at is None else time.monotonic() - self.spawned_at

    async def _feed_stdin(self, payload: bytes, previous: Optional[asyncio.Task]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        if previous is not None and not previous.done():
            with contextlib.suppress(BaseException):
                await previous
        stdin = self._proc.stdin
        try:
            async with asyncio.timeout(self.stdin_timeout_s):
                for offset in range(0, len(payload), 65536):
                    stdin.write(payload[offset:offset + 65536])
                    await stdin.drain()
        except Exception as e:
            message = f"stdin write failed: {type(e).__name__}: {e}"
            self.stderr_tail.append(message)
            self._writer_error = Error(message=message)
            self._available.set()
            # A blocked/failed writer must also stop the upstream process.
            self.kill_now()
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_impl())

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        try:
            while True:
                try:
                    raw = await stdout.readline()
                except ValueError:
                    # A line longer than the buffer cap: asyncio re-raises the
                    # LimitOverrunError as ValueError with the buffer already
                    # advanced — skip the line and keep reading.
                    logger.warning("skipped oversized stdout line (> %d bytes)", _READ_LIMIT)
                    continue
                except asyncio.IncompleteReadError:
                    break
                if not raw:
                    break
                self._got_line = True
                self._started.set()
                line = raw.decode("utf-8", "replace")
                ev = parse_line(line)
                if ev is not None:
                    if isinstance(ev, Init) and ev.conversation_id:
                        self._on_init(ev.conversation_id)
                    elif isinstance(ev, Error):
                        self._saw_error = True
                    if self._closed:
                        break
                    await self._queue.put(ev)
                    self._available.set()
                elif line.strip():
                    self.raw_tail.append(line.strip()[:500])
        finally:
            self._eof = True
            self._started.set()
            self._available.set()

    def _on_init(self, conversation_id: str) -> None:
        """Every ``init`` line: record the agy conversation id in the ownership
        ledger. :attr:`conversation_id` itself is learned by the turn loop
        (thinking passthrough tells a fresh process from a reused one by it)."""
        if self.observer is not None:
            try:
                self.observer.on_init(self, conversation_id)
            except Exception:  # pragma: no cover - defensive
                logger.warning("housekeeping on_init failed", exc_info=True)

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        stderr = self._proc.stderr
        while True:
            try:
                raw = await stderr.readline()
            except (asyncio.LimitOverrunError, ValueError):
                continue
            if not raw:
                break
            msg = raw.decode("utf-8", "replace").rstrip()
            if msg:
                self.stderr_tail.append(msg[:500])
                logger.debug("agy stderr: %s", msg)

    # ── reading ───────────────────────────────────────────────────────────—

    def inject(self, item: object) -> None:
        """Push a non-stdout item (an MCP PendingCall) into the event stream."""
        self._injected.append(item)
        self._available.set()

    def push_back(self, ev: Union[ChatEvent, object]) -> None:
        """Return an already-popped event to the front of the stream (FIFO
        among pushed-back items)."""
        self._pushback.append(ev)

    def _take_nowait(self) -> Union[ChatEvent, object, None]:
        if self._writer_error is not None:
            error, self._writer_error = self._writer_error, None
            return error
        if self._pushback:
            return self._pushback.popleft()
        if not self._queue.empty():
            return self._queue.get_nowait()
        if self._injected:
            return self._injected.popleft()
        if self._eof or self._closed:
            return STREAM_CLOSED
        return None

    async def next_event(
        self, timeout: Optional[float] = None
    ) -> Union[ChatEvent, object, None]:
        """Pop the next item: a ChatEvent, an injected item, or
        :data:`STREAM_CLOSED` at EOF. Returns ``None`` if ``timeout`` elapses
        first. The fast path (something already queued) arms no timer."""
        item = self._take_nowait()
        if item is not None:
            return item
        if timeout is not None and timeout <= 0:
            return None
        try:
            async with asyncio.timeout(timeout):
                while True:
                    self._available.clear()
                    item = self._take_nowait()
                    if item is not None:
                        return item
                    await self._available.wait()
        except TimeoutError:
            return None

    def diagnostics(self) -> str:
        """A short description of why the stream may have ended without a result."""
        parts: list[str] = []
        if self.raw_tail:
            parts.append("stdout: " + " | ".join(self.raw_tail))
        if self.stderr_tail:
            parts.append("stderr: " + " | ".join(self.stderr_tail))
        if self._proc is not None and self._proc.returncode is not None:
            parts.append(f"exit code {self._proc.returncode}")
        return "; ".join(parts) if parts else "no diagnostics captured"

    # ── shutdown ──────────────────────────────────────────────────────────—

    def kill_now(self) -> None:
        """Issue the kill SYNCHRONOUSLY (no await) — the one teardown step that
        must happen even while the caller is being cancelled. Idempotent.

        POSIX: SIGKILL to the whole process group (agy + its children).
        Windows: start ``taskkill /PID <pid> /T /F`` without waiting for it (a
        tree kill; TerminateProcess on the leader alone orphans agy's
        children). Reaping is left to :meth:`aclose`."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        if os.name == "nt":
            if self._killer is None:
                try:
                    self._killer = subprocess.Popen(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except OSError:
                    with contextlib.suppress(OSError):
                        proc.kill()
            return
        self._signal_owned(signal.SIGKILL)

    async def aclose(self, *, graceful: bool = False) -> None:
        """Stop the process and its reader tasks; finish the shared teardown
        even if callers are repeatedly cancelled.

        ``graceful`` (an IDLE process between turns): close stdin and give agy
        a moment to exit 0 on its own. Otherwise (abort mid-turn — agy has no
        per-turn cancel) the kill is issued synchronously, before any await, so
        a cancellation landing right now cannot leave the process alive."""
        if not graceful and self._close_task is None:
            self.kill_now()
        with anyio.CancelScope(shield=True):
            if self._close_task is None:
                self._graceful_close = graceful
                self._close_task = asyncio.create_task(self._close_impl())
            cancelled = False
            while True:
                try:
                    await asyncio.shield(self._close_task)
                    break
                except asyncio.CancelledError:
                    if self._close_task.cancelled():
                        raise  # never spin on an independently cancelled teardown
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError

    def _signal_owned(self, sig: int, *, group_seen_alive: bool = False) -> None:
        """POSIX: signal the whole process group (agy + its children).

        Once the leader has exited and been reaped, its pgid is only ours while
        some member of the group survives — after that the kernel may hand the
        number to an unrelated new session. So the group is signalled only
        while the leader is alive, or when the caller observed it alive during
        this very teardown (``group_seen_alive``: the leader was alive when
        teardown began, or :meth:`_orphaned_group_alive` just found members of
        the group — lingering children pin the id)."""
        try:
            leader_alive = self._proc is not None and self._proc.returncode is None
            if self._pgid is not None and (leader_alive or group_seen_alive):
                os.killpg(self._pgid, sig)
            elif self._proc is not None and self._proc.returncode is None:
                self._proc.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _orphaned_group_alive(self) -> bool:
        """Linux: our leader is gone — do members of its process group (agy's
        orphaned children) still exist? While one does, the kernel cannot
        recycle the pgid, so it is still ours. A live process whose PID equals
        the pgid is NOT ours (our leader was reaped): that number was recycled,
        so the answer is no. Without /proc (other POSIX) the group is never
        signalled after the leader exited."""
        pgid = self._pgid
        if pgid is None or not os.path.isdir("/proc"):
            return False
        members = False
        try:
            entries = os.listdir("/proc")
        except OSError:
            return False
        for name in entries:
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as f:
                    fields = f.read().rsplit(b")", 1)[1].split()
            except (OSError, IndexError):
                continue
            # After "(comm)": state ppid pgrp session ...
            if int(name) == pgid:
                return False
            if len(fields) > 2 and fields[2] == str(pgid).encode():
                members = True
        return members

    async def _kill_tree_windows(self, proc: asyncio.subprocess.Process) -> None:
        """Windows has no process groups or SIGTERM for console apps
        (``send_signal(SIGTERM)`` is TerminateProcess on the leader only, which
        orphans agy's children). ``taskkill /T /F`` walks the tree from the
        leader — so it must run while the leader is still alive. The Popen is
        synchronous so a cancellation cannot interrupt it half-way; its exit is
        then awaited off-loop."""
        if proc.returncode is None and self._killer is None:
            try:
                self._killer = subprocess.Popen(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError:
                logger.warning("taskkill unavailable; killing the agy leader only", exc_info=True)
        if self._killer is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                await asyncio.to_thread(self._killer.wait, 10)
        if proc.returncode is None:
            # Already gone (ProcessLookupError) or access denied while exiting.
            with contextlib.suppress(OSError):
                proc.kill()

    async def _close_impl(self) -> None:
        self._closed = True
        self._available.set()
        try:
            proc = self._proc
            # The leader was alive when teardown began: the group id is still
            # ours for the final group SIGKILL below (see _signal_owned).
            seen_alive = proc is not None and (proc.returncode is None
                                               or self._orphaned_group_alive())
            if (self._graceful_close and proc is not None and proc.returncode is None
                    and self._killer is None and proc.stdin is not None):
                # Graceful: EOF on stdin ends agy's read loop; it exits 0.
                with contextlib.suppress(Exception):
                    proc.stdin.close()
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(_GRACEFUL_EXIT_S):
                        await proc.wait()
            tasks = [t for t in (self._reader_task, self._stderr_task, self._stdin_task)
                     if t is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if proc is not None and os.name == "nt":
                await self._kill_tree_windows(proc)
                await proc.wait()
            elif proc is not None:
                if proc.returncode is None:
                    self._signal_owned(signal.SIGTERM)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        pass
                # The leader may already have exited while descendants remain.
                # Skipped when it was already gone before teardown: that pgid
                # may have been recycled by now.
                self._signal_owned(signal.SIGKILL, group_seen_alive=seen_alive)
                self._pgid = None
                await proc.wait()
        finally:
            if self.observer is not None:
                try:
                    self.observer.on_exit(self, failed=bool(self.boot_failed or self._saw_error))
                except Exception:  # pragma: no cover - defensive
                    logger.warning("housekeeping on_exit failed", exc_info=True)
            self.closed_event.set()
