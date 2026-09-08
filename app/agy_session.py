"""Async ``agy`` CLI driver — one subprocess per turn.

Unlike the `claude` CLI (bidirectional stream-json over a persistent stdin/stdout
process), `agy` is invoked once per turn: the process runs the turn to
completion and exits. Conversation continuity is server-side in agy:
``--conversation <id>`` resumes the stored history, so a follow-up turn passes
only the newest user message.

The prompt travels over **stdin** as one NDJSON line
(``{"event":"user","message":{"role":"user","content":...}}``) using
``--input-format stream-json`` (agy >= 1.1.15; verified against 1.1.27). This
removes the argv size ceiling (Linux ``MAX_ARG_STRLEN`` = 128 KiB per argument,
Windows ~32 K chars) that the previous ``-p <prompt>`` transport hit — prompts of
several hundred KB are delivered intact. ``--print`` still has to be given (with
an empty value) to select print mode; ``-p --input-format`` would swallow the
flag as the prompt. stdin is closed after the single line, so agy runs exactly
one turn and exits.

The prompt may be deferred: ``prompt=None`` spawns the process and lets it boot
(~3.5 s on this host, measured against 1.1.27) without a user turn; a later
:meth:`send` delivers the NDJSON line. ``--print-timeout`` starts counting at
the first prompt, not at boot (verified: an idle process outlived a 15 s
timeout), which is what makes :mod:`app.warmpool` possible.

A reader task parses stdout JSONL lines into :data:`~app.events.ChatEvent`
values pushed onto an :class:`asyncio.Queue`; :data:`STREAM_CLOSED` marks EOF
(process exit). The last few unparseable stdout lines and a stderr tail are kept
for diagnostics (agy prints some errors as plain text, e.g. the headless
permission notice).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import time

from collections import deque
from pathlib import Path
from typing import Optional, Union

import anyio

from app.events import ChatEvent, Error, parse_line
from app.timing import log_spawn

logger = logging.getLogger("agi.session")

# Sentinel placed on the event queue when the CLI's stdout reaches EOF (the
# process exited). Consumers stop waiting for turn events when they see it.
STREAM_CLOSED = object()

# StreamReader line buffer. Tool outputs embedded in step_update lines can be
# sizeable; keep the cap generous and skip — rather than crash on — any
# pathologically long line.
_READ_LIMIT = 64 * 1024 * 1024


def stdin_payload(prompt: str) -> bytes:
    """The single NDJSON line agy reads as the user turn (``--input-format
    stream-json``). Shape verified against agy 1.1.27; any other ``event``
    value is rejected by the CLI with ``stream input message is missing the
    "event" field`` or similar."""
    return (
        json.dumps(
            {"event": "user", "message": {"role": "user", "content": prompt}},
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


class AgySession:
    """Drives one ``agy`` subprocess for a single turn."""

    def __init__(
        self,
        *,
        agy_bin: str,
        model: Optional[str],
        workdir: Union[str, Path],
        prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
        agent: Optional[str] = None,
        print_timeout_s: int = 600,
        disable_slash_commands: bool = True,
        timing_log: bool = False,
        timing_label: str = "session",
        event_queue_size: int = 256,
        stdin_timeout_s: float = 30,
    ) -> None:
        if event_queue_size < 1 or stdin_timeout_s <= 0:
            raise ValueError("event_queue_size and stdin_timeout_s must be positive")
        self.event_queue_size = event_queue_size
        self.stdin_timeout_s = stdin_timeout_s
        self.agy_bin = agy_bin
        self.model = model
        self.workdir = str(workdir)
        self.prompt = prompt
        self.conversation_id = conversation_id
        self.agent = agent
        self.print_timeout_s = print_timeout_s
        self.disable_slash_commands = disable_slash_commands
        self.timing_log = timing_log
        self.timing_label = timing_label

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._queue: asyncio.Queue[Union[ChatEvent, object]] = asyncio.Queue(maxsize=event_queue_size)
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._stdin_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._eof = False
        self._sent = False
        self.spawned_at: Optional[float] = None
        self._available = asyncio.Event()
        self._writer_error: Optional[Error] = None
        self._close_task: Optional[asyncio.Task[None]] = None
        self._pgid: Optional[int] = None
        self._pushback: list[Union[ChatEvent, object]] = []
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
            "--print-timeout", f"{self.print_timeout_s}s",
            "--add-dir", self.workdir,
        ]
        if self.disable_slash_commands:
            # User content is data, never a slash command / skill invocation.
            args.append("--disable-slash-commands")
        if self.model:
            args += ["--model", self.model]
        if self.conversation_id:
            args += ["--conversation", self.conversation_id]
        if self.agent:
            args += ["--agent", self.agent]
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
        args = self._build_args()
        logger.info(
            "spawning agy: model=%s workdir=%s conv=%s agent=%s deferred=%s",
            self.model, self.workdir, self.conversation_id, self.agent, self.prompt is None,
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
                start_new_session=(os.name == "posix"),
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "agy CLI not found on PATH — install the Antigravity CLI "
                "(https://antigravity.google/docs/cli-install) and sign in"
            ) from e
        if os.name == "posix":
            self._pgid = self._proc.pid
        self.spawned_at = time.monotonic()
        log_spawn(self.timing_log, self.timing_label, (time.monotonic() - spawn_t0) * 1000)
        self._reader_task = asyncio.create_task(self._read_stdout(), name="agy-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="agy-stderr")
        if self.prompt is not None:
            self.send(self.prompt)

    def send(self, prompt: str) -> None:
        """Deliver the single user turn. Exactly once per process.

        Runs in the background: agy takes seconds to boot before it drains
        stdin, and a payload larger than the pipe buffer would otherwise block
        the caller (and the HTTP response) for that long."""
        if self._proc is None:
            raise RuntimeError("session not started")
        if self._sent:
            raise RuntimeError("prompt already sent")
        self._sent = True
        self.prompt = prompt
        payload = stdin_payload(prompt)
        logger.debug("agy stdin: %d bytes", len(payload))
        self._stdin_task = asyncio.create_task(self._feed_stdin(payload), name="agy-stdin")

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def idle_s(self) -> float:
        """Seconds since spawn; meaningful for pooled, not-yet-prompted procs."""
        return 0.0 if self.spawned_at is None else time.monotonic() - self.spawned_at

    async def _feed_stdin(self, payload: bytes) -> None:
        assert self._proc is not None and self._proc.stdin is not None
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
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_impl())
        finally:
            with contextlib.suppress(Exception):
                stdin.close()

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
                line = raw.decode("utf-8", "replace")
                ev = parse_line(line)
                if ev is not None:
                    if self._closed:
                        break
                    await self._queue.put(ev)
                    self._available.set()
                elif line.strip():
                    self.raw_tail.append(line.strip()[:500])
        finally:
            self._eof = True
            self._available.set()

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

    def push_back(self, ev: Union[ChatEvent, object]) -> None:
        """Return an already-popped event to the front of the stream."""
        self._pushback.append(ev)

    async def next_event(
        self, timeout: Optional[float] = None
    ) -> Union[ChatEvent, object, None]:
        """Pop the next event. Returns :data:`STREAM_CLOSED` at EOF, or ``None``
        if ``timeout`` elapses first."""
        async def take():
            while True:
                if self._writer_error is not None:
                    error, self._writer_error = self._writer_error, None
                    return error
                if self._pushback:
                    return self._pushback.pop(0)
                if not self._queue.empty():
                    return self._queue.get_nowait()
                if self._eof or self._closed:
                    return STREAM_CLOSED
                self._available.clear()
                await self._available.wait()
        try:
            async with asyncio.timeout(timeout):
                return await take()
        except TimeoutError:
            return None

    def diagnostics(self) -> str:
        """A short description of why the stream may have ended without a result."""
        parts: list[str] = []
        if self.raw_tail:
            parts.append("stdout: " + " | ".join(self.raw_tail))
        if self.stderr_tail:
            parts.append("stderr: " + " | ".join(self.stderr_tail))
        return "; ".join(parts) if parts else "no diagnostics captured"

    # ── shutdown ──────────────────────────────────────────────────────────—

    async def aclose(self) -> None:
        """Finish shared teardown even if callers are repeatedly cancelled."""
        with anyio.CancelScope(shield=True):
            if self._close_task is None:
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

    def _signal_owned(self, sig: int) -> None:
        try:
            if self._pgid is not None:
                os.killpg(self._pgid, sig)
            elif self._proc is not None and self._proc.returncode is None:
                self._proc.send_signal(sig)
        except ProcessLookupError:
            pass

    async def _close_impl(self) -> None:
        self._closed = True
        self._available.set()
        tasks = [t for t in (self._reader_task, self._stderr_task, self._stdin_task)
                 if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        proc = self._proc
        if proc is not None:
            self._signal_owned(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            finally:
                # The leader may already have exited while descendants remain.
                self._signal_owned(signal.SIGKILL)
                self._pgid = None
            await proc.wait()
