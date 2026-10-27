from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agy_session import AgySession  # noqa: E402

FAKE_AGY = str(Path(__file__).with_name("fake_agy.py"))


@pytest.fixture(autouse=True)
def _no_warm_pool_by_default(monkeypatch):
    """The production default is a 1-process warm pool; unit tests opt in
    explicitly (``warm_pool_size=1``) so background boots never race the
    process accounting other tests assert on."""
    monkeypatch.setenv("AGI_WARM_POOL_SIZE", "0")


@pytest.fixture
def fake_agy(monkeypatch):
    """Route AgySession spawns to tests/fake_agy.py.

    AgySession execs its binary directly, so we rewrite the argv it builds:
    ``[agy_bin, ...flags]`` becomes ``[python, fake_agy.py, ...flags]``. The
    fake reads FAKE_AGY_SCRIPT to pick a canned behavior; the original flags
    stay in argv for the "echoargs" mode to assert on.
    """
    orig = AgySession._build_args

    def patched(self):
        args = orig(self)
        return [sys.executable, FAKE_AGY] + args[1:]

    monkeypatch.setattr(AgySession, "_build_args", patched)

    def set_mode(mode: str, **env: str) -> None:
        monkeypatch.setenv("FAKE_AGY_SCRIPT", mode)
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    return set_mode


@pytest.fixture
def fake_agy_bin(tmp_path) -> str:
    """An executable wrapper so code that execs ``agy_bin`` directly (the
    model catalog's ``agy models``) hits tests/fake_agy.py too."""
    wrapper = tmp_path / "agy-fake"
    if os.name == "nt":  # pragma: no cover
        wrapper = wrapper.with_suffix(".cmd")
        wrapper.write_text(f'@"{sys.executable}" "{FAKE_AGY}" %*\n')
    else:
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_AGY}" "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return str(wrapper)


def pid_alive(pid: int) -> bool:
    """True while ``pid`` is a live (non-zombie) process — portable, and safe:
    on Windows ``os.kill(pid, 0)`` would TerminateProcess it."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def kill_pid(pid: int) -> None:
    """Best-effort fixture cleanup of a leaked test child."""
    import signal

    try:
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except OSError:
        pass
