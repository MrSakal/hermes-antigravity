from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agy_session import AgySession  # noqa: E402

FAKE_AGY = str(Path(__file__).with_name("fake_agy.py"))


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
