#!/usr/bin/env python3
"""Smoke test: drive the real `agy` CLI for two turns via AgySession.

agy is one subprocess per turn (the prompt travels in argv), so unlike the
hermes-claude smoke test this spawns TWO sessions: turn 1 starts a fresh
conversation, turn 2 resumes it with ``--conversation <id>`` and must see the
first turn's history. Hits the live CLI. Run:

    .venv/Scripts/python tests/scripts/smoke_session.py
"""
import asyncio
import sys
import tempfile
from pathlib import Path

from app.agy_session import STREAM_CLOSED, AgySession
from app.events import Error, Init, TextDelta, ToolStep, TurnDone

MODEL = "Gemini 3.5 Flash (Medium)"
WORKDIR = Path(tempfile.gettempdir()) / "agi_smoke_ws"


async def run_turn(prompt: str, conversation_id: str | None) -> tuple[str, TurnDone | None]:
    print(f"\n>>> SEND: {prompt!r} (conv={conversation_id})", flush=True)
    sess = AgySession(
        agy_bin="agy",
        model=MODEL,
        workdir=WORKDIR,
        prompt=prompt,
        conversation_id=conversation_id,
    )
    await sess.start()
    chunks: list[str] = []
    try:
        while True:
            ev = await sess.next_event(timeout=120)
            if ev is None:
                print("  [timeout]", flush=True)
                print("  diagnostics:", sess.diagnostics(), flush=True)
                return "".join(chunks), None
            if ev is STREAM_CLOSED:
                print("  [stream closed without result]", flush=True)
                print("  diagnostics:", sess.diagnostics(), flush=True)
                return "".join(chunks), None
            if isinstance(ev, Init):
                print(f"  init conv={ev.conversation_id} cwd={ev.cwd}", flush=True)
            elif isinstance(ev, TextDelta):
                chunks.append(ev.text)
            elif isinstance(ev, ToolStep):
                print(f"  tool step: {ev.tool_name} ({ev.state})", flush=True)
            elif isinstance(ev, TurnDone):
                print(f"  TURN DONE conv={ev.conversation_id} turns={ev.num_turns} "
                      f"usage_in={ev.usage.get('input_tokens')} out={ev.usage.get('output_tokens')}",
                      flush=True)
                # agy may deliver the final text only on the result line.
                text = "".join(chunks) or (ev.response_text or "")
                return text, ev
            elif isinstance(ev, Error):
                print(f"  ERROR: {ev.message}", flush=True)
                return "".join(chunks), None
    finally:
        await sess.aclose()


async def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    ok = True

    t1, d1 = await run_turn(
        "Remember the codeword ZEBRA. Reply with exactly the word PONG and nothing else.", None
    )
    print(f"  collected text: {t1!r}", flush=True)
    ok &= "PONG" in t1.upper() and d1 is not None

    conv_id = d1.conversation_id if d1 else None
    if not conv_id:
        print("  FAIL: no conversation_id from turn 1 — cannot test resume", flush=True)
        ok = False
    else:
        t2, d2 = await run_turn(
            "What was the codeword I asked you to remember? Reply with just that word.", conv_id
        )
        print(f"  collected text: {t2!r}", flush=True)
        ok &= "ZEBRA" in t2.upper() and d2 is not None
        if d2 and d2.conversation_id and d2.conversation_id != conv_id:
            print(f"  FAIL: resumed turn came back on a different conversation "
                  f"({d2.conversation_id} != {conv_id})", flush=True)
            ok = False

    print("\nSMOKE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
