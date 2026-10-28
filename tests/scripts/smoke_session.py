#!/usr/bin/env python3
"""Smoke test: drive the real `agy` CLI for two turns via AgySession.

One persistent process (stdin stream-json mode): turn 1 is the boot prompt,
turn 2 is a second stdin line on the SAME process and must see the first
turn's history — exactly what the server's IDLE reuse relies on. Hits the
live CLI. Run:

    .venv/Scripts/python tests/scripts/smoke_session.py
"""
import asyncio
import sys
import tempfile
from pathlib import Path

from app.agy_session import STREAM_CLOSED, AgySession
from app.events import Error, Init, TextDelta, ToolStep, TurnDone

MODEL = "Gemini 3.8 Flash (Medium)"
WORKDIR = Path(tempfile.gettempdir()) / "agi_smoke_ws"


async def read_turn(sess: AgySession) -> tuple[str, TurnDone | None]:
    chunks: list[str] = []
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


async def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    ok = True
    first = "Remember the codeword ZEBRA. Reply with exactly the word PONG and nothing else."
    print(f"\n>>> SEND: {first!r}", flush=True)
    sess = AgySession(agy_bin="agy", model=MODEL, workdir=WORKDIR, prompt=first)
    await sess.start()
    try:
        t1, d1 = await read_turn(sess)
        print(f"  collected text: {t1!r}", flush=True)
        ok &= "PONG" in t1.upper() and d1 is not None
        if not sess.running:
            print("  FAIL: process exited after turn 1", flush=True)
            ok = False
        else:
            second = "What was the codeword I asked you to remember? Reply with just that word."
            print(f"\n>>> SEND: {second!r} (same process)", flush=True)
            sess.send(second)
            t2, d2 = await read_turn(sess)
            print(f"  collected text: {t2!r}", flush=True)
            ok &= "ZEBRA" in t2.upper() and d2 is not None
            if d1 and d2 and d1.conversation_id and d2.conversation_id != d1.conversation_id:
                print(f"  FAIL: turn 2 came back on a different conversation "
                      f"({d2.conversation_id} != {d1.conversation_id})", flush=True)
                ok = False
    finally:
        await sess.aclose(graceful=True)

    print("\nSMOKE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
