from __future__ import annotations

import json

from app.thinking import ThinkingReader, TurnThinking


def transcript(tmp_path, conv="c1"):
    p = tmp_path / conv / ".system_generated" / "logs" / "transcript_full.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def entry(thinking=None, content="answer", typ="PLANNER_RESPONSE") -> str:
    v = {"step_index": 2, "type": typ, "status": "DONE", "content": content}
    if thinking is not None:
        v["thinking"] = thinking
    return json.dumps(v) + "\n"


def test_missing_file_yields_nothing(tmp_path):
    r = ThinkingReader(tmp_path, "nope", skip_existing=False)
    assert r.poll() == []


def test_thinking_blocks_discovered_incrementally(tmp_path):
    tf = transcript(tmp_path)
    r = ThinkingReader(tmp_path, "c1", skip_existing=False)
    assert r.poll() == []

    with tf.open("a", encoding="utf-8") as f:
        f.write(entry(thinking="let me think about A"))
    assert r.poll() == ["let me think about A"]
    assert r.poll() == []  # nothing new

    with tf.open("a", encoding="utf-8") as f:
        f.write(entry(thinking="now about B"))
    assert r.poll() == ["now about B"]


def test_skip_existing_only_surfaces_new(tmp_path):
    tf = transcript(tmp_path)
    tf.write_text(entry(thinking="old turn thinking"), encoding="utf-8")
    r = ThinkingReader(tmp_path, "c1", skip_existing=True)
    assert r.poll() == []
    with tf.open("a", encoding="utf-8") as f:
        f.write(entry(thinking="fresh thinking"))
    assert r.poll() == ["fresh thinking"]


def test_partial_line_left_for_later(tmp_path):
    tf = transcript(tmp_path)
    r = ThinkingReader(tmp_path, "c1", skip_existing=False)
    full = entry(thinking="complete thought")
    tf.write_text(full[: len(full) // 2], encoding="utf-8")  # no trailing newline
    assert r.poll() == []
    with tf.open("a", encoding="utf-8") as f:
        f.write(full[len(full) // 2:])
    assert r.poll() == ["complete thought"]


def test_entries_without_thinking_ignored(tmp_path):
    tf = transcript(tmp_path)
    tf.write_text(
        entry() + entry(thinking="") + "not json at all\n" + entry(thinking="real"),
        encoding="utf-8",
    )
    r = ThinkingReader(tmp_path, "c1", skip_existing=False)
    assert r.poll() == ["real"]


def test_rewritten_file_does_not_duplicate(tmp_path):
    tf = transcript(tmp_path)
    r = ThinkingReader(tmp_path, "c1", skip_existing=False)
    tf.write_text(entry(thinking="thought X") + entry(thinking="thought Y"), encoding="utf-8")
    assert r.poll() == ["thought X", "thought Y"]
    # File rewritten SHORTER (e.g. compaction): re-read from the start, but the
    # seen-set keeps already-emitted blocks from repeating.
    tf.write_text(entry(thinking="thought Y"), encoding="utf-8")
    assert r.poll() == []
    with tf.open("a", encoding="utf-8") as f:
        f.write(entry(thinking="thought Z"))
    assert r.poll() == ["thought Z"]


def test_turn_thinking_binds_on_init(tmp_path):
    tf = transcript(tmp_path, conv="conv-9")
    tf.write_text(entry(thinking="early"), encoding="utf-8")
    t = TurnThinking(tmp_path)  # fresh conversation: id unknown yet
    assert t.poll() == []
    t.on_init("conv-9")
    assert t.poll() == ["early"]  # fresh conv reads from the start


def test_turn_thinking_resume_skips_history(tmp_path):
    tf = transcript(tmp_path, conv="conv-9")
    tf.write_text(entry(thinking="previous turn"), encoding="utf-8")
    t = TurnThinking(tmp_path, conversation_id="conv-9")  # resume
    assert t.poll() == []
    with tf.open("a", encoding="utf-8") as f:
        f.write(entry(thinking="this turn"))
    assert t.poll() == ["this turn"]
