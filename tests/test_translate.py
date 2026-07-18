from __future__ import annotations

from app.events import TurnDone
from app.openai_models import ChatMessage
from app.translate import (
    build_prompt,
    fold_conversation,
    message_text,
    split_system,
    usage_from_turn,
)


def msg(role: str, content) -> ChatMessage:
    return ChatMessage(role=role, content=content)


def test_message_text_forms():
    assert message_text(msg("user", "hi")) == "hi"
    assert message_text(msg("user", [{"type": "text", "text": "a"}, {"type": "image_url"}, "b"])) == "ab"
    assert message_text(msg("assistant", None)) == ""


def test_split_system_joins():
    convo, system = split_system([
        msg("system", "one"), msg("user", "q"), msg("system", "two"),
    ])
    assert system == "one\n\ntwo"
    assert [m.role for m in convo] == ["user"]


def test_fold_single_user_passthrough():
    assert fold_conversation([msg("user", "hello")]) == "hello"


def test_fold_transcript():
    out = fold_conversation([
        msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2"),
    ])
    assert out.startswith("Conversation so far:")
    assert "User: q1" in out and "Assistant: a1" in out
    assert out.rstrip().endswith("User: q2")


def test_build_prompt_injects_system():
    p = build_prompt([msg("user", "hi")], "be brief", inject_system=True)
    assert p.startswith("<system_instructions>\nbe brief\n</system_instructions>")
    assert p.rstrip().endswith("hi")
    p2 = build_prompt([msg("user", "hi")], "be brief", inject_system=False)
    assert p2 == "hi"
    p3 = build_prompt([msg("user", "hi")], None, inject_system=True)
    assert p3 == "hi"


def test_usage_mapping():
    t = TurnDone(usage={"input_tokens": 100, "output_tokens": 30,
                        "thinking_tokens": 25, "total_tokens": 130})
    u = usage_from_turn(t)
    assert u == {
        "prompt_tokens": 100,
        "completion_tokens": 30,
        "total_tokens": 130,
        "completion_tokens_details": {"reasoning_tokens": 25},
    }


def test_usage_mapping_no_thinking_no_total():
    t = TurnDone(usage={"input_tokens": 10, "output_tokens": 5})
    u = usage_from_turn(t)
    assert u == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_usage_never_fabricates():
    assert usage_from_turn(TurnDone(usage={})) == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }
