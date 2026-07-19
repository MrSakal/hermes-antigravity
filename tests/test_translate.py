from __future__ import annotations

from app.events import TurnDone
from app.openai_models import ChatMessage
from app.translate import (
    ATTACHMENTS_DIR,
    build_prompt,
    fold_conversation,
    materialize_messages,
    message_text,
    message_text_with_images,
    split_system,
    usage_from_turn,
)

# A valid 1x1 red PNG.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
    "/pLvAAAAAElFTkSuQmCC"
)
PNG_DATA_URL = "data:image/png;base64," + PNG_B64


def msg(role: str, content) -> ChatMessage:
    return ChatMessage(role=role, content=content)


def image_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


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


# ── image materialization ────────────────────────────────────────────────────


def test_image_written_and_referenced(tmp_path):
    m = msg("user", [{"type": "text", "text": "what color? "}, image_part(PNG_DATA_URL)])
    out = message_text_with_images(m, tmp_path)
    assert out.startswith("what color? ")
    assert f"[Attached image file: {ATTACHMENTS_DIR}/img-" in out
    files = list((tmp_path / ATTACHMENTS_DIR).glob("img-*.png"))
    assert len(files) == 1
    assert files[0].read_bytes().startswith(b"\x89PNG")


def test_image_deduplicated_by_content(tmp_path):
    m = msg("user", [image_part(PNG_DATA_URL), image_part(PNG_DATA_URL)])
    out = message_text_with_images(m, tmp_path)
    assert out.count("[Attached image file:") == 2
    assert len(list((tmp_path / ATTACHMENTS_DIR).glob("img-*.png"))) == 1


def test_invalid_images_skipped(tmp_path):
    m = msg("user", [
        {"type": "text", "text": "t"},
        image_part("https://example.com/remote.png"),      # remote URL
        image_part("data:image/tiff;base64," + PNG_B64),   # unknown media type
        image_part("data:image/png;base64,!!!not-base64"),  # invalid payload
    ])
    assert message_text_with_images(m, tmp_path) == "t"
    assert not (tmp_path / ATTACHMENTS_DIR).exists()


def test_materialize_messages_preserves_originals(tmp_path):
    original = [
        msg("user", "plain"),
        msg("user", [{"type": "text", "text": "see: "}, image_part(PNG_DATA_URL)]),
    ]
    out = materialize_messages(original, tmp_path)
    assert out[0] is original[0]  # untouched message passes through by identity
    assert isinstance(out[1].content, str)
    assert "[Attached image file:" in out[1].content
    # The input list is never mutated (the conv cache keys on the original).
    assert isinstance(original[1].content, list)
