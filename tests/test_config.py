from __future__ import annotations

import pytest

from app.config import PromptTooLarge, Settings, UnknownModel, check_prompt_size, resolve_model


def make_settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_display_name_passthrough_case_insensitive():
    s = make_settings()
    assert resolve_model("Gemini 3.8 Flash (Low)", s) == "Gemini 3.8 Flash (Low)"
    assert resolve_model("gemini 3.8 flash (low)", s) == "Gemini 3.8 Flash (Low)"
    assert resolve_model("claude sonnet 5.5 (high)", s) == "Claude Sonnet 5.5 (High)"


def test_static_table_is_the_agy_148_lineup():
    from app.config import KNOWN_MODELS
    assert KNOWN_MODELS == [
        "Gemini 4 Argon (Medium)", "Gemini 4 Argon (High)", "Gemini 4 Argon (Low)",
        "Gemini 3.8 Flash (High)", "Gemini 3.8 Flash (Medium)", "Gemini 3.8 Flash (Low)",
        "Claude Opus 5.5 (Low)", "Claude Opus 5.5 (Medium)", "Claude Opus 5.5 (High)",
        "Claude Sonnet 5.5 (Low)", "Claude Sonnet 5.5 (Medium)", "Claude Sonnet 5.5 (High)",
        "GPT-OSS 120B (Medium)",
    ]


def test_slug_with_effort_param():
    s = make_settings()
    assert resolve_model("gemini-3.8-flash", s) == "Gemini 3.8 Flash (Medium)"
    assert resolve_model("gemini-3.8-flash", s, effort="high") == "Gemini 3.8 Flash (High)"
    assert resolve_model("gemini-4-argon", s, effort="low") == "Gemini 4 Argon (Low)"
    assert resolve_model("flash", s) == "Gemini 3.8 Flash (Medium)"
    assert resolve_model("argon", s) == "Gemini 4 Argon (Medium)"


def test_slug_effort_suffix_overrides():
    s = make_settings()
    assert resolve_model("gemini-4-argon-low", s, effort="high") == "Gemini 4 Argon (Low)"
    assert resolve_model("claude-opus-5-5-high", s) == "Claude Opus 5.5 (High)"


def test_retired_model_is_unknown():
    """Gemini 3.5 / 3.7 / 3.1 Pro and Claude 4.6 are gone on 1.4.8 — they must
    not silently resolve."""
    s = make_settings()
    for gone in ("gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.1-pro", "claude-opus-4-6-thinking"):
        with pytest.raises(UnknownModel):
            resolve_model(gone, s)


def test_family_aliases_follow_the_lineup():
    s = make_settings()
    assert resolve_model("sonnet", s, effort="low") == "Claude Sonnet 5.5 (Low)"
    assert resolve_model("opus", s) == "Claude Opus 5.5 (Medium)"
    assert resolve_model("claude", s, effort="high") == "Claude Opus 5.5 (High)"
    assert resolve_model("gpt-oss", s) == "GPT-OSS 120B (Medium)"
    assert resolve_model("gpt-oss-120b", s, effort="high") == "GPT-OSS 120B (Medium)"


def test_agy_native_slugs_accepted():
    """`agy models` slugs (column 1) resolve to their display names."""
    s = make_settings()
    assert resolve_model("claude-sonnet-5-5-medium", s) == "Claude Sonnet 5.5 (Medium)"
    assert resolve_model("gemini-3.8-flash-low", s) == "Gemini 3.8 Flash (Low)"
    assert resolve_model("gpt-oss-120b-medium", s) == "GPT-OSS 120B (Medium)"


def test_empty_falls_back_to_default():
    s = make_settings(default_model="Gemini 4 Argon (High)")
    assert resolve_model(None, s) == "Gemini 4 Argon (High)"
    assert resolve_model("", s) == "Gemini 4 Argon (High)"


def test_unknown_is_rejected_in_strict_mode():
    s = make_settings()
    assert s.strict_model is True
    with pytest.raises(UnknownModel):
        resolve_model("gpt-4o", s)


def test_unknown_falls_back_when_not_strict():
    s = make_settings(default_model="Gemini 4 Argon (High)", strict_model=False)
    assert resolve_model("gpt-4o", s) == "Gemini 4 Argon (High)"


def test_live_catalog_replaces_static_tables():
    s = make_settings()
    live_models = ["Gemini 9.9 Flash (Medium)", "Gemini 9.9 Flash (High)",
                   "Claude Opus 4.6 (Thinking)"]
    live_fams = {
        "gemini-9.9-flash": {"medium": live_models[0], "high": live_models[1]},
        "claude-opus-4-6-thinking": {"": live_models[2]},
    }
    kw = dict(known_models=live_models, families=live_fams)
    # Unknown statically, resolvable through the live catalog.
    assert resolve_model("gemini-9.9-flash", s, **kw) == "Gemini 9.9 Flash (Medium)"
    assert resolve_model("gemini-9.9-flash-high", s, **kw) == "Gemini 9.9 Flash (High)"
    assert resolve_model("gemini 9.9 flash (high)", s, **kw) == "Gemini 9.9 Flash (High)"
    # Aliases still map onto live families.
    assert resolve_model("opus", s, **kw) == "Claude Opus 4.6 (Thinking)"
    # Statically-known but absent from the live catalog → gone.
    with pytest.raises(UnknownModel):
        resolve_model("gemini-3.8-flash", s, **kw)


def test_prompt_size_is_bytes_not_chars():
    s = make_settings(max_prompt_bytes=10)
    assert check_prompt_size("abc", s) == 3
    with pytest.raises(PromptTooLarge) as ei:
        check_prompt_size("ő" * 6, s)  # 6 chars, 12 bytes
    assert ei.value.nbytes == 12 and ei.value.cap == 10


def test_tools_mode_validation():
    assert make_settings(plain_chat_tools="ReadOnly").plain_chat_tools == "readonly"
    with pytest.raises(ValueError):
        make_settings(plain_chat_tools="yolo")


def test_loopback_detection():
    assert make_settings(host="127.0.0.1").is_loopback_host()
    assert make_settings(host="localhost").is_loopback_host()
    assert not make_settings(host="0.0.0.0").is_loopback_host()


def test_workdir_guard(tmp_path):
    s = make_settings(default_workdir=tmp_path / "ws")
    assert s.resolve_workdir(None) == tmp_path / "ws"
    inside = s.resolve_workdir(str(tmp_path / "ws" / "sub"))
    assert inside.name == "sub"
    try:
        s.resolve_workdir(str(tmp_path / "elsewhere"))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for out-of-root workdir")


def test_aliases_follow_current_families_of_live_catalog():
    """agy 1.4.8 lineup: fixed alias targets (claude-sonnet-4-6, gemini-3.1-pro)
    are gone, so short aliases fall back to keyword matching; display family
    names and family slugs resolve too. Unknown ids stay strict."""
    from app.config import UnknownModel, resolve_model
    from app.model_catalog import ModelCatalog, parse_models_output

    cat = ModelCatalog("unused")
    cat._entries = parse_models_output(
        "gemini-4-argon-medium\tGemini 4 Argon (Medium)\n"
        "gemini-4-argon-low\tGemini 4 Argon (Low)\n"
        "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
        "claude-sonnet-5-5-medium\tClaude Sonnet 5.5 (Medium)\n"
        "claude-opus-5-5-high\tClaude Opus 5.5 (High)\n")
    kw = {"known_models": cat.display_names(), "families": cat.families()}
    s = Settings(_env_file=None)
    assert resolve_model("argon", s, "low", **kw) == "Gemini 4 Argon (Low)"
    assert resolve_model("gemini", s, **kw) == "Gemini 4 Argon (Medium)"
    assert resolve_model("Gemini 4 Argon", s, **kw) == "Gemini 4 Argon (Medium)"
    assert resolve_model("gemini-4-argon-low", s, **kw) == "Gemini 4 Argon (Low)"
    assert resolve_model("sonnet", s, **kw) == "Claude Sonnet 5.5 (Medium)"
    assert resolve_model("claude", s, **kw) == "Claude Opus 5.5 (High)"
    assert resolve_model("flash", s, **kw) == "Gemini 3.8 Flash (Low)"
    with pytest.raises(UnknownModel):
        resolve_model("pro", s, **kw)
    with pytest.raises(UnknownModel):
        resolve_model("no-such-model", s, **kw)
