from __future__ import annotations

from app.config import Settings, resolve_model


def make_settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_display_name_passthrough_case_insensitive():
    s = make_settings()
    assert resolve_model("Gemini 3.5 Flash (Low)", s) == "Gemini 3.5 Flash (Low)"
    assert resolve_model("gemini 3.5 flash (low)", s) == "Gemini 3.5 Flash (Low)"
    assert resolve_model("claude sonnet 4.6 (thinking)", s) == "Claude Sonnet 4.6 (Thinking)"


def test_slug_with_effort_param():
    s = make_settings()
    assert resolve_model("gemini-3.5-flash", s) == "Gemini 3.5 Flash (Medium)"
    assert resolve_model("gemini-3.5-flash", s, effort="high") == "Gemini 3.5 Flash (High)"
    assert resolve_model("gemini-3.5-flash", s, effort="low") == "Gemini 3.5 Flash (Low)"


def test_slug_effort_suffix_overrides():
    s = make_settings()
    assert resolve_model("gemini-3.5-flash-low", s, effort="high") == "Gemini 3.5 Flash (Low)"
    assert resolve_model("gemini-3.1-pro-high", s) == "Gemini 3.1 Pro (High)"


def test_pro_family_without_medium():
    s = make_settings()
    # No medium variant: effortless request prefers high, then low.
    assert resolve_model("gemini-3.1-pro", s) == "Gemini 3.1 Pro (High)"
    assert resolve_model("gemini-3.1-pro", s, effort="low") == "Gemini 3.1 Pro (Low)"


def test_single_variant_families_ignore_effort():
    s = make_settings()
    assert resolve_model("claude-sonnet-4.6", s, effort="low") == "Claude Sonnet 4.6 (Thinking)"
    assert resolve_model("opus", s) == "Claude Opus 4.6 (Thinking)"
    assert resolve_model("gpt-oss", s) == "GPT-OSS 120B (Medium)"


def test_unknown_falls_back_to_default():
    s = make_settings(default_model="Gemini 3.5 Flash (High)")
    assert resolve_model("gpt-4o", s) == "Gemini 3.5 Flash (High)"
    assert resolve_model(None, s) == "Gemini 3.5 Flash (High)"
    assert resolve_model("", s) == "Gemini 3.5 Flash (High)"


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
