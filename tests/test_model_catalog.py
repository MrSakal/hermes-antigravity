from __future__ import annotations

import pytest

from app.model_catalog import ModelCatalog, ModelEntry, parse_models_output

SAMPLE = """Fetching available models...
gemini-3.8-flash-high\tGemini 3.8 Flash (High)
gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)
gemini-3.8-flash-low\tGemini 3.8 Flash (Low)
gemini-3.1-pro-high\tGemini 3.1 Pro (High)
gemini-3.1-pro-low\tGemini 3.1 Pro (Low)
claude-sonnet-4-6\tClaude Sonnet 4.6 (Thinking)
claude-opus-4-6-thinking\tClaude Opus 4.6 (Thinking)
gpt-oss-120b-medium\tGPT-OSS 120B (Medium)
"""


def test_parse_skips_banner_and_dedupes():
    entries = parse_models_output(SAMPLE + "gemini-3.8-flash-high\tdup\n")
    assert len(entries) == 8
    assert entries[0] == ModelEntry("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)")


def test_family_and_effort_split():
    e = ModelEntry("gemini-3.8-flash-medium", "Gemini 3.8 Flash (Medium)")
    assert e.family == "gemini-3.8-flash" and e.effort == "medium"
    s = ModelEntry("claude-sonnet-4-6", "Claude Sonnet 4.6 (Thinking)")
    assert s.family is None and s.effort is None
    # "gpt-oss-120b-medium" has an effort suffix → family gpt-oss-120b
    g = ModelEntry("gpt-oss-120b-medium", "GPT-OSS 120B (Medium)")
    assert g.family == "gpt-oss-120b" and g.effort == "medium"


def test_families_shape_matches_config_tables():
    cat = ModelCatalog("unused")
    cat._entries = parse_models_output(SAMPLE)
    fams = cat.families()
    assert fams["gemini-3.8-flash"] == {
        "high": "Gemini 3.8 Flash (High)",
        "medium": "Gemini 3.8 Flash (Medium)",
        "low": "Gemini 3.8 Flash (Low)",
    }
    assert fams["claude-sonnet-4-6"] == {"": "Claude Sonnet 4.6 (Thinking)"}


async def test_refresh_runs_agy_models(fake_agy_bin):
    cat = ModelCatalog(fake_agy_bin)
    assert not cat.loaded
    assert await cat.refresh() is True
    assert cat.loaded
    assert "gemini-9.9-flash-medium" in cat.slugs()
    assert "Gemini 9.9 Flash (Low)" in cat.display_names()


async def test_refresh_missing_binary_is_soft_failure():
    cat = ModelCatalog("definitely-not-a-real-binary-xyz")
    assert await cat.refresh() is False
    assert not cat.loaded
    assert cat.entries == []


async def test_refresh_timeout_is_soft_failure(tmp_path):
    slow = tmp_path / "slow"
    slow.write_text("#!/bin/sh\nsleep 5\n")
    slow.chmod(0o755)
    cat = ModelCatalog(str(slow), query_timeout_s=0.2)
    assert await cat.refresh() is False


@pytest.mark.parametrize("text", ["", "garbage\nno tabs here\n"])
def test_parse_empty(text):
    assert parse_models_output(text) == []
