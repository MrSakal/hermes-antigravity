from __future__ import annotations

import pytest

from app.config import EFFORT_LADDER, Settings, clamp_effort, resolve_model
from app.openai_models import ChatCompletionRequest
from app.routes.chat import resolve_request_model

LMH = {"low": "X (Low)", "medium": "X (Medium)", "high": "X (High)"}


@pytest.mark.parametrize("effort,expected", [
    ("none", "low"), ("minimal", "low"), ("low", "low"), ("medium", "medium"),
    ("high", "high"), ("xhigh", "high"), ("max", "high"), ("", None), ("turbo", None),
])
def test_clamp_three_variants(effort, expected):
    assert clamp_effort(effort, LMH) == expected


def test_clamp_sparse_variants():
    assert clamp_effort("none", {"medium": "M"}) == "medium"
    assert clamp_effort("max", {"medium": "M"}) == "medium"
    assert clamp_effort("medium", {"low": "L", "high": "H"}) == "low"  # tie → lower


@pytest.mark.parametrize("effort", EFFORT_LADDER)
def test_request_accepts_full_openai_vocabulary(effort):
    req = ChatCompletionRequest(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}],
                                reasoning_effort=effort)
    assert req.reasoning_effort == effort


@pytest.mark.parametrize("effort,display", [
    ("none", "Gemini 3.8 Flash (Low)"), ("minimal", "Gemini 3.8 Flash (Low)"),
    ("medium", "Gemini 3.8 Flash (Medium)"), ("xhigh", "Gemini 3.8 Flash (High)"),
    ("max", "Gemini 3.8 Flash (High)"), (None, "Gemini 3.8 Flash (Medium)"),
])
def test_family_plus_effort_resolves_variant(effort, display):
    req = ChatCompletionRequest(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}],
                                reasoning_effort=effort)
    assert resolve_request_model(req, Settings(), None) == display


def test_slug_suffix_still_wins_over_effort():
    assert resolve_model("gemini-3.8-flash-high", Settings(), effort="none") == "Gemini 3.8 Flash (High)"


def test_single_effort_family_ignores_effort():
    fams = {"gpt-oss-120b": {"medium": "GPT-OSS 120B (Medium)"}}
    for eff in EFFORT_LADDER:
        assert resolve_model("gpt-oss-120b", Settings(), effort=eff, families=fams) == "GPT-OSS 120B (Medium)"
