"""Parser tests against real line shapes captured from agy 1.1.4."""

from __future__ import annotations

import json

from app.events import (
    Error,
    Init,
    TextDelta,
    ToolStep,
    TurnDone,
    parse_line,
)

INIT_LINE = json.dumps({
    "event": "init",
    "conversation_id": "ec365cf0-a62d-4914-87b2-f72602eb7bc6",
    "init": {"cwd": "C:\\Users\\x\\Documents",
             "tools": ["ask_permission", "call_mcp_tool", "list_dir"],
             "permission_mode": "always-proceed"},
})

TEXT_ACTIVE = json.dumps({
    "event": "step_update",
    "step_update": {"conversation_id": "c", "step_index": 7, "state": "ACTIVE",
                    "step_type": "agent_response", "text_delta": "Here are the files"},
})

TEXT_DONE = json.dumps({
    "event": "step_update",
    "step_update": {"conversation_id": "c", "step_index": 2, "state": "DONE",
                    "step_type": "agent_response", "text_delta": "PONG\n",
                    "duration_seconds": 2.28,
                    "usage": {"input_tokens": 18618, "output_tokens": 316,
                              "thinking_tokens": 314, "total_tokens": 18934}},
})

TOOL_ACTIVE = json.dumps({
    "event": "step_update",
    "step_update": {"conversation_id": "c", "step_index": 8, "state": "ACTIVE",
                    "step_type": "tool", "tool_name": "call_mcp_tool",
                    "tool_info": {"name": "call_mcp_tool",
                                  "parameters": {"Arguments": {"text": "hi"},
                                                 "ServerName": "hermes",
                                                 "ToolName": "get_weather"}}},
})

TOOL_DONE = json.dumps({
    "event": "step_update",
    "step_update": {"conversation_id": "c", "step_index": 8, "state": "DONE",
                    "step_type": "tool", "tool_name": "call_mcp_tool",
                    "duration_seconds": 90.02,
                    "tool_info": {"name": "call_mcp_tool",
                                  "parameters": {"Arguments": {"text": "hi"},
                                                 "ServerName": "sequential-thinking",
                                                 "ToolName": "sequentialthinking"},
                                  "output": "{\"ok\": true}"}},
})

RESULT_OK = json.dumps({
    "event": "result",
    "result": {"conversation_id": "c", "status": "SUCCESS", "response": "PONG\n",
               "duration_seconds": 4.16, "num_turns": 1,
               "usage": {"input_tokens": 18717, "output_tokens": 319,
                         "thinking_tokens": 314, "total_tokens": 19036}},
})

RESULT_ERR = json.dumps({
    "event": "result",
    "result": {"conversation_id": "", "status": "ERROR", "response": "",
               "error": "invalid --model \"flash\": model flash is not recognized",
               "duration_seconds": 0, "num_turns": 0,
               "usage": {"input_tokens": 0, "output_tokens": 0,
                         "thinking_tokens": 0, "total_tokens": 0}},
})


def test_init():
    ev = parse_line(INIT_LINE)
    assert isinstance(ev, Init)
    assert ev.conversation_id == "ec365cf0-a62d-4914-87b2-f72602eb7bc6"
    assert "call_mcp_tool" in ev.tools
    assert ev.permission_mode == "always-proceed"


def test_text_deltas():
    ev = parse_line(TEXT_ACTIVE)
    assert isinstance(ev, TextDelta) and ev.text == "Here are the files"
    ev = parse_line(TEXT_DONE)
    assert isinstance(ev, TextDelta) and ev.text == "PONG\n"


def test_tool_steps_and_mcp_helpers():
    ev = parse_line(TOOL_ACTIVE)
    assert isinstance(ev, ToolStep)
    assert ev.state == "ACTIVE" and ev.is_mcp and ev.matches_server("hermes")
    assert ev.hermes_function_name == "get_weather"
    assert ev.hermes_arguments == {"text": "hi"}

    ev = parse_line(TOOL_DONE)
    assert isinstance(ev, ToolStep)
    assert ev.state == "DONE" and ev.is_mcp and not ev.matches_server("hermes")
    assert ev.output == "{\"ok\": true}"


def test_result_success():
    ev = parse_line(RESULT_OK)
    assert isinstance(ev, TurnDone)
    assert ev.response_text == "PONG\n"
    assert ev.usage["thinking_tokens"] == 314
    assert ev.conversation_id == "c"


def test_result_error():
    ev = parse_line(RESULT_ERR)
    assert isinstance(ev, Error)
    assert "not recognized" in ev.message


def test_defensive_parsing():
    assert parse_line("") is None
    assert parse_line("jetski: no output produced") is None
    assert parse_line("[1,2,3]") is None
    assert parse_line(json.dumps({"event": "wat"})) is None
    assert parse_line(json.dumps({"event": "step_update"})) is None
    assert parse_line(json.dumps({"event": "step_update",
                                  "step_update": {"step_type": "checkpoint"}})) is None
    assert parse_line(json.dumps({"event": "result", "result": "oops"})) is None
