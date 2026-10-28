"""A stand-in for the `agy` CLI in stdin mode, used by unit tests.

Mirrors agy 1.4.8 (see app/agy_session.py): the ``init`` line is printed at
BOOT, before any stdin message; each stdin line
``{"event":"user","message":{"role":"user","content":...}}`` runs ONE turn
that ends with exactly one ``result`` line, and the process then waits for the
next line (persistent, same conversation). Content must be a string or a list
of text blocks — any other block errors the turn and exits rc=1, like the real
CLI. stdin EOF → exit 0.

FAKE_AGY_SCRIPT is a ';'-separated list of per-turn modes: turn N uses entry N
(the last one repeats):
  * "simple"     — text "Hello " + "world", result SUCCESS
  * "echo"       — text "echo: <turn content>", result SUCCESS
  * "tool"       — text, builtin tool step, text, result SUCCESS
  * "table"      — a markdown pipe table, result SUCCESS
  * "resultonly" — no text deltas; the text is only on the result line
  * "error"      — result status ERROR (process stays alive)
  * "die"        — one text delta, then exit 1 without a result
  * "noise"      — non-JSON noise + stderr text, exit 0 without a result
                   (no boot init line)
  * "echoargs"   — argv + the turn content as JSON on one line (no boot init)
  * "echoprompt" — result whose response is "len=<chars>;tail=<last 8 chars>"
  * "hang"       — text "Working " + a call_mcp_tool step, then block (as agy
                   does inside our MCP call_tool) until the file named by
                   FAKE_AGY_RELEASE exists or FAKE_AGY_HANG_S elapses, then
                   text "Done." and result SUCCESS
  * "models"     — (as the `agy models` subcommand) print a catalog
  * "bootfail"   — like agy logged out / given a bad --model: a stderr line
                   and exit 1 at BOOT, before any stdout line
Other env:
  * FAKE_AGY_CONV_ID — conversation id ("auto" = per pid).
  * FAKE_AGY_LOG — append one JSON line per process start (argv, agent file
    content as read at boot) and per received turn.
  * FAKE_AGY_PLUGIN — managed plugin dir, to read agents/<--agent>.md from.
  * FAKE_AGY_DELAY — seconds to sleep before each result line.
  * FAKE_AGY_BOOT_S — seconds to sleep before the boot init line.
  * FAKE_AGY_STATE_ROOT — a fake ``~/.gemini/antigravity-cli``: at boot the
    process leaves agy's per-conversation footprint there (conversations/
    <id>.db{,-wal}, presence/<id>.lock, brain/<id>/, annotations/<id>.pbtxt,
    implicit/<id>.pb, cache/*.json entries) and, without ``--log-file``, a
    ``log/cli-*.log`` — like agy 1.4.8. ``--log-file <path>`` is honored
    (also for ``models``). FAKE_AGY_CONV_ID=uuid mints a uuid4 per process.
"""
import json
import os
import sys
import time
import uuid as _uuid

ARGV = sys.argv[1:]
LOG_FILE = None
if "--log-file" in ARGV:
    _i = ARGV.index("--log-file")
    LOG_FILE = ARGV[_i + 1] if _i + 1 < len(ARGV) else None
    ARGV = ARGV[:_i] + ARGV[_i + 2:]
STATE_ROOT = os.environ.get("FAKE_AGY_STATE_ROOT")


def _win_parent(pid):
    """(parent pid, parent exe name) of ``pid`` via a Toolhelp32 snapshot."""
    import ctypes
    from ctypes import wintypes

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
    k = ctypes.WinDLL("kernel32")
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k.CreateToolhelp32Snapshot(2, 0)
    procs = {}
    try:
        e = PE()
        e.dwSize = ctypes.sizeof(PE)
        ok = k.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            procs[e.th32ProcessID] = (e.th32ParentProcessID, e.szExeFile)
            ok = k.Process32NextW(snap, ctypes.byref(e))
    finally:
        k.CloseHandle(wintypes.HANDLE(snap))
    ppid = procs.get(pid, (None,))[0]
    return (ppid, procs[ppid][1]) if ppid in procs else None


def crash_marker():
    """agy 1.4.8 creates crashes/crash_<pid>_<uuid>.log at every start and
    leaves it behind. <pid> is the process the server spawned: on Windows a
    venv python.exe is a launcher whose child runs this script."""
    if not STATE_ROOT:
        return
    pid = os.getppid() if os.name == "nt" and sys.prefix != sys.base_prefix else os.getpid()
    if os.name == "nt":  # the `agy models` wrapper is a .cmd: cmd.exe -> launcher -> us
        parent = _win_parent(pid)
        if parent and parent[1].lower() == "cmd.exe":
            pid = parent[0]
    os.makedirs(os.path.join(STATE_ROOT, "crashes"), exist_ok=True)
    open(os.path.join(STATE_ROOT, "crashes", f"crash_{pid}_{_uuid.uuid4()}.log"), "a").close()


def cli_log(msg):
    """agy's own log: --log-file, else <state root>/log/cli-<ts>.log."""
    path = LOG_FILE
    if path is None and STATE_ROOT:
        os.makedirs(os.path.join(STATE_ROOT, "log"), exist_ok=True)
        path = os.path.join(STATE_ROOT, "log", f"cli-{time.strftime('%Y%m%d_%H%M%S')}-{os.getpid()}.log")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _arg(flag):
    if flag in ARGV:
        i = ARGV.index(flag)
        if i + 1 < len(ARGV):
            return ARGV[i + 1]
    return None


MODES = [m.strip() for m in os.environ.get("FAKE_AGY_SCRIPT", "simple").split(";") if m.strip()]
DELAY = float(os.environ.get("FAKE_AGY_DELAY", "0"))
_conv_env = os.environ.get("FAKE_AGY_CONV_ID", "conv-fake-1")
CONV = (f"conv-{os.getpid()}" if _conv_env == "auto"
        else str(_uuid.uuid4()) if _conv_env == "uuid" else _conv_env)
LOG = os.environ.get("FAKE_AGY_LOG")


def leave_footprint():
    """What agy 1.4.8 leaves per conversation (see the module docstring)."""
    if not STATE_ROOT:
        return
    r = STATE_ROOT
    for sub in ("conversations", "presence", "annotations", "implicit", "cache"):
        os.makedirs(os.path.join(r, sub), exist_ok=True)
    os.makedirs(os.path.join(r, "brain", CONV), exist_ok=True)
    for path in (os.path.join(r, "conversations", CONV + ".db"),
                 os.path.join(r, "conversations", CONV + ".db-wal"),
                 os.path.join(r, "presence", CONV + ".lock"),
                 os.path.join(r, "annotations", CONV + ".pbtxt"),
                 os.path.join(r, "implicit", CONV + ".pb"),
                 os.path.join(r, "brain", CONV, "transcript.jsonl")):
        with open(path, "a", encoding="utf-8") as f:
            f.write("x")
    for name, update in (("last_conversations.json", lambda d: d.__setitem__(os.getcwd(), CONV)),
                         ("conversation_metadata.json",
                          lambda d: d.setdefault("conversations", {}).__setitem__(
                              CONV, {"is_internal": False}))):
        path = os.path.join(r, "cache", name)
        for _ in range(20):  # a concurrent fake may hold it (Windows)
            try:
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    data = {}
                update(data)
                tmp = f"{path}.{_uuid.uuid4()}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, path)
                break
            except OSError:
                time.sleep(0.05)


def log(obj):
    if not LOG:
        return
    # ppid too: a venv python.exe on Windows is a launcher that spawns the
    # real interpreter, so the pid the server sees may be our parent's.
    obj = {"pid": os.getpid(), "ppid": os.getppid(), **obj}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def init_line():
    emit({
        "event": "init",
        "conversation_id": CONV,
        "init": {"cwd": os.getcwd(), "tools": ["call_mcp_tool", "list_dir"],
                 "permission_mode": "always-proceed"},
    })


def text(step, s, state="ACTIVE"):
    emit({"event": "step_update", "step_update": {
        "conversation_id": CONV, "step_index": step, "state": state,
        "step_type": "agent_response", "text_delta": s}})


def tool_step(step, name, params, state, output=None):
    info = {"name": name, "parameters": params}
    if output is not None:
        info["output"] = output
    emit({"event": "step_update", "step_update": {
        "conversation_id": CONV, "step_index": step, "state": state,
        "step_type": "tool", "tool_name": name, "tool_info": info}})


def result(status="SUCCESS", response="", error=None):
    if DELAY:
        time.sleep(DELAY)
    r = {"conversation_id": CONV, "status": status, "response": response,
         "duration_seconds": 0.5, "num_turns": 1,
         "usage": {"input_tokens": 100, "output_tokens": 20,
                   "thinking_tokens": 5, "total_tokens": 120}}
    if error:
        r["error"] = error
    emit({"event": "result", "result": r})


def content_text(content):
    """Return the turn text, or None when a non-text block is present."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if not isinstance(block, dict) or block.get("type") != "text":
            return None
        parts.append(block.get("text", ""))
    return "".join(parts)


def read_agent_file():
    agent, plugin = _arg("--agent"), os.environ.get("FAKE_AGY_PLUGIN")
    if not agent or not plugin:
        return None
    try:
        with open(os.path.join(plugin, "agents", agent + ".md"), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def run_turn(mode, msg_text):
    if mode == "noise":
        sys.stdout.write("jetski: no output produced -- permission denied\n")
        sys.stdout.flush()
        sys.stderr.write("some stderr detail\n")
        sys.stderr.flush()
        sys.exit(0)
    if mode == "echoargs":
        emit({"argv": ARGV, "prompt": msg_text})
        result(response="argv echoed")
        return
    if mode == "echoprompt":
        result(response=f"len={len(msg_text)};tail={msg_text[-8:]}")
        return
    if mode == "error":
        result(status="ERROR", error='invalid --model "bogus": not recognized')
        return
    if mode == "die":
        text(2, "partial ")
        sys.exit(1)
    if mode == "echo":
        text(2, "echo: " + msg_text)
        result(response="echo: " + msg_text)
        return
    if mode == "table":
        t = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        text(2, t)
        result(response=t)
        return
    if mode == "resultonly":
        result(response="Only in result")
        return
    if mode == "hang":
        text(2, "Working ")
        tool_step(3, "call_mcp_tool",
                  {"ServerName": os.environ.get("FAKE_AGY_SERVER", "hermes-x"),
                   "ToolName": "get_weather", "Arguments": {}}, "ACTIVE")
        release = os.environ.get("FAKE_AGY_RELEASE")
        end = time.time() + float(os.environ.get("FAKE_AGY_HANG_S", "60"))
        while time.time() < end:
            if release and os.path.exists(release):
                break
            time.sleep(0.05)
        text(4, "Done.")
        result(response="Working Done.")
        return
    # simple / tool
    text(2, "Hello ")
    if mode == "tool":
        tool_step(3, "list_dir", {"DirectoryPath": "C:/x"}, "ACTIVE")
        tool_step(3, "list_dir", {"DirectoryPath": "C:/x"}, "DONE", output="a.txt\nb.txt")
    text(4, "world", state="DONE")
    result(response="Hello world")


def main():
    crash_marker()
    if ARGV and ARGV[0] == "models":
        cli_log("models")
        # `agy models` subcommand: banner + slug\tdisplay lines.
        sys.stdout.write("Fetching available models...\n")
        for slug, disp in [
            ("gemini-9.9-flash-high", "Gemini 9.9 Flash (High)"),
            ("gemini-9.9-flash-medium", "Gemini 9.9 Flash (Medium)"),
            ("gemini-9.9-flash-low", "Gemini 9.9 Flash (Low)"),
            ("claude-sonnet-4-6", "Claude Sonnet 4.6 (Thinking)"),
        ]:
            sys.stdout.write(f"{slug}\t{disp}\n")
        return
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:  # pragma: no cover
        pass
    if os.environ.get("FAKE_AGY_BOOT_S"):
        time.sleep(float(os.environ["FAKE_AGY_BOOT_S"]))
    log({"start": True, "argv": ARGV, "agent_file": read_agent_file(), "log_file": LOG_FILE})
    cli_log("boot")
    if MODES[0] == "bootfail":
        sys.stderr.write("Error: not signed in -- run `agy` to log in\n")
        sys.stderr.flush()
        sys.exit(1)
    # Like agy 1.4.8: the init line is printed at BOOT, before any stdin
    # message (transient agent-file deletion and the warm pool rely on it).
    if MODES[0] not in ("echoargs", "noise"):
        leave_footprint()
        init_line()
    turn = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            return  # stdin closed → exit 0
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("event") != "user":
            result(status="ERROR", error='stream input message is missing the "event" field')
            continue
        content = msg.get("message", {}).get("content")
        msg_text = content_text(content)
        log({"turn": turn, "content": content})
        if msg_text is None:
            result(status="ERROR", error="unsupported content block")
            sys.exit(1)
        mode = MODES[min(turn, len(MODES) - 1)]
        turn += 1
        run_turn(mode, msg_text)


if __name__ == "__main__":
    main()
