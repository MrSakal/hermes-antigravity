"""Post-process Claude's streamed assistant text for chat clients (e.g. Discord).

Two streaming-safe transforms, combined in :class:`OutputFilter`:

1. **Segment joining** (:class:`SegmentJoiner`) — the ``claude`` CLI streams each
   assistant *text block* as a run of ``text_delta`` chunks with no trailing
   newline. When a built-in tool (Read/Edit/Bash/…) runs between two text blocks
   the bridge surfaces nothing for the gap, so the blocks arrive glued::

       "...only works for one-off CLI runs.Found it — ~/.hermes/SOUL.md is ..."

   Newlines appear to "randomly" disappear because the loss happens *only* at
   these tool-call boundaries. The joiner guarantees one blank line at the seam.

2. **Table flattening** (:class:`TableFlattener`) — GitHub-flavoured Markdown
   pipe tables don't render in Discord; they show as raw ``| a | b |`` noise.
   Whole table blocks are rewritten as a fixed-width ASCII grid inside a ```` ```
   ```` fence (Discord renders fences monospaced, so columns line up). Detection
   is fence-aware so ``|`` inside a code block is left untouched.

Both are fed text incrementally via ``feed`` and drained once at end-of-turn via
``flush``. They are total — they never raise on odd input.
"""

from __future__ import annotations

import re

# A separator-row cell: optional leading/trailing colon around one-or-more dashes.
_SEP_CELL = re.compile(r":?-+:?")


class SegmentJoiner:
    """Insert a blank line between assistant text segments split by a tool use.

    Fed the raw text of each ``TextDelta`` via :meth:`feed`; told about each
    tool-call boundary via :meth:`tool_boundary`. Returns text ready to emit
    (possibly newline-prefixed). Stateful and total — never raises.
    """

    def __init__(self) -> None:
        self._emitted = False
        self._pending_break = False
        self._trailing_newlines = 0

    def tool_boundary(self) -> None:
        """Record that a built-in tool ran between two text segments."""
        if self._emitted:
            self._pending_break = True

    def feed(self, text: str) -> str:
        """Return ``text`` to emit, prefixed with a separator when one is due."""
        if not text:
            return ""
        if self._pending_break:
            self._pending_break = False
            # Aim for exactly one blank line (two newlines) at the seam, counting
            # newlines the previous segment already left and ones this one leads
            # with, so we never stack 3+ blank lines.
            leading = len(text) - len(text.lstrip("\n"))
            need = 2 - self._trailing_newlines - leading
            if need > 0:
                text = "\n" * need + text
        self._emitted = True
        stripped = text.rstrip("\n")
        if stripped == "":
            self._trailing_newlines += len(text)
        else:
            self._trailing_newlines = len(text) - len(stripped)
        return text

    def flush(self) -> str:
        """Drain any buffered tail. The joiner buffers nothing; always ``""``."""
        return ""


# ── table flattening ─────────────────────────────────────────────────────────


def _split_cells(line: str) -> list[str]:
    """Split one GFM table row into trimmed cell texts (outer pipes optional)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_row(line: str) -> bool:
    cells = _split_cells(line)
    if not cells:
        return False
    return all(c and _SEP_CELL.fullmatch(c) for c in cells)


def _alignments(sep_line: str) -> list[str]:
    aligns = []
    for c in _split_cells(sep_line):
        left, right = c.startswith(":"), c.endswith(":")
        aligns.append("c" if left and right else "r" if right else "l")
    return aligns


def _looks_like_table(rows: list[str]) -> bool:
    """A header row plus a separator row is the minimum for a real GFM table."""
    return len(rows) >= 2 and _is_separator_row(rows[1])


def render_ascii_table(rows: list[str]) -> str:
    """Render GFM pipe-table ``rows`` as a fenced, column-aligned ASCII grid."""
    header = _split_cells(rows[0])
    aligns = _alignments(rows[1])
    body = [_split_cells(r) for r in rows[2:]]
    ncols = max([len(header), len(aligns)] + [len(r) for r in body])

    def pad(r: list[str]) -> list[str]:
        return r + [""] * (ncols - len(r))

    header = pad(header)
    body = [pad(r) for r in body]
    aligns = (aligns + ["l"] * ncols)[:ncols]

    widths = [len(header[i]) for i in range(ncols)]
    for r in body:
        for i in range(ncols):
            widths[i] = max(widths[i], len(r[i]))

    def cell(val: str, i: int) -> str:
        w, a = widths[i], aligns[i]
        return val.rjust(w) if a == "r" else val.center(w) if a == "c" else val.ljust(w)

    def row(r: list[str]) -> str:
        return " | ".join(cell(r[i], i) for i in range(ncols))

    sep = "-+-".join("-" * widths[i] for i in range(ncols))
    lines = [row(header), sep] + [row(r) for r in body]
    return "```\n" + "\n".join(lines) + "\n```\n"


class TableFlattener:
    """Rewrite Markdown pipe tables into fenced ASCII as text streams through.

    Normal prose streams through with at most a line's leading whitespace
    buffered, so token-by-token streaming is preserved. Only consecutive
    pipe-leading lines (table candidates) are held until the block ends, then
    converted (if they form a real table) or emitted verbatim (if not). Lines
    inside a ```` ``` ```` fence are never treated as tables.
    """

    def __init__(self, *, max_pending_chars: int = 65536) -> None:
        self.max_pending_chars = max(1, max_pending_chars)
        self._line: list[str] = []
        self._line_size = 0
        self._decided: bool | None = None
        self._table: list[str] = []
        self._table_size = 0
        self._in_fence = False
        self._prefix = ""
        self._raw = False

    def feed(self, text: str) -> str:
        if self._raw:
            return text
        out: list[str] = []
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if self._raw:
                out.append(part)
                if i < len(parts) - 1:
                    out.append("\n")
                continue
            self._chunk(part, out)
            if i < len(parts) - 1:
                if self._raw:
                    out.append("\n")
                else:
                    self._newline(out)
        return "".join(out)

    def _chunk(self, text: str, out: list[str]) -> None:
        if not text:
            return
        if self._decided is None:
            stripped = text.lstrip()
            if stripped:
                self._decided = stripped.startswith("|") and not self._in_fence
                if not self._decided:
                    out.append(self._flush_table())
                    out.extend(self._line)
                    self._line = []
                    self._line_size = 0
                self._prefix = stripped[:3]
        elif self._decided is False and len(self._prefix) < 3:
            self._prefix += text[:3 - len(self._prefix)]
        if self._decided is False:
            out.append(text)
            return
        if self._table_size + self._line_size + len(text) > self.max_pending_chars:
            # Once a pending block overflows, preserve all remaining bytes raw.
            # Never reinterpret a suffix of the same oversized table as a table.
            out.extend(self._table)
            out.extend(self._line)
            out.append(text)
            self._table = []
            self._line = []
            self._table_size = self._line_size = 0
            self._raw = True
            return
        self._line.append(text)
        self._line_size += len(text)

    def _newline(self, out: list[str]) -> None:
        if self._decided is True:
            if self._table_size + self._line_size + 1 > self.max_pending_chars:
                out.extend(self._table)
                out.extend(self._line)
                out.append("\n")
                self._table = []
                self._table_size = 0
                self._raw = True
            else:
                self._table.append("".join(self._line) + "\n")
                self._table_size += self._line_size + 1
        elif self._decided is False:
            out.append("\n")
        else:
            out.append(self._flush_table())
            out.extend(self._line)
            out.append("\n")
        if self._prefix.startswith("```"):
            self._in_fence = not self._in_fence
        self._line = []
        self._line_size = 0
        self._prefix = ""
        self._decided = None

    def _flush_table(self) -> str:
        rows, self._table = self._table, []
        self._table_size = 0
        if _looks_like_table(rows):
            # Ragged rows can amplify a small input into a huge padded grid.
            # Estimate its size in linear input work before allocating that grid.
            widths: list[int] = []
            for row in rows:
                for i, cell in enumerate(_split_cells(row)):
                    if i == len(widths):
                        widths.append(len(cell))
                    else:
                        widths[i] = max(widths[i], len(cell))
            projected = (sum(widths) + 3 * len(widths) + 1) * len(rows) + 8
            if projected <= self.max_pending_chars * 4:
                return render_ascii_table(rows)
        return "".join(rows)

    def flush(self) -> str:
        if self._raw:
            return ""
        if self._decided is True:
            self._table.append("".join(self._line))
            result = self._flush_table()
        elif self._decided is None:
            result = self._flush_table() + "".join(self._line)
        else:
            result = ""
        self._line = []
        self._line_size = 0
        self._prefix = ""
        self._decided = None
        return result


# ── combined filter ──────────────────────────────────────────────────────────


class OutputFilter:
    """Segment-join, then optionally table-flatten, the assistant text stream."""

    def __init__(self, *, flatten_tables: bool = True) -> None:
        self._joiner = SegmentJoiner()
        self._tables = TableFlattener() if flatten_tables else None

    def feed(self, text: str) -> str:
        joined = self._joiner.feed(text)
        if not joined:
            return ""
        return self._tables.feed(joined) if self._tables else joined

    def tool_boundary(self) -> None:
        self._joiner.tool_boundary()

    def flush(self) -> str:
        # The joiner buffers nothing; only the table flattener may hold a tail.
        return self._tables.flush() if self._tables else ""
