"""Metrics, request validation and reference-safe attachment retention."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable

from jsonschema.validators import validator_for
from jsonschema.exceptions import SchemaError

from app.errors import OpenAIError
from app.translate import _IMAGE_EXT


class Metrics:
    def __init__(self):
        self.counters = Counter()
        self.http_active = 0
        self.duration = deque(maxlen=1024)
        self.ttft = deque(maxlen=1024)

    def snapshot(self):
        def percentile(values, p):
            ordered = sorted(values)
            return ordered[min(len(ordered)-1, int((len(ordered)-1)*p))] if ordered else None
        return {"counters": dict(self.counters), "http_active": self.http_active,
                "sample_window": 1024, "duration_s_p50": percentile(self.duration, .5),
                "duration_s_p95": percentile(self.duration, .95),
                "ttft_s_p50": percentile(self.ttft, .5), "ttft_s_p95": percentile(self.ttft, .95)}


def validate_request(req, settings) -> dict[str, bytes]:
    """Reject unsupported contracts and oversized schemas/images BEFORE writes."""
    if req.tool_choice not in (None, 'auto', 'none'):
        raise OpenAIError('only tool_choice auto/none is supported by this CLI',
                          status_code=400, param='tool_choice', code='unsupported_parameter')
    if req.effort and req.reasoning_effort and req.effort != req.reasoning_effort:
        raise OpenAIError('effort and reasoning_effort conflict', param='reasoning_effort')
    if req.tools:
        if len(req.tools) > settings.max_tools:
            raise OpenAIError('too many tools', status_code=400, param='tools', code='context_length_exceeded')
        try:
            schema_bytes = len(json.dumps([t.model_dump() for t in req.tools], ensure_ascii=False, allow_nan=False).encode())
        except (ValueError, RecursionError) as exc:
            raise OpenAIError('invalid tool schema JSON', param='tools') from exc
        if schema_bytes > settings.max_tool_schema_bytes:
            raise OpenAIError('tool schemas exceed byte limit', status_code=400, param='tools', code='context_length_exceeded')
        names = [t.function.name for t in req.tools]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise OpenAIError('tool names must be nonempty and unique', param='tools')
        try:
            for tool in req.tools:
                schema = tool.function.parameters
                validator_for(schema).check_schema(schema)
        except (SchemaError, ValueError, RecursionError) as exc:
            raise OpenAIError('invalid JSON Schema in tool parameters', param='tools') from exc
    images = {}
    count = total = 0
    for message in req.messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if not isinstance(part, dict) or part.get('type') != 'image_url':
                continue
            count += 1
            if count > settings.max_images:
                raise OpenAIError('too many images', param='messages', code='context_length_exceeded')
            if not settings.image_input:
                raise OpenAIError('image input is disabled', param='messages', code='unsupported_parameter')
            image = part.get('image_url')
            url = image.get('url', '') if isinstance(image, dict) else ''
            if not isinstance(url, str):
                raise OpenAIError('invalid image URL', param='messages')
            header, sep, data = url.partition(',')
            mime = header[5:].split(';')[0].lower() if header.startswith('data:') else ''
            ext = _IMAGE_EXT.get(mime)
            if not sep or ';base64' not in header.lower() or not ext or not data:
                raise OpenAIError('images must be supported base64 data URLs', param='messages')
            if len(data) > ((settings.max_image_bytes+2)//3)*4:
                raise OpenAIError('image exceeds byte limit', param='messages', code='context_length_exceeded')
            try:
                raw = base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise OpenAIError('invalid image base64', param='messages') from exc
            total += len(raw)
            if len(raw) > settings.max_image_bytes or total > settings.max_total_image_bytes:
                raise OpenAIError('images exceed decoded byte limit', param='messages', code='context_length_exceeded')
            images['img-'+hashlib.sha1(raw).hexdigest()[:16]+ext] = raw
    return images


def _touch_nofollow(target: Path) -> None:
    """Bump mtime (GC recency) without following a symlink planted at ``target``.

    Windows has no ``lutimes`` (``follow_symlinks=False`` raises
    NotImplementedError), so there the symlink check is explicit instead."""
    if os.utime in os.supports_follow_symlinks:
        os.utime(target, None, follow_symlinks=False)
    elif not target.is_symlink():
        os.utime(target, None)


class AttachmentStore:
    """Quota per workspace; active/suspended requests pin the whole workspace.

    Names are bridge-generated only; never follow symlinks or remove unrelated
    files. Pin/write/release/GC run synchronously on the single serving loop, so
    a GC tick cannot race an in-process writer. Shared multi-worker use unsupported.
    """
    def __init__(self, settings):
        self.ttl = settings.attachment_ttl_s
        self.cap = settings.attachment_max_bytes
        self.interval = settings.attachment_gc_interval_s
        self.pins = Counter()
        self.workdirs = {settings.default_workdir.resolve()}
        # Set by main: records a workdir in the housekeeping ledger.
        self.on_workdir: Callable[[Path], None] | None = None

    def add_workdirs(self, workdirs) -> None:
        """Workdirs earlier runs used (from the ledger): GC them too."""
        for w in workdirs:
            self.workdirs.add(Path(w).resolve())

    @staticmethod
    def _files(workdir):
        directory = workdir / 'attachments'
        if directory.is_symlink():
            raise OpenAIError('attachment directory may not be a symlink', status_code=400)
        if not directory.exists():
            return []
        return [p for p in directory.glob('img-*')
                if re.fullmatch(r'img-[0-9a-f]{16}\.(png|jpg|gif|webp)', p.name)
                and p.is_file() and not p.is_symlink()]

    def gc(self):
        for workdir in tuple(self.workdirs):
            if self.pins[workdir]:
                continue
            try:
                files = sorted(self._files(workdir), key=lambda p:p.stat().st_mtime)
                total = sum(p.stat().st_size for p in files)
                for path in files:
                    stat = path.stat()
                    if time.time()-stat.st_mtime > self.ttl or total > self.cap:
                        path.unlink(missing_ok=True)
                        total -= stat.st_size
            except (OSError, OpenAIError):
                continue  # inaccessible workspaces do not crash background GC

    def pin_and_write(self, workdir: Path, images: dict[str, bytes]) -> Callable[[], None]:
        workdir = workdir.resolve()
        self.workdirs.add(workdir)
        if self.on_workdir is not None:
            self.on_workdir(workdir)  # ledger: GC'd across restarts, purged
        self.gc()
        directory = workdir/'attachments'
        files = self._files(workdir)
        total = sum(p.stat().st_size for p in files)
        added = 0
        for name, data in images.items():
            target = directory/name
            if target.is_symlink():
                raise OpenAIError('attachment may not be a symlink', status_code=400)
            if not target.exists():
                added += len(data)
        if total+added > self.cap:
            raise OpenAIError('attachment storage quota exceeded', status_code=507, code='storage_quota_exceeded')
        self.pins[workdir] += 1
        released = False
        def release():
            nonlocal released
            if not released:
                released = True
                self.pins[workdir] -= 1
                if self.pins[workdir] == 0:
                    del self.pins[workdir]
        try:
            if images:
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            for name, data in images.items():
                target = directory/name
                if not target.exists():
                    fd, tmp = tempfile.mkstemp(prefix='.agi-', dir=directory)
                    try:
                        with os.fdopen(fd, 'wb') as out:
                            out.write(data)
                        os.replace(tmp, target)
                    finally:
                        Path(tmp).unlink(missing_ok=True)
                _touch_nofollow(target)
        except BaseException:
            release()
            raise
        return release

    async def gc_loop(self):
        while True:
            await asyncio.sleep(self.interval)
            self.gc()
