"""Deterministic CPU-only filter benchmark, optionally against a git baseline.

Run: .venv/bin/python scripts/bench_filter.py --baseline-ref 5cf6d78
No upstream calls. Payloads are deliberately generated performance fixtures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.textfilter import OutputFilter


def measure(factory, text, chunk_size, repeats):
    times = []
    for _ in range(repeats):
        filt = factory(flatten_tables=True)
        start = time.perf_counter()
        chunks = [text] if chunk_size == 0 else (text[i:i+chunk_size] for i in range(0, len(text), chunk_size))
        result = ''.join(filt.feed(chunk) for chunk in chunks) + filt.flush()
        times.append((time.perf_counter()-start)*1000)
        assert result == text, 'benchmark must preserve the fixture exactly'
    return round(statistics.median(times), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ref')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    factories = {'current': OutputFilter}
    if args.baseline_ref:
        source = subprocess.check_output(['git', 'show', args.baseline_ref+':app/textfilter.py'], cwd=ROOT)
        module = types.ModuleType('baseline_textfilter')
        exec(compile(source, 'git-baseline:textfilter.py', 'exec'), module.__dict__)
        factories['baseline'] = module.OutputFilter
    rows = []
    for size in (16_000, 64_000, 256_000):
        text = 'x'*size
        for chunk_size in (0, 1):
            row = {'characters': size, 'chunk_size': chunk_size or 'whole', 'median_ms': {}}
            for name, factory in factories.items():
                row['median_ms'][name] = measure(factory, text, chunk_size, args.repeats)
            rows.append(row)
    print(json.dumps({'repeats': args.repeats, 'baseline_ref': args.baseline_ref, 'results': rows}, indent=2))


if __name__ == '__main__':
    main()
