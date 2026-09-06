"""Audit regressions for canonical discovery, exclusive cache reuse and bounded filtering."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from app.config import _FAMILY_VARIANTS, resolve_model
from app.convcache import ConversationCache, history_key
from app.main import create_app
from app.model_catalog import ModelCatalog, ModelEntry
from app.openai_models import ChatMessage
from app.textfilter import TableFlattener


def msg(role, content, **kw):
    return ChatMessage(role=role, content=content, **kw)


def history():
    return [msg('user', 'q'), msg('assistant', 'a'), msg('user', 'next')]


def test_structured_fingerprints():
    def image(value):
        return [msg('user', [{'type': 'text', 'text': 'image'},
                             {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + value}}])]
    assert history_key(image('AAAA')) != history_key(image('BBBB'))
    assert history_key([msg('user', 'q', name='one')]) != history_key([msg('user', 'q', name='two')])
    assert history_key([msg('tool', 'a', tool_call_id='one')]) != history_key([msg('tool', 'a', tool_call_id='two')])
    def call(arg):
        return [msg('assistant', None, tool_calls=[{'id': 'id', 'function': {'name': 'f', 'arguments': arg}}])]
    assert history_key(call('{}')) != history_key(call('{"x":1}'))
    assert history_key(history(), 'one') != history_key(history(), 'two')


def test_namespace_and_atomic_thread_claim():
    cache = ConversationCache()
    cache.store(history()[:1], 'a', 'cid', namespace='one')
    assert cache.lookup(history(), namespace='two') is None
    barrier = Barrier(8)
    def claim(_):
        barrier.wait()
        return cache.lookup(history(), namespace='one')
    with ThreadPoolExecutor(max_workers=8) as pool:
        hits = list(pool.map(claim, range(8)))
    assert sum(h is not None for h in hits) == 1
    assert len(cache) == 0


def test_store_invalidates_prefixes_across_namespaces():
    cache = ConversationCache()
    cache.store(history()[:1], 'a', 'cid', 'one')
    cache.store(history(), 'new answer', 'cid', 'two')
    assert cache.lookup(history(), 'one') is None
    assert cache.lookup(history() + [msg('assistant', 'new answer'), msg('user', 'branch')], 'two')


def test_claim_invalidates_all_entries_for_same_conversation():
    cache = ConversationCache()
    cache.store(history()[:1], 'a', 'cid')
    # Simulate legacy duplicate prefixes retained before the atomic-claim fix.
    cache._entries['old-key'] = next(iter(cache._entries.values()))
    assert cache.lookup(history())
    assert not len(cache)


def test_ambiguous_continuations_miss_without_claiming():
    cache = ConversationCache()
    cache.store(history()[:1], 'a', 'cid')
    multimodal = history()[:-1] + [msg('user', [
        {'type': 'text', 'text': 'next'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}])]
    assert cache.lookup(multimodal) is None
    assert cache.lookup(history())
    remote = [msg('user', [{'type': 'image_url', 'image_url': {'url': 'https://example.test/image'}}])]
    cache.store(remote, 'a', 'cid')
    assert len(cache) == 0


@pytest.mark.parametrize('live', [False, True])
def test_discovery_and_aliases_share_canonical_variants(live):
    app = create_app()
    entries = [ModelEntry(f'{family}-{eff}' if eff else family, display)
               for family, variants in _FAMILY_VARIANTS.items() for eff, display in variants.items()]
    if live:
        cat = ModelCatalog('unused')
        cat._entries = entries
        cat._loaded_at = 1
        app.state.model_catalog = cat
    client = TestClient(app)
    ids = [x['id'] for x in client.get('/v1/models').json()['data']]
    assert len(ids) == len(set(ids)) == 14
    assert ids == [e.slug for e in entries]
    assert [m['name'] for m in client.get('/api/tags').json()['models']] == ids
    assert [m['id'] for m in client.get('/api/v1/models').json()['data']] == ids
    aliases = ids + [e.display for e in entries] + list(_FAMILY_VARIANTS) + ['flash', 'sonnet', 'opus']
    for alias in aliases:
        assert client.get('/v1/models/' + alias).status_code == 200, alias
        assert client.post('/api/show', json={'model': alias}).status_code == 200, alias
        assert resolve_model(alias, app.state.settings) in [e.display for e in entries]
    app.state.settings = app.state.settings.model_copy(update={'strict_model': False})
    assert client.get('/v1/models/unknown-model').status_code == 404
    assert client.post('/api/show', json={'name': 'unknown-model'}).status_code == 404


@pytest.mark.parametrize('step', [1, 17, 300000])
@pytest.mark.parametrize('source', [
    'x' * 256000,
    ' ' * 100 + '| h |\n|---|\n',
    '| h |\n|---|\n' + '| row |\n' * 100,
    '| h |\n|---|\n|' + 'x' * 100,
])
def test_overflow_raw_and_bounded(source, step):
    f = TableFlattener(max_pending_chars=32)
    out = []
    for start in range(0, len(source), step):
        out.append(f.feed(source[start:start + step]))
        assert f._line_size + f._table_size <= 32
    out.append(f.flush())
    assert ''.join(out) == source


def test_fragmented_fence_and_formatting_match_whole_feed():
    source = '  ```python\n| h |\n|---|\n```\n| A | B |\n|---|---|\n| x | y |\n'
    f = TableFlattener()
    expected = f.feed(source) + f.flush()
    g = TableFlattener()
    assert ''.join(g.feed(c) for c in source) + g.flush() == expected
    assert expected.startswith('  ```python\n| h |\n|---|\n```\n')


async def test_catalog_freshness_status(fake_agy_bin):
    cat = ModelCatalog(fake_agy_bin)
    assert cat.status == 'fallback' and cat.age_s is None
    assert await cat.refresh()
    assert cat.status == 'fresh' and cat.age_s >= 0
    success = cat.last_success_at
    cat.agy_bin = '/nonexistent/audit-agy'
    assert not await cat.refresh()
    assert cat.status == 'stale' and cat.last_error == 'binary_not_found'
    assert cat.last_success_at == success and cat.last_attempt_at >= success


def test_ragged_table_does_not_expand_without_bound():
    source = '| ' + 'x' * 1000 + ' |\n|---|\n|' + 'a|' * 1000 + '\n' + '| a |\n' * 1000
    f = TableFlattener()
    assert f.feed(source) + f.flush() == source
