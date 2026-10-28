"""Audit regressions for canonical discovery, exclusive reuse claims and bounded filtering.

The history-prefix cache (``convcache``) these tests used to cover was
subsumed by the conversation manager's IDLE reuse index; its
invariants (structured fingerprints, namespacing, single-use atomic claims)
are asserted against :func:`app.conversation.history_key` and
``ConversationManager._claim_idle`` instead."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent_config import AgentPluginManager
from app.config import Settings, _FAMILY_VARIANTS, resolve_model
from app.conversation import Conversation, ConversationManager, history_key
from app.main import create_app
from app.mcp_bridge import ConversationBridge, McpBridge
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
    assert history_key('s', image('AAAA')) != history_key('s', image('BBBB'))
    assert history_key('s', [msg('user', 'q', name='one')]) != history_key('s', [msg('user', 'q', name='two')])
    assert history_key('s', [msg('tool', 'a', tool_call_id='one')]) != history_key('s', [msg('tool', 'a', tool_call_id='two')])

    def call(call_id, arg):
        return [msg('assistant', None, tool_calls=[{'id': call_id, 'function': {'name': 'f', 'arguments': arg}}])]
    # Tool calls match by id + name: clients re-serialize the argument JSON.
    assert history_key('s', call('id', '{}')) == history_key('s', call('id', '{ }'))
    assert history_key('s', call('id1', '{}')) != history_key('s', call('id2', '{}'))
    # The spawn signature namespaces every key.
    assert history_key('one', history()) != history_key('two', history())
    # Assistant text is compared whitespace-stripped; system messages live in
    # the signature (agent file / preamble hash), not in the key.
    assert history_key('s', [msg('user', 'q'), msg('assistant', ' a\n')]) == \
        history_key('s', [msg('system', 'sys'), msg('user', 'q'), msg('assistant', 'a')])


async def test_idle_claim_is_single_use_and_atomic(tmp_path):
    mgr = ConversationManager(McpBridge(), Settings(_env_file=None, default_workdir=tmp_path),
                              agents=AgentPluginManager(tmp_path / 'plugin'))
    key = history_key('sig', history()[:-1])

    def park(conv_id):
        conv = Conversation(conv_id=conv_id, slot=None, session=SimpleNamespace(running=True),
                            bridge=ConversationBridge(conv_id, []), model='m', state='idle',
                            reuse_key=key)
        mgr._conversations[conv_id] = conv
        mgr._idle[key] = conv_id
        return conv
    conv = park('c1')
    results = await asyncio.gather(*(mgr._claim_idle(key) for _ in range(8)))
    hits = [c for c in results if c is not None]
    assert hits == [conv] and conv.state == 'running'
    assert not mgr._idle
    # Another signature never sees it.
    park('c2')
    assert await mgr._claim_idle(history_key('other', history()[:-1])) is None


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
    # Default: one id per model (family slug); effort travels separately.
    assert ids == list(_FAMILY_VARIANTS)
    assert [m['name'] for m in client.get('/api/tags').json()['models']] == ids
    assert [m['id'] for m in client.get('/api/v1/models').json()['data']] == ids
    # Legacy superset on request: variant slugs, multi-variant families, display names.
    app.state.settings = app.state.settings.model_copy(update={'advertise_models': 'all'})
    full = [x['id'] for x in client.get('/v1/models').json()['data']]
    slugs = [e.slug for e in entries]
    families = [f for f, v in _FAMILY_VARIANTS.items() if '' not in v]
    assert len(full) == len(set(full))
    assert full == slugs + families + [e.display for e in entries]
    app.state.settings = app.state.settings.model_copy(update={'advertise_models': 'families'})
    # Every legacy id stays a valid input even though it is no longer listed.
    aliases = full + list(_FAMILY_VARIANTS) + ['flash', 'sonnet', 'opus']
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
], ids=['long-line', 'indented-table', 'long-table', 'ragged-row'])
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
