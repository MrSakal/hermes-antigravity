from __future__ import annotations

import asyncio
import base64
import os
import time

import httpx
import pytest

import app.main as main
import app.routes.chat as chat
from app.config import Settings
from app.errors import OpenAIError
from app.openai_models import ChatCompletionRequest
from app.resources import AttachmentStore, Metrics, validate_request


def request(**kwargs):
    return ChatCompletionRequest(messages=[{'role':'user','content':'hi'}], **kwargs)


def image_request(raw=b'fixture'):
    return ChatCompletionRequest(messages=[{'role':'user','content':[
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}}
    ]}])


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, default_workdir=tmp_path/'ws',
                    mcp_config_path=tmp_path/'mcp.json', agent_plugin_path=tmp_path/'plugin',
                    agy_brain_path=tmp_path/'brain')


@pytest.fixture
def application(settings, monkeypatch):
    monkeypatch.setattr(main, 'get_settings', lambda:settings)
    monkeypatch.setattr(chat, 'get_settings', lambda:settings)
    return main.create_app()


@pytest.mark.parametrize('choice', ['required', {'type':'function','function':{'name':'foo'}}])
def test_unsupported_tool_choice_is_explicit(settings, choice):
    with pytest.raises(OpenAIError) as error:
        validate_request(request(tool_choice=choice), settings)
    assert error.value.param == 'tool_choice'


def test_image_validation_before_writes(settings):
    settings.max_image_bytes = 2
    with pytest.raises(OpenAIError):
        validate_request(image_request(b'12345'), settings)
    assert not settings.default_workdir.exists()


def test_total_image_and_count_limits(settings):
    req = image_request(b'12345')
    req.messages *= 2
    settings.max_total_image_bytes = 8
    with pytest.raises(OpenAIError):
        validate_request(req, settings)
    settings.max_total_image_bytes = 100
    settings.max_images = 1
    with pytest.raises(OpenAIError):
        validate_request(req, settings)


def test_schema_and_tool_count_limits(settings):
    tool = {'type':'function','function':{'name':'foo','description':'x'*100}}
    settings.max_tool_schema_bytes = 30
    with pytest.raises(OpenAIError):
        validate_request(request(tools=[tool]), settings)
    settings.max_tool_schema_bytes = 10000
    settings.max_tools = 0
    with pytest.raises(OpenAIError):
        validate_request(request(tools=[tool]), settings)


def test_gc_preserves_pinned_files_and_foreign_files(settings):
    store = AttachmentStore(settings)
    name = 'img-'+'a'*16+'.png'
    release = store.pin_and_write(settings.default_workdir, {name:b'fixture'})
    target = settings.default_workdir/'attachments'/name
    foreign = target.with_name('user-image.png')
    foreign.write_bytes(b'user data')
    old = time.time()-settings.attachment_ttl_s-10
    os.utime(target, (old,old))
    store.gc()
    assert target.exists()
    release(); release()
    store.gc()
    assert not target.exists()
    assert foreign.read_bytes() == b'user data'
    assert not store.pins


def test_quota_does_not_delete_active_files(settings):
    settings.attachment_max_bytes = 6
    store = AttachmentStore(settings)
    release = store.pin_and_write(settings.default_workdir, {'img-'+'a'*16+'.png':b'12345'})
    with pytest.raises(OpenAIError) as error:
        store.pin_and_write(settings.default_workdir, {'img-'+'b'*16+'.png':b'12345'})
    assert error.value.status_code == 507
    assert store.pins[settings.default_workdir.resolve()] == 1
    release()


def test_symlink_attachment_directory_rejected(settings, tmp_path):
    target = tmp_path/'foreign'
    target.mkdir()
    settings.default_workdir.mkdir()
    try:
        (settings.default_workdir/'attachments').symlink_to(target, target_is_directory=True)
    except OSError as exc:  # Windows without Developer Mode / SeCreateSymbolicLink
        pytest.skip(f'cannot create symlinks here: {exc}')
    store = AttachmentStore(settings)
    with pytest.raises(OpenAIError):
        store.pin_and_write(settings.default_workdir, {'img-'+'a'*16+'.png':b'123'})
    assert not list(target.iterdir())


async def test_body_limit_before_parsing(application, settings):
    settings.max_request_bytes = 64
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://audit') as client:
        result = await client.post('/v1/chat/completions', content=b'x'*65)
    assert result.status_code == 413
    assert application.state.metrics.http_active == 0


async def test_chunked_body_limit(application, settings):
    settings.max_request_bytes = 64
    async def chunks():
        yield b'x'*40
        yield b'y'*40
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://audit') as client:
        result = await client.post('/v1/chat/completions', content=chunks())
    assert result.status_code == 413


async def test_mcp_remote_rejected_even_without_api_key(application):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application, client=('192.0.2.1',1234)), base_url='http://audit') as client:
        result = await client.post('/mcp/fixture', json={})
    assert result.status_code == 403


async def test_non_ascii_bad_auth_returns_401(settings, monkeypatch):
    settings.api_key = 'fixture-key'
    monkeypatch.setattr(main, 'get_settings', lambda:settings)
    application = main.create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://audit') as client:
        result = await client.get('/v1/models', headers={b'authorization':b'Bearer \xff'})
    assert result.status_code == 401


async def test_resource_cleanup_and_metrics_after_plain_turn(application, fake_agy):
    fake_agy('simple')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://audit') as client:
        result = await client.post('/v1/chat/completions', json=request(max_tokens=20).model_dump())
        metrics = (await client.get('/metrics')).json()
    assert result.status_code == 200, result.text
    assert result.headers['x-agi-ignored-parameters'] == 'max_tokens'
    assert metrics['processes_running'] == 0
    assert metrics['attachment_workspaces_pinned'] == 0
    assert metrics['http_active'] == 0
    assert metrics['ttft_s_p50'] is not None


async def test_none_uses_plain_route(application, fake_agy):
    fake_agy('simple')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://audit') as client:
        result = await client.post('/v1/chat/completions', json=request(tools=[{'function':{'name':'foo'}}], tool_choice='none').model_dump())
    assert result.status_code == 200, result.text
    # Plain turn: parked for reuse WITHOUT an MCP server / bridge registration.
    convs = list(application.state.conv_manager._conversations.values())
    assert [(c.state, c.slot) for c in convs] == [('idle', None)]
    assert not application.state.mcp._registry
    await application.state.conv_manager.close_all()


async def test_readiness_is_not_fabricated(application):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://audit') as client:
        live = await client.get('/healthz')
        ready = await client.get('/readyz')
    assert live.status_code == 200
    assert ready.status_code == 503  # no lifespan/catalog load in this fixture
    assert ready.json()['inference_probe'] is False


async def test_invalid_tool_schema_is_http_400_without_side_effects(application):
    tool = {'function': {'name': 'bad', 'parameters': {'type': 'NOT_A_JSON_TYPE'}}}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application, raise_app_exceptions=False), base_url='http://audit') as client:
        result = await client.post('/v1/chat/completions', json=request(tools=[tool]).model_dump())
    assert result.status_code == 400, result.text
    assert result.json()['error']['param'] == 'tools'
    assert application.state.metrics.http_active == 0
    assert not application.state.mcp._registry
    assert not application.state.settings.default_workdir.exists()
