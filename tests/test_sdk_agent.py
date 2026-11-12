import asyncio
import importlib.metadata
import subprocess
import sys
from dataclasses import make_dataclass
from types import ModuleType, SimpleNamespace
from pathlib import Path
import runpy
import io
import urllib.request

import pytest

from nice_validation.sdk_agent import sdk_options, sdk_session

SDK_ENV={'ANTHROPIC_BASE_URL':'http://10.1.2.3:8000',
         'ANTHROPIC_API_KEY':'mock-key','ANTHROPIC_AUTH_TOKEN':'mock-key'}


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk=ModuleType('claude_agent_sdk')
    sdk.ClaudeAgentOptions=lambda **kw: SimpleNamespace(**kw)
    async def empty_query(**kwargs):
        if False:yield None
    sdk.query=empty_query
    monkeypatch.setitem(sys.modules,'claude_agent_sdk',sdk)
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'0.2.87')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'2.1.153 (Claude Code)')
    return sdk


def test_sdk_uses_native_preset_pinned_binary_and_isolated_tools(fake_sdk, tmp_path):
    options=sdk_options('initial',tmp_path,SDK_ENV,'/opt/nice-sdk/bin/claude-pinned')
    assert options.max_turns==250
    assert options.cli_path=='/opt/nice-sdk/bin/claude-pinned'
    assert options.system_prompt=={'type':'preset','preset':'claude_code'}
    assert options.setting_sources==[] and options.strict_mcp_config
    assert options.mcp_servers=={}
    assert 'Task' not in options.tools and 'WebFetch' in options.disallowed_tools


@pytest.mark.parametrize('scenario',['no_final','api_error','bad_final','timeout'])
def test_sdk_failure_preserves_events_without_completion(fake_sdk,tmp_path,scenario):
    assistant=make_dataclass('AssistantMessage',['content','error'])
    result=make_dataclass('ResultMessage',['subtype','is_error'])
    async def query(**kw):
        yield assistant([], 'invalid_request' if scenario=='api_error' else None)
        if scenario=='bad_final':yield result('error_max_turns',True)
        if scenario=='timeout':await asyncio.sleep(10)
    fake_sdk.query=query
    output=tmp_path/'events.jsonl'
    with pytest.raises((RuntimeError,TimeoutError)):
        asyncio.run(sdk_session('test','initial',tmp_path,SDK_ENV,'/opt/nice-sdk/bin/claude-pinned',
                               output,wall_seconds=.01))
    assert output.read_text(encoding='utf-8')


def test_sdk_refuses_to_overwrite_previous_events(fake_sdk,tmp_path):
    output=tmp_path/'events.jsonl'
    output.write_text('old evidence')
    with pytest.raises(FileExistsError):
        asyncio.run(sdk_session('test','initial',tmp_path,SDK_ENV,'/opt/nice-sdk/bin/claude-pinned',output))
    assert output.read_text()=='old evidence'


def test_configuration_matches_sdk_runtime_versions():
    import yaml
    from nice_validation.sdk_agent import SDK_VERSION, CLI_VERSION
    cfg=yaml.safe_load(Path('configs/reevaluation.yaml').read_text(encoding='utf-8'))
    assert cfg['agent_interface']=='claude_agent_sdk'
    assert cfg['claude_agent_sdk_version']==SDK_VERSION
    assert cfg['claude_code_version']==CLI_VERSION
    assert cfg['agent']['max_turns']==250


def test_tool_hook_clamps_timeout_and_disables_background(fake_sdk,tmp_path):
    fake_sdk.HookMatcher=lambda **kw:SimpleNamespace(**kw)
    result=make_dataclass('ResultMessage',['subtype','is_error'])
    seen=[]
    async def query(**kw):
        hook=kw['options'].hooks['PreToolUse'][0].hooks[0]
        update=await hook({'hook_event_name':'PreToolUse','tool_name':'Bash',
                           'tool_input':{'command':'sleep 2','timeout':999999,'run_in_background':True}},'id',None)
        updated=update['hookSpecificOutput']['updatedInput']
        assert updated['timeout']==1000 and updated['run_in_background'] is False
        yield result('error_max_turns',True)
    fake_sdk.query=query
    receipt=asyncio.run(sdk_session('test','initial',tmp_path,SDK_ENV,'/opt/nice-sdk/bin/claude-pinned',
        tmp_path/'events.jsonl',tool_seconds=1,allow_action_limit=True,event_sink=seen.append))
    assert receipt['result']['subtype']=='error_max_turns'
    assert any('ToolSupervisor' in line for line in seen)


def test_build_download_requires_reference_checksum_before_network(monkeypatch):
    monkeypatch.delenv('CLAUDE_SHA256',raising=False)
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('network before checksum validation'))
    with pytest.raises(SystemExit,match='Supply official'):
        runpy.run_path('docs/reevaluation/install_sdk_runtime.py')


def test_build_rejects_corrupt_runtime_without_executing_it(monkeypatch):
    import platform
    monkeypatch.setenv('CLAUDE_SHA256','0'*64)
    monkeypatch.setattr(platform,'system',lambda:'Linux')
    monkeypatch.setattr(platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'corrupt release'))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:pytest.fail('executed unchecked binary'))
    with pytest.raises(SystemExit,match='checksum mismatch'):
        runpy.run_path('docs/reevaluation/install_sdk_runtime.py')


@pytest.mark.parametrize('change',[{'ANTHROPIC_BASE_URL':''}, {'ANTHROPIC_BASE_URL':'http://host/v1'},
                                   {'ANTHROPIC_AUTH_TOKEN':''}])
def test_sdk_rejects_unconfigured_backend(fake_sdk,tmp_path,change):
    with pytest.raises(ValueError):
        sdk_options('initial',tmp_path,{**SDK_ENV,**change},'/opt/nice-sdk/bin/claude-pinned')


def test_sdk_overrides_wrong_auxiliary_model_and_nested_flag(fake_sdk,tmp_path):
    env={**SDK_ENV,'CLAUDECODE':'1','ANTHROPIC_DEFAULT_HAIKU_MODEL':'external-model'}
    options=sdk_options('chosen-adapter',tmp_path,env,'/opt/nice-sdk/bin/claude-pinned')
    assert 'CLAUDECODE' not in options.env
    assert options.env['ANTHROPIC_DEFAULT_HAIKU_MODEL']=='chosen-adapter'
    assert env['ANTHROPIC_DEFAULT_HAIKU_MODEL']=='external-model'
