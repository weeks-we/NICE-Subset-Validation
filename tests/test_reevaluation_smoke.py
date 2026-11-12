import pytest
import json
import subprocess
import sys
import tempfile
import importlib.metadata
from dataclasses import make_dataclass
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace, ModuleType

from scripts.smoke_reevaluation import PROGRAM, smoke_command


def test_smoke_routes_every_model_alias_without_secret_in_command(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret-value")
    cmd = smoke_command("existing:image", "restricted-net", "http://10.1.2.3:8000", "low-lr1", "2.1.153")
    assert "secret-value" not in " ".join(cmd)
    assert "--pull=never" in cmd and "1000:1000" in cmd
    for tier in ["OPUS", "SONNET", "HAIKU"]:
        assert "ANTHROPIC_DEFAULT_" + tier + "_MODEL=low-lr1" in cmd
    assert "CLAUDE_CODE_SUBAGENT_MODEL=low-lr1" in cmd
    assert "-v" not in cmd and "--privileged" not in cmd
    compile(PROGRAM, "container_probe", "exec")


@pytest.mark.parametrize("model,version", [("low/lr1", "2.1.153"), ("", "2.1.153"), ("initial", "latest"), ("initial", "")])
def test_smoke_rejects_unpinned_or_ambiguous_identity(model, version):
    with pytest.raises(ValueError):
        smoke_command("image", "net", "http://10.1.2.3:8000", model, version)


def test_smoke_requires_explicit_local_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="credentials"):
        smoke_command("image", "net", "http://10.1.2.3:8000", "initial", "2.1.153")


@pytest.mark.parametrize('scenario', ['complete', 'single_round', 'tool_error', 'wrong_challenge'])
def test_multiround_receipt_rejects_incomplete_feedback(monkeypatch, capsys, tmp_path, scenario):
    monkeypatch.setenv('ANTHROPIC_BASE_URL','http://10.1.2.3:8000')
    monkeypatch.setenv('ANTHROPIC_API_KEY','mock-key')
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN','mock-key')
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', lambda **kw: nullcontext(str(tmp_path)))
    monkeypatch.setattr(sys, 'argv', ['probe', '2.1.153', 'initial'])
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: '2.1.153 (Claude Code)')
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.2.87')
    assistant=make_dataclass('AssistantMessage', ['content'])
    user=make_dataclass('UserMessage', ['content'])
    result=make_dataclass('ResultMessage', ['subtype','is_error'])
    async def fake_query(prompt, options):
        root = Path(options.cwd)
        (root/'proof.txt').write_text((root/'source.txt').read_text())
        (root/'challenge.txt').write_text('fresh-result')
        (root/'answer.txt').write_text('wrong' if scenario=='wrong_challenge' else 'fresh-result')
        tools = ['Read'] if scenario=='single_round' else ['Read','Write','Bash','Write']
        for i,name in enumerate(tools):
            yield assistant([{'id':str(i),'name':name}])
            yield user([{'tool_use_id':str(i),'is_error':scenario=='tool_error'}])
        yield result('success',False)
    sdk=ModuleType('claude_agent_sdk')
    sdk.query=fake_query
    sdk.ClaudeAgentOptions=lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules,'claude_agent_sdk',sdk)
    if scenario=='complete':
        exec(compile(PROGRAM, 'mock_probe', 'exec'), {})
    else:
        with pytest.raises(SystemExit):
            exec(compile(PROGRAM, 'mock_probe', 'exec'), {})
    receipt=json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt['status']==('PASS' if scenario=='complete' else 'FAIL')
    assert receipt['thinking_blocks_observed']==0  # No false certification of thinking.


def test_outer_timeout_saves_partial_logs_and_failure(monkeypatch,tmp_path):
    from scripts import smoke_reevaluation as smoke
    output=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['smoke','--image','prepared:image','--network','restricted',
        '--endpoint','http://10.1.2.3:8000','--model','initial','--claude-version','2.1.153',
        '--output',str(output),'--execute-server'])
    monkeypatch.setattr(smoke.platform,'system',lambda:'Linux')
    monkeypatch.setattr(smoke,'probe_network',lambda *a:{'status':'PASS'})
    monkeypatch.setenv('ANTHROPIC_API_KEY','mock-key')
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN','mock-key')
    def run(cmd,**kw):
        if 'run' in cmd:
            raise subprocess.TimeoutExpired(cmd,660,output=b'partial-sdk-event',stderr=b'partial-error')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    with pytest.raises(subprocess.TimeoutExpired):smoke.main()
    assert (output/'stdout.jsonl').read_text()=='partial-sdk-event'
    assert (output/'stderr.log').read_text()=='partial-error'
    assert json.loads((output/'status.json').read_text())['status']=='FAIL'
