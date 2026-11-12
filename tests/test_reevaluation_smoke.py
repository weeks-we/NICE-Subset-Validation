import pytest
import json
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from scripts.smoke_reevaluation import PROGRAM, smoke_command


def test_smoke_routes_every_model_alias_without_secret_in_command(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret-value")
    cmd = smoke_command("existing:image", "restricted-net", "http://10.1.2.3:8000", "low-lr1", "2.1.187")
    assert "secret-value" not in " ".join(cmd)
    assert "--pull=never" in cmd and "1000:1000" in cmd
    for tier in ["OPUS", "SONNET", "HAIKU"]:
        assert "ANTHROPIC_DEFAULT_" + tier + "_MODEL=low-lr1" in cmd
    assert "CLAUDE_CODE_SUBAGENT_MODEL=low-lr1" in cmd
    assert "-v" not in cmd and "--privileged" not in cmd
    compile(PROGRAM, "container_probe", "exec")


@pytest.mark.parametrize("model,version", [("low/lr1", "2.1.187"), ("", "2.1.187"), ("initial", "latest"), ("initial", "")])
def test_smoke_rejects_unpinned_or_ambiguous_identity(model, version):
    with pytest.raises(ValueError):
        smoke_command("image", "net", "http://10.1.2.3:8000", model, version)


def test_smoke_requires_explicit_local_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="credentials"):
        smoke_command("image", "net", "http://10.1.2.3:8000", "initial", "2.1.187")


@pytest.mark.parametrize('scenario', ['complete', 'single_round', 'tool_error', 'wrong_challenge'])
def test_multiround_receipt_rejects_incomplete_feedback(monkeypatch, capsys, tmp_path, scenario):
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', lambda **kw: nullcontext(str(tmp_path)))
    monkeypatch.setattr(sys, 'argv', ['probe', '2.1.153', 'initial'])
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: '2.1.153 (Claude Code)')
    def fake_run(*args, **kwargs):
        root = Path(kwargs['cwd'])
        (root/'proof.txt').write_text((root/'source.txt').read_text())
        (root/'challenge.txt').write_text('fresh-result')
        (root/'answer.txt').write_text('wrong' if scenario=='wrong_challenge' else 'fresh-result')
        tools = ['Read'] if scenario=='single_round' else ['Read','Write','Bash','Write']
        events = []
        for name in tools:
            events.extend([{'type':'assistant','message':{'content':[{'type':'tool_use','name':name}]}},
                {'type':'user','message':{'content':[{'type':'tool_result','is_error':scenario=='tool_error'}]}}])
        events.append({'type':'result','is_error':False})
        return SimpleNamespace(returncode=0,stdout='\n'.join(json.dumps(e) for e in events),stderr='')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    if scenario=='complete':
        exec(compile(PROGRAM, 'mock_probe', 'exec'), {})
    else:
        with pytest.raises(SystemExit):
            exec(compile(PROGRAM, 'mock_probe', 'exec'), {})
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['status']==('PASS' if scenario=='complete' else 'FAIL')
    assert receipt['thinking_blocks_observed']==0  # No false certification of thinking.
