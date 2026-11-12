import pytest

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
