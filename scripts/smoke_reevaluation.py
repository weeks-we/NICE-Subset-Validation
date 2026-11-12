"""Fail-fast, real Claude Agent SDK -> local model -> tools smoke; no benchmark."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.core import sha, write
from nice_validation.reevaluation_deploy import network_environment, probe_network
from nice_validation.sdk_agent import SDK_VERSION, CLI_VERSION


PROGRAM = (Path(__file__).resolve().parents[1] / 'nice_validation/sdk_agent.py').read_text(encoding='utf-8') + r'''
import pathlib,secrets,tempfile,sys
expected_version,model=sys.argv[1:]
if expected_version != CLI_VERSION:raise RuntimeError('Unexpected Claude runtime version')
with tempfile.TemporaryDirectory(prefix='nice-cli-smoke-') as directory:
    root=pathlib.Path(directory)
    nonce=secrets.token_hex(24)
    (root/'source.txt').write_text(nonce+'\n')
    (root/'verify.py').write_text("import pathlib,secrets\nr=pathlib.Path('.')\nassert (r/'proof.txt').read_text()==(r/'source.txt').read_text()\np=secrets.token_hex(24)\n(r/'challenge.txt').write_text(p+'\\n')\nprint('NEXT_CHALLENGE='+p)\n")
    env=dict(os.environ,HOME=str(root/'home'),CLAUDE_CONFIG_DIR=str(root/'home/.claude'))
    pathlib.Path(env['HOME']).mkdir()
    prompt='Complete these steps in order, waiting for each tool result. First read source.txt with Read. Next create proof.txt with Write containing exactly that content. Next use Bash to execute python verify.py. It verifies your file and generates a NEW random challenge that does not exist before execution. Finally use Write to create answer.txt containing exactly the NEXT_CHALLENGE value returned by Bash. Do not use Bash to write answer.txt. Do not ask questions. Finish after writing answer.txt. This is an integration check, not a benchmark.'
    receipt=asyncio.run(sdk_session(prompt,model,root,env,
        '/opt/nice-sdk/bin/claude-pinned',root/'sdk-events.jsonl',max_turns=12,wall_seconds=600,
        event_sink=lambda line:print(line,flush=True)))
    tool_calls=receipt['tool_calls'];tool_results=receipt['tool_results']
    challenge_ok=((root/'challenge.txt').is_file() and (root/'answer.txt').is_file()
        and (root/'challenge.txt').read_text()==(root/'answer.txt').read_text())
    passed=(receipt['assistant_rounds']>=3 and {'Read','Write','Bash'}.issubset({c['name'] for c in tool_calls})
        and len(tool_results)>=3 and not any(b.get('is_error') for b in tool_results)
        and {c['id'] for c in tool_calls}.issubset({b['tool_use_id'] for b in tool_results})
        and challenge_ok and (root/'proof.txt').is_file() and (root/'proof.txt').read_text()==nonce+'\n')
    print(json.dumps({**receipt,'status':'PASS' if passed else 'FAIL',
        'tool_result_count':len(tool_results),'challenge_verified':bool(challenge_ok)}))
    if not passed:sys.exit(1)
'''


def smoke_command(image, network, endpoint, model, version, credentials=None):
    if version != CLI_VERSION:
        raise ValueError("Use the frozen Claude runtime version " + CLI_VERSION)
    if not model or any(c.isspace() for c in model) or "/" in model:
        raise ValueError("Use the actual served model/adapter alias, without slash")
    env = network_environment(endpoint)
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "8192"  # Short engineering probe, not formal budget.
    env.update({"ANTHROPIC_MODEL": model, "CLAUDE_CODE_SUBAGENT_MODEL": model,
                **{"ANTHROPIC_DEFAULT_" + tier + "_MODEL": model for tier in ["OPUS", "SONNET", "HAIKU"]}})
    command = ["docker", "run", "--rm", "--pull=never", "--network", network,
               "--user", "1000:1000", "--cap-drop=ALL", "--security-opt=no-new-privileges",
               "--pids-limit=256", "--workdir", "/tmp", "--tmpfs", "/tmp:rw,exec,mode=1777"]
    for key, value in env.items():
        command += ["--env", key + "=" + value]
    # Forward values without printing them or exposing secrets in command arguments.
    for key in ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"]:
        if not (os.environ if credentials is None else credentials).get(key):
            raise ValueError("Set local inference credentials in environment: " + key)
        command += ["--env", key]
    command += ["--entrypoint", "/opt/nice-sdk/bin/python", image, "-c", PROGRAM, version, model]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="Prepared SDK image with /opt/nice-sdk and pinned Claude runtime")
    p.add_argument("--network", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--claude-version", required=True)
    p.add_argument("--output", required=True, type=Path, help="New independent smoke directory")
    p.add_argument("--execute-server", action="store_true")
    a = p.parse_args()
    if not a.execute_server:
        print("Plan only: network probe followed by isolated real CLI/model file-tool smoke; no benchmark")
        return
    if platform.system() != "Linux":
        p.error("Real smoke is authorized Linux server execution only")
    if a.output.exists():
        p.error("Use a fresh smoke output directory; preserve previous evidence")
    a.output.mkdir(parents=True)
    container_name = "nice-cli-smoke-" + uuid.uuid4().hex
    write(a.output / "status.json", {"status": "RUNNING", "formal_evidence": False})
    try:
        network = probe_network(a.image, a.network, a.endpoint)
        write(a.output / "network.json", network)
        command = smoke_command(a.image, a.network, a.endpoint, a.model, a.claude_version)
        command[2:2] = ["--name", container_name]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=660)
        except subprocess.TimeoutExpired as exc:
            for filename, value in [('stdout.jsonl',exc.stdout),('stderr.log',exc.stderr)]:
                if isinstance(value,bytes):value=value.decode('utf-8',errors='replace')
                (a.output/filename).write_text(value or '',encoding='utf-8')
            raise
        (a.output / "stdout.jsonl").write_text(result.stdout, encoding="utf-8")
        (a.output / "stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise ValueError("Real Claude Code/model/tools smoke failed; inspect logs")
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        if (receipt.get("status") != "PASS" or receipt.get("model") != a.model
                or receipt.get('sdk_version') != SDK_VERSION or not receipt.get("tool_calls")):
            raise ValueError("Missing real tool execution receipt")
        write(a.output / "status.json", {"status": "PASS", "formal_evidence": False,
              "model": a.model, "cli_version": a.claude_version,
              "sdk_version": SDK_VERSION, "agent_interface": "claude_agent_sdk",
              "stdout_sha256": sha(a.output / "stdout.jsonl"),
              "scope": "Network and real CLI/model/file-tools only; capacity/LoRA numerics/official grade not certified"})
    except BaseException as exc:
        write(a.output / "status.json", {"status": "FAIL", "formal_evidence": False,
                                        "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        # A killed docker client can leave its container alive. Remove only ours.
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
    print("Real SDK smoke PASS; complete capacity, adapter and official grading acceptance before formal jobs")


if __name__ == "__main__":
    main()
