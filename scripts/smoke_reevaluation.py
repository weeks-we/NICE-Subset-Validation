"""Fail-fast, real Claude Code -> local model -> file tools smoke; no benchmark."""
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


PROGRAM = r'''
import json,os,pathlib,secrets,subprocess,tempfile,sys
expected_version,model=sys.argv[1:]
version=subprocess.check_output(['claude','--version'],text=True,timeout=30).strip()
if version.split()[0]!=expected_version:raise RuntimeError('Claude Code version mismatch: '+version)
with tempfile.TemporaryDirectory(prefix='nice-cli-smoke-') as directory:
    root=pathlib.Path(directory)
    nonce=secrets.token_hex(24)
    (root/'source.txt').write_text(nonce)
    (root/'verify.py').write_text("import pathlib,secrets\nr=pathlib.Path('.')\nassert (r/'proof.txt').read_text()==(r/'source.txt').read_text()\np=secrets.token_hex(24)\n(r/'challenge.txt').write_text(p)\nprint('NEXT_CHALLENGE='+p)\n")
    env=dict(os.environ,HOME=str(root/'home'),CLAUDE_CONFIG_DIR=str(root/'home/.claude'))
    pathlib.Path(env['HOME']).mkdir()
    prompt='Complete these steps in order, waiting for each tool result. First read source.txt with Read. Next create proof.txt with Write containing exactly that content. Next use Bash to execute python verify.py. It verifies your file and generates a NEW random challenge that does not exist before execution. Finally use Write to create answer.txt containing exactly the NEXT_CHALLENGE value returned by Bash. Do not use Bash to write answer.txt. Do not ask questions. Finish after writing answer.txt. This is an integration check, not a benchmark.'
    result=subprocess.run(['claude','-p',prompt,'--model',model,'--max-turns','12',
        '--tools','Bash,Read,Edit,Write,Glob,Grep','--dangerously-skip-permissions',
        '--output-format','stream-json','--verbose'],cwd=root,env=env,
        capture_output=True,text=True,timeout=600)
    # Full stream retained by outer controller; never include API secrets.
    events=[]
    for line in result.stdout.splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    finals=[e for e in events if e.get('type')=='result']
    tool_calls=[]
    assistant_rounds=0
    thinking_blocks=0
    tool_results=[]
    for e in events:
        if e.get('type')=='assistant':
            assistant_rounds+=1
            for block in e.get('message',{}).get('content',[]):
                if block.get('type')=='tool_use':tool_calls.append(block.get('name'))
                if block.get('type')=='thinking' and block.get('thinking'):thinking_blocks+=1
        if e.get('type')=='user':
            for block in e.get('message',{}).get('content',[]):
                if block.get('type')=='tool_result':tool_results.append(block)
    challenge_ok=((root/'challenge.txt').is_file() and (root/'answer.txt').is_file()
        and (root/'challenge.txt').read_text()==(root/'answer.txt').read_text())
    passed=(result.returncode==0 and finals and not finals[-1].get('is_error',False)
        and assistant_rounds>=3 and {'Read','Write','Bash'}.issubset(tool_calls)
        and len(tool_results)>=3 and not any(b.get('is_error') for b in tool_results)
        and challenge_ok and (root/'proof.txt').is_file() and (root/'proof.txt').read_text()==nonce)
    print(json.dumps({'status':'PASS' if passed else 'FAIL','model':model,'cli_version':version,
        'tool_calls':tool_calls,'exit_code':result.returncode,'events':events,
        'assistant_rounds':assistant_rounds,'tool_result_count':len(tool_results),
        'thinking_blocks_observed':thinking_blocks,'challenge_verified':bool(challenge_ok),
        'stderr':result.stderr,'scope':'real multi-round CLI/model/Read/Write/Bash feedback only; thinking continuity and official grade not certified'}))
    if not passed:sys.exit(1)
'''


def smoke_command(image, network, endpoint, model, version):
    if not version or version == "latest":
        raise ValueError("Pin an exact Claude Code version")
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
        if not os.environ.get(key):
            raise ValueError("Set local inference credentials in environment: " + key)
        command += ["--env", key]
    command += ["--entrypoint", "python", image, "-c", PROGRAM, version, model]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="Locally prepared image with python and fixed claude CLI")
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
        result = subprocess.run(command, capture_output=True, text=True, timeout=660)
        (a.output / "stdout.jsonl").write_text(result.stdout, encoding="utf-8")
        (a.output / "stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise ValueError("Real Claude Code/model/tools smoke failed; inspect logs")
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        if receipt.get("status") != "PASS" or receipt.get("model") != a.model or not receipt.get("tool_calls"):
            raise ValueError("Missing real tool execution receipt")
        write(a.output / "status.json", {"status": "PASS", "formal_evidence": False,
              "model": a.model, "cli_version": a.claude_version,
              "stdout_sha256": sha(a.output / "stdout.jsonl"),
              "scope": "Network and real CLI/model/file-tools only; capacity/LoRA numerics/official grade not certified"})
    except BaseException as exc:
        write(a.output / "status.json", {"status": "FAIL", "formal_evidence": False,
                                        "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        # A killed docker client can leave its container alive. Remove only ours.
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
    print("Real integration smoke PASS; complete capacity, adapter and official grading acceptance before formal jobs")


if __name__ == "__main__":
    main()
