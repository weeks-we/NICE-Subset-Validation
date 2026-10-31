"""LangChain tool routing, isolated Docker episodes and official task grading."""

from __future__ import annotations

import json
import hashlib
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from nice_validation.core import Run, digest, read, require_commit, sha, write
from nice_validation.protocol import TOOLS, arguments


class InfraError(RuntimeError):
    pass


class TaskFailure(RuntimeError):
    pass


def patch_command(root: str, commit: str, output: str | None = None) -> str:
    """Diff against the immutable starting commit, including staged/new files."""
    require_commit(commit, "patch starting commit")
    directory = shlex.quote(root)
    destination = " --output=" + shlex.quote(output) if output is not None else ""
    return (
        f"git -C {directory} add -N -- . && "
        f"git -C {directory} diff --binary --no-ext-diff --no-textconv{destination} {shlex.quote(commit)} --"
    )


EDITOR_PROGRAM = r"""
import hashlib,json,os,pathlib,sys
args=json.load(sys.stdin)
p=pathlib.Path(args['path'])
command=args['command']
if not p.is_absolute():raise ValueError('path must be absolute')
store=pathlib.Path(os.environ.get('NICE_EDITOR_HISTORY','/tmp/nice-editor-history'))
store.mkdir(parents=True,exist_ok=True)
history=store/hashlib.sha256(str(p).encode()).hexdigest()
versions=json.loads(history.read_text(encoding='utf-8')) if history.exists() else []
def read_text():return p.read_bytes().decode('utf-8')
def publish(text,old):
    p.write_bytes(text.encode('utf-8'))
    versions.append(old)
    history.write_text(json.dumps(versions),encoding='utf-8')
if command=='view':
    if p.is_dir():
        print('\n'.join(str(x) for x in sorted(p.glob('*'))+sorted(p.glob('*/*')) if not any(q.startswith('.') for q in x.relative_to(p).parts)))
    else:
        lines=read_text().splitlines()
        left,right=args.get('view_range',[1,-1])
        right=len(lines) if right==-1 else right
        if left<1 or right<left or right>len(lines):raise ValueError('view_range out of range')
        print('\n'.join(f'{i+1}\t{line}' for i,line in enumerate(lines) if left<=i+1<=right))
elif command=='create':
    if p.exists():raise ValueError('file already exists')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(args['file_text'].encode('utf-8'))
    print('Created')
elif command=='str_replace':
    text=read_text();old=args['old_str']
    if not old or text.count(old)!=1:raise ValueError('old_str must match exactly once')
    publish(text.replace(old,args.get('new_str',''),1),text)
    print('Replaced')
elif command=='insert':
    text=read_text();lines=text.splitlines();line=args['insert_line']
    if not 0<=line<=len(lines):raise ValueError('insert_line out of range')
    inserted=args['new_str'].splitlines()
    lines[line:line]=inserted
    publish('\n'.join(lines)+'\n',text)
    print('Inserted')
elif command=='undo_edit':
    if not versions:raise ValueError('no edit history for this path')
    previous=versions.pop()
    p.write_bytes(previous.encode('utf-8'))
    history.write_text(json.dumps(versions),encoding='utf-8')
    print('Undone')
else:raise ValueError('unsupported editor command')
"""


class DockerTask:
    def __init__(self, cfg: dict, task: dict, folder: Path):
        self.cfg, self.task, self.folder = cfg, task, folder
        self.container: str | None = None
        self.shell = None
        self.shell_lines = queue.Queue()
        self.deadline: float | None = None
        self.checkout_commit: str | None = None

    def __enter__(self):
        images = read(Path(self.cfg["agent"]["image_map"]))
        lock = read(Path(self.cfg["agent"]["image_revision_lock"]))
        entry = images.get(self.task["instance_id"])
        if not entry:
            raise InfraError("task_image_missing")
        if isinstance(entry, str):
            raise InfraError(
                "Image map must include image, base_commit and public_environment_only attestation"
            )
        if (
            entry["base_commit"] != self.task["base_commit"]
            or entry.get("public_environment_only") is not True
        ):
            raise InfraError("task_image_identity_or_isolation_mismatch")
        image = entry["image"]
        if "@sha256:" not in image and not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise InfraError("Task images must use immutable repository digests")
        if lock.get("image_map_sha256") != sha(Path(self.cfg["agent"]["image_map"])):
            raise InfraError("image_map_lock_mismatch")
        p = subprocess.run(
            [
                "docker",
                "create",
                "--label",
                f"nice.run={self.cfg.get('_run_identity', digest(str(Path(self.cfg['output']).resolve())))}",
                "--label",
                f"nice.attempt={digest(str(self.folder.resolve()))}",
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "512",
                "--workdir",
                self.cfg["agent"]["workdir"],
                "--entrypoint",
                "/bin/bash",
                image,
                "-lc",
                "sleep infinity",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if p.returncode:
            raise InfraError("docker_create:" + p.stderr)
        self.container = p.stdout.strip()
        try:
            write(
                self.folder / "container.json",
                {
                    "container_id": self.container,
                    "run_identity": self.cfg.get("_run_identity"),
                    "task_id": self.task["instance_id"],
                    "image": image,
                    "status": "created",
                },
            )
            subprocess.run(
                ["docker", "start", self.container],
                check=True,
                capture_output=True,
                timeout=60,
            )
            self.shell = subprocess.Popen(
                [
                    "docker",
                    "exec",
                    "-i",
                    self.container,
                    "bash",
                    "--noprofile",
                    "--norc",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            def pump():
                for line in self.shell.stdout:
                    self.shell_lines.put(line)
                self.shell_lines.put(None)

            threading.Thread(target=pump, daemon=True).start()
            setup = entry.get("shell_setup", "")
            if setup:
                self.command(setup, self.cfg["agent"]["tool_seconds"])
            result = self.command(
                "git rev-parse HEAD", self.cfg["agent"]["tool_seconds"]
            )
            expected_commit = (
                entry.get("checkout_commit")
                or self.command(
                    "git rev-parse "
                    + shlex.quote(self.task["base_commit"] + "^{commit}"),
                    60,
                ).strip()
            )
            if result.strip() != expected_commit:
                raise InfraError("task_checkout_base_commit_mismatch")
            require_commit(expected_commit, "task starting commit")
            self.checkout_commit = expected_commit
            write(self.folder / "patch_base.json", {"commit": expected_commit})
            if self.command("git status --porcelain --untracked-files=all", 60).strip():
                raise InfraError("task_checkout_not_clean")
            hidden = self.command(
                'for p in /eval.sh /tmp/test.patch /tmp/patch.diff; do if [ -e "$p" ]; then echo "$p"; fi; done',
                60,
            )
            if hidden.strip():
                raise InfraError("task_image_contains_grader_artifacts:" + hidden)
            return self
        except BaseException:
            self.close()
            raise

    def command(
        self, command: str, timeout: int, stdin: str | None = None, editor: bool = False
    ) -> str:
        if self.container is None:
            raise InfraError("container_not_initialized")
        if not editor:
            marker = "NICE_" + uuid.uuid4().hex
            try:
                self.shell.stdin.write(
                    command + "\nprintf '\\n" + marker + ':%s\\n\' "$?"\n'
                )
                self.shell.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise TaskFailure("SHELL_TERMINATED") from exc
            lines, started = [], time.monotonic()
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TaskFailure("TOOL_TIMEOUT")
                try:
                    line = self.shell_lines.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TaskFailure("TOOL_TIMEOUT") from exc
                if line is None:
                    raise TaskFailure("SHELL_TERMINATED")
                if line.startswith(marker + ":"):
                    code = int(line.strip().rsplit(":", 1)[1])
                    return "".join(lines).rstrip("\n") + (
                        f"\n[exit_code={code}]" if code else ""
                    )
                lines.append(line)
        argv = [
            "docker",
            "exec",
            "-i",
            self.container,
            "timeout",
            "-k",
            "5",
            str(timeout),
        ]
        argv += ["python", "-c", EDITOR_PROGRAM] if editor else ["bash", "-lc", command]
        try:
            p = subprocess.run(
                argv, input=stdin, capture_output=True, text=True, timeout=timeout + 15
            )
        except subprocess.TimeoutExpired as exc:
            raise InfraError("docker_exec_unresponsive") from exc
        if p.returncode in {125, 126, 127} and (
            "Error response from daemon" in p.stderr or "No such container" in p.stderr
        ):
            raise InfraError("docker_exec:" + p.stderr)
        output = p.stdout + p.stderr
        if p.returncode:
            output += f"\n[exit_code={p.returncode}]"
        return output

    def tools(self):
        from langchain_core.tools import StructuredTool

        def remaining_tool_seconds() -> int:
            remaining = (
                int(self.deadline - time.monotonic())
                if self.deadline
                else self.cfg["agent"]["tool_seconds"]
            )
            if remaining <= 0:
                raise TaskFailure("WALL_LIMIT")
            return min(remaining, self.cfg["agent"]["tool_seconds"])

        def bash(command: str) -> str:
            """Execute a shell command in the task repository."""
            return self.command(command, remaining_tool_seconds())

        def str_replace_editor(
            command: str,
            path: str,
            file_text: str | None = None,
            old_str: str | None = None,
            new_str: str | None = None,
            insert_line: int | None = None,
            view_range: list[int] | None = None,
        ) -> str:
            """View or edit a UTF-8 file in the task repository."""
            args = {"command": command, "path": path}
            for key, value in [
                ("file_text", file_text),
                ("old_str", old_str),
                ("new_str", new_str),
                ("insert_line", insert_line),
                ("view_range", view_range),
            ]:
                if value is not None:
                    args[key] = value
            return self.command(
                "", remaining_tool_seconds(), json.dumps(args), editor=True
            )

        def submit() -> str:
            """Finish and submit the current repository diff."""
            return "Submitted"

        functions = {
            "bash": bash,
            "str_replace_editor": str_replace_editor,
            "submit": submit,
        }
        return {
            name: StructuredTool.from_function(
                function,
                args_schema=next(
                    t["function"]["parameters"]
                    for t in TOOLS
                    if t["function"]["name"] == name
                ),
            )
            for name, function in functions.items()
        }

    def patch(self) -> str:
        if self.checkout_commit is None:
            raise InfraError("patch_starting_commit_missing")
        result = subprocess.run(
            [
                "docker",
                "exec",
                self.container,
                "bash",
                "-lc",
                patch_command(self.cfg["agent"]["workdir"], self.checkout_commit),
            ],
            capture_output=True,
            timeout=60,
        )
        stderr = (
            result.stderr.decode("utf-8", errors="replace")
            if isinstance(result.stderr, bytes)
            else result.stderr
        )
        if result.returncode:
            if "fatal:" in stderr or stderr.lstrip().startswith("error:"):
                raise TaskFailure("PATCH_COLLECTION_FAILED")
            raise InfraError("patch_collection:" + stderr)
        try:
            return (
                result.stdout.decode("utf-8")
                if isinstance(result.stdout, bytes)
                else result.stdout
            )
        except UnicodeDecodeError as exc:
            raise TaskFailure("PATCH_COLLECTION_FAILED") from exc

    def close(self) -> None:
        if self.container:
            result = subprocess.run(
                ["docker", "rm", "-f", self.container],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode:
                raise InfraError("docker_cleanup:" + result.stderr)
            self.container = None
            path = self.folder / "container.json"
            if path.is_file():
                write(path, {**read(path), "status": "removed"})
        if self.shell is not None:
            self.shell.terminate()
            try:
                self.shell.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.shell.kill()
                self.shell.wait(timeout=10)

    def __exit__(self, exception_type, exception, traceback):
        try:
            if isinstance(exception, TaskFailure):
                try:
                    captured = self.patch()
                except TaskFailure:
                    captured = ""
                (self.folder / "patch.diff").write_bytes(captured.encode("utf-8"))
        finally:
            self.close()


def official_grade(cfg: dict, task: dict, patch: str, folder: Path) -> bool:
    """Invoke SWE-bench on a frozen local task file; no floating dataset reload."""
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    from nice_validation.environments import official_identity

    write(folder / "harness_identity.json", official_identity(cfg))
    entry = read(Path(cfg["agent"]["image_map"]))[task["instance_id"]]
    expected_image = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            entry.get("grader_image", entry["image"]),
            "--format",
            "{{.Id}}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    original_image = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            entry["official_image_name"],
            "--format",
            "{{.Id}}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    if expected_image != original_image:
        raise InfraError("official_grader_image_tag_changed")
    dataset = folder / "task.json"
    predictions = folder / "predictions.jsonl"
    write(dataset, [{**task, "image": entry["image"]}])
    predictions.write_text(
        json.dumps(
            {
                "instance_id": task["instance_id"],
                "model_name_or_path": "nice_model",
                "model_patch": patch,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if not patch.strip():
        write(
            folder / "verdict.json",
            {"resolved": False, "reason": "empty_patch_official_failure_convention"},
        )
        return False
    run_id = "grade_" + uuid.uuid4().hex[:12]
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset),
        "--predictions_path",
        str(predictions),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--timeout",
        str(cfg["agent"]["evaluation_timeout"]),
    ]
    backend = task.get(
        "evaluation_backend", cfg["data"].get("validation_kind", "swesmith")
    )
    if backend == "swesmith":
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "scripts/swesmith_grade.py"),
            "--dataset_path",
            str(dataset),
            "--predictions_path",
            str(predictions),
            "--run_id",
            run_id,
            "--commit",
            entry["checkout_commit"],
            "--timeout",
            str(cfg["agent"]["evaluation_timeout"]),
        ]
    elif backend == "live":
        cmd[0] = cfg["transfer"]["live_python"]
        cmd += ["--namespace", cfg["transfer"]["suites"]["live"]["namespace"]]
    else:
        cmd += ["--namespace", "swebench"]
    write(folder / "command.json", cmd)
    with (
        (folder / "harness.stdout.log").open("w", encoding="utf-8") as stdout,
        (folder / "harness.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        try:
            p = subprocess.run(
                cmd,
                cwd=folder,
                stdout=stdout,
                stderr=stderr,
                timeout=cfg["agent"]["evaluation_timeout"] + 300,
            )
        except subprocess.TimeoutExpired as exc:
            raise InfraError("official_harness_timeout") from exc
    if p.returncode:
        raise InfraError(f"official_harness_exit_{p.returncode}")
    final_image = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            entry["official_image_name"],
            "--format",
            "{{.Id}}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    if final_image != expected_image:
        raise InfraError("official_grader_image_tag_changed_during_evaluation")
    reports = list(folder.glob(f"logs/run_evaluation/{run_id}/**/report.json"))
    reports += list(folder.glob(f"logs/run_evaluation/{run_id}/**/report.jsonl"))
    for report in reports:
        value = read(report)
        evidence = value.get(task["instance_id"], value)
        if isinstance(evidence, dict) and isinstance(evidence.get("resolved"), bool):
            write(
                folder / "verdict.json",
                {
                    "resolved": evidence["resolved"],
                    "report": str(report.relative_to(folder)),
                    "report_sha256": sha(report),
                },
            )
            return evidence["resolved"]
    raise InfraError(
        "official_harness_verdict_missing; inspect installed SWE-bench API and logs"
    )


def request_generation(
    url: str, payload: dict, timeout: float, deadline: float | None = None
) -> dict:
    # An explicitly supplied episode deadline separates a task budget from an
    # independent service timeout. Connection failures before it remain infra.
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TaskFailure("WALL_LIMIT")
        timeout = min(timeout, remaining)
    req = urllib.request.Request(
        url.rstrip("/") + "/generate",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.load(response)
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskFailure("WALL_LIMIT")
        return value
    except urllib.error.HTTPError as exc:
        details = exc.read().decode()
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskFailure("WALL_LIMIT") from exc
        if "CONTEXT_LIMIT" in details:
            raise TaskFailure("CONTEXT_LIMIT") from exc
        if "INVALID_TOOL_CALL" in details:
            raise TaskFailure("INVALID_TOOL_CALL") from exc
        raise InfraError("model_service:" + details) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskFailure("WALL_LIMIT") from exc
        raise InfraError("model_service_unavailable") from exc


def real_episode(
    run: Run, task: dict, model_id: str, nice: bool, seed: int, folder: Path
) -> dict:
    from nice_validation.model import tree_identity

    cfg = {**run.cfg, "_run_identity": run.identity}
    budget = cfg["agent"]
    initial = Path(cfg["model"]["adapter_path"])
    adapter = (
        initial
        if model_id == "initial"
        else run.root / "training" / model_id / "adapter"
    )
    adapter_sha = tree_identity(adapter)
    messages = [
        {
            "role": "system",
            "content": "Fix the public issue in the repository. Use the declared tools. Bash shell state persists within this task. Call submit when finished.",
        },
        {"role": "user", "content": task["problem_statement"]},
    ]
    traces, tools_count, outputs, inputs = [], 0, 0, 0
    started = time.monotonic()
    stop_reason = "ACTION_LIMIT"
    patch = ""
    try:
        with DockerTask(cfg, task, folder) as environment:
            environment.deadline = started + budget["wall_seconds"]
            tools = environment.tools()
            for index in range(budget["max_actions"]):
                remaining = budget["wall_seconds"] - (time.monotonic() - started)
                if remaining <= 0:
                    stop_reason = "WALL_LIMIT"
                    break
                allowance = min(
                    budget["max_new_tokens"], budget["total_new_tokens"] - outputs
                )
                if allowance <= 0:
                    stop_reason = "TOKEN_LIMIT"
                    break
                payload = {
                    "messages": messages,
                    "nice": nice,
                    "seed": int(digest([seed, index])[:8], 16),
                    "max_tokens": allowance,
                    "adapter_path": str(adapter.resolve()),
                    "adapter_sha256": adapter_sha,
                }
                write(folder / f"call_{index:03d}.request.json", payload)
                response = request_generation(
                    budget["service_url"],
                    payload,
                    remaining,
                    deadline=started + budget["wall_seconds"],
                )
                write(folder / f"call_{index:03d}.response.json", response)
                run.event(
                    cfg.get(
                        "_evaluation_scope",
                        "rollout-nice" if nice else "evaluate:" + model_id,
                    ),
                    "model_call",
                    task_id=task["instance_id"],
                    subset_id=None if model_id == "initial" else model_id,
                    call_index=index,
                    **response["usage"],
                    service_url=budget["service_url"],
                    context_length=len(response["input_ids"]),
                    prefill_seconds=None,
                    decode_seconds=None,
                )
                traces.append(response)
                write(folder / "trace.json", traces)
                inputs += response["usage"]["input_tokens"]
                outputs += response["usage"]["output_tokens"]
                if response.get("parse_error"):
                    raise TaskFailure(response["parse_error"])
                messages.append(response["message"])
                calls = response["message"].get("tool_calls", [])
                if not calls:
                    stop_reason = "FINAL_OUTPUT"
                    break
                finished = False
                for ci, call in enumerate(calls):
                    if time.monotonic() >= started + budget["wall_seconds"]:
                        raise TaskFailure("WALL_LIMIT")
                    try:
                        name, args = arguments(call)
                    except ValueError as exc:
                        raise TaskFailure("INVALID_TOOL_CALL:" + str(exc)) from exc
                    if name == "submit":
                        stop_reason, finished = "SUBMITTED", True
                        break
                    tool_start = time.monotonic()
                    result = tools[name].invoke(args)
                    write(
                        folder / f"call_{index:03d}.tool_{ci:02d}.json",
                        {
                            "call": call,
                            "result": result,
                            "wall_seconds": time.monotonic() - tool_start,
                        },
                    )
                    # Full output is retained. The observation policy is explicit and replayed.
                    observed = (
                        result
                        if len(result) <= 20000
                        else result[:10000]
                        + "\n[tool-output-policy-v1: middle omitted; full output saved]\n"
                        + result[-10000:]
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": observed,
                        }
                    )
                    tools_count += 1
                if finished:
                    break
            patch = environment.patch()
    except TaskFailure as exc:
        stop_reason = str(exc)
        if (folder / "patch.diff").exists():
            patch = (folder / "patch.diff").read_bytes().decode("utf-8")
    write(folder / "trace.json", traces)
    (folder / "patch.diff").write_bytes(patch.encode("utf-8"))
    exhausted = stop_reason in {
        "ACTION_LIMIT",
        "WALL_LIMIT",
        "TOKEN_LIMIT",
        "CONTEXT_LIMIT",
        "TOOL_TIMEOUT",
        "SHELL_TERMINATED",
        "PATCH_COLLECTION_FAILED",
    } or stop_reason.startswith("INVALID_TOOL_CALL")
    automatic_submission = (
        cfg["agent"].get("budget_exhaustion_policy", "fail") == "grade_patch"
        and stop_reason
        in {"ACTION_LIMIT", "WALL_LIMIT", "TOKEN_LIMIT", "CONTEXT_LIMIT"}
        and bool(patch.strip())
    )
    grade_eligible = not exhausted or automatic_submission
    resolved = (
        official_grade(cfg, task, patch, folder / "grading")
        if grade_eligible
        else False
    )
    return {
        "resolved": resolved,
        "status": "RESOLVED" if resolved else "TASK_FAILED",
        "stop_reason": stop_reason,
        "automatic_submission": automatic_submission,
        "patch_sent_for_grading": grade_eligible and bool(patch.strip()),
        "patch_bytes": len(patch.encode("utf-8")),
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "logprob_replay_tokens": sum(
            t["usage"].get("logprob_replay_tokens", 0) for t in traces
        ),
        "thinking_tokens": None,
        "cached_input_tokens": None,
        "tool_calls": tools_count,
        "wall_seconds": time.monotonic() - started,
    }


def _episode_job(
    run: Run,
    stage: str,
    model_id: str,
    nice: bool,
    ti: int,
    task: dict,
    repeat: int,
    folder: Path,
) -> dict:
    task_folder = folder / task["instance_id"] / f"rollout_{repeat}"
    final = task_folder / "result.json"
    if final.exists() and read(final)["status"] != "INFRA_ERROR":
        value = read(final)
        if value.get("run_identity") != run.identity:
            raise ValueError("Episode belongs to a different run identity")
        for name, checksum in value.get("evidence_checksums", {}).items():
            if not (run.root / name).is_file() or sha(run.root / name) != checksum:
                raise ValueError(f"Episode evidence corrupted: {name}")
        return value
    attempts = sorted(task_folder.glob("attempt_*"))
    result, infra_types = None, {}
    completed_attempts = []
    for old in attempts:
        p = old / "result.json"
        if not p.exists():
            write(
                p,
                {
                    "status": "INFRA_ERROR",
                    "error_type": "interrupted_episode",
                    "resolved": None,
                    "cost": "partial cost in call/tool evidence; unknown remainder",
                    "wall_seconds": None,
                },
            )
        value = read(p)
        if value.get("status") == "INFRA_ERROR":
            key = value.get("error_type", "interrupted_episode")
            infra_types[key] = infra_types.get(key, 0) + 1
        else:
            if value.get("status") not in {"RESOLVED", "TASK_FAILED"} or any(
                value.get(key) != expected
                for key, expected in {
                    "run_identity": run.identity,
                    "task_id": task["instance_id"],
                    "model_id": model_id,
                    "repeat": repeat,
                    "seed": 2026 if not nice else 42 + ti * 4 + repeat,
                }.items()
            ):
                raise ValueError(
                    "Completed attempt identity/status mismatch; preserve its evidence"
                )
            completed_attempts.append(value)
    if completed_attempts:
        if len(completed_attempts) != 1:
            raise ValueError(
                "Multiple completed attempts exist; preserve evidence and investigate"
            )
        result = completed_attempts[0]
        run.event(
            stage,
            "completed_attempt_recovered",
            task_id=task["instance_id"],
            model_id=model_id,
            attempt_id=result["attempt_id"],
        )
    for attempt in [] if result is not None else range(len(attempts), 9):
        if any(n > run.cfg["agent"]["infra_retries"] for n in infra_types.values()):
            break
        attempt_folder = task_folder / f"attempt_{attempt:02d}"
        attempt_folder.mkdir(parents=True, exist_ok=False)
        seed = 2026 if not nice else 42 + ti * 4 + repeat
        write(
            attempt_folder / "intent.json",
            {
                "task_id": task["instance_id"],
                "model_id": model_id,
                "seed": seed,
                "nice": nice,
                "run_identity": run.identity,
                "attempt": attempt,
            },
        )
        started = time.monotonic()
        try:
            if run.cfg["mode"] == "mock":
                success = (
                    int(
                        digest(
                            {
                                "model": model_id,
                                "task": task["instance_id"],
                                "repeat": repeat,
                            }
                        )[:8],
                        16,
                    )
                    % 3
                    == 0
                )
                result = {
                    "resolved": success,
                    "status": "RESOLVED" if success else "TASK_FAILED",
                    "stop_reason": "SYNTHETIC_FIXTURE",
                    "input_tokens": None,
                    "output_tokens": None,
                    "thinking_tokens": None,
                    "cached_input_tokens": None,
                    "tool_calls": None,
                    "wall_seconds": time.monotonic() - started,
                }
                write(attempt_folder / "trace.json", [])
                (attempt_folder / "patch.diff").write_text(
                    "synthetic_fixture\n", encoding="utf-8"
                )
            else:
                if task.get("evaluation_backend") == "harbor":
                    from nice_validation.transfer import native_episode

                    result = native_episode(run, task, model_id, attempt_folder)
                else:
                    result = real_episode(
                        run, task, model_id, nice, seed, attempt_folder
                    )
        except (InfraError, FileNotFoundError, subprocess.SubprocessError) as exc:
            error_type = str(exc).split(":", 1)[0]
            calls = [read(p) for p in attempt_folder.glob("call_*.response.json")]
            result = {
                "resolved": None,
                "status": "INFRA_ERROR",
                "stop_reason": str(exc),
                "error_type": error_type,
                "input_tokens": (
                    sum(c["usage"]["input_tokens"] for c in calls) if calls else None
                ),
                "output_tokens": (
                    sum(c["usage"]["output_tokens"] for c in calls) if calls else None
                ),
                "wall_seconds": time.monotonic() - started,
                "tool_calls": len(list(attempt_folder.glob("call_*.tool_*.json"))),
            }
            infra_types[error_type] = infra_types.get(error_type, 0) + 1
        result.update(
            {
                "task_id": task["instance_id"],
                "model_id": model_id,
                "subset_id": None if model_id == "initial" else model_id,
                "seed": seed,
                "repeat": repeat,
                "attempt_id": attempt,
                "mode": run.cfg["mode"],
                "run_identity": run.identity,
                "evidence_path": str(attempt_folder.relative_to(run.root)).replace(
                    "\\", "/"
                ),
            }
        )
        write(attempt_folder / "result.json", result)
        run.event(stage, "episode_attempt", **result)
        if result["status"] != "INFRA_ERROR":
            break
    if result is None:
        result = read(final) if final.exists() else read(attempts[-1] / "result.json")
    result["evidence_checksums"] = {
        str(p.relative_to(run.root)).replace("\\", "/"): sha(p)
        for p in sorted(task_folder.rglob("*"))
        if p.is_file() and p != final
    }
    write(final, result)
    return result


def episodes(run: Run, model_id: str, nice: bool) -> None:
    if not run.valid("prepare-data"):
        raise ValueError("Complete prepare-data first")
    if run.cfg["mode"] == "real":
        from nice_validation.smoke import require_real_smoke

        require_real_smoke(run)
        frozen = read(run.root / "manifests/validation_freeze.json")
        if (
            not run.valid("prepare-environments")
            or frozen["validation_sha256"]
            != sha(run.root / "manifests/validation.json")
            or frozen["environment_lock_sha256"]
            != sha(Path(run.cfg["agent"]["image_revision_lock"]))
        ):
            raise ValueError(
                "Complete prepare-environments and final validation freeze before main model contact"
            )
    if not nice and model_id != "initial" and not run.valid("train:" + model_id):
        raise ValueError("Complete the selected model training before evaluation")
    stage = "rollout-nice" if nice else "evaluate:" + model_id
    with run.stage(stage) as out:
        if out is None:
            return
        folder = run.root / ("rollouts/nice" if nice else "evaluation/" + model_id)
        folder.mkdir(parents=True, exist_ok=True)
        tasks = read(run.root / "manifests/validation.json")
        workers = run.cfg["agent"]["workers"]
        endpoints = run.cfg["agent"]["service_endpoints"]

        def worker_shard(worker_index):
            import copy

            worker_run = copy.copy(run)
            worker_run.cfg = copy.deepcopy(run.cfg)
            worker_run.cfg["agent"]["service_url"] = endpoints[worker_index]
            completed = []
            for ti, task in enumerate(tasks):
                if ti % workers != worker_index:
                    continue
                for repeat in range(run.cfg["experiment"]["rollouts"] if nice else 1):
                    completed.append(
                        _episode_job(
                            worker_run, stage, model_id, nice, ti, task, repeat, folder
                        )
                    )
            return completed

        if workers == 1:
            results = worker_shard(0)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = [
                    result
                    for shard in pool.map(worker_shard, range(workers))
                    for result in shard
                ]
        results.sort(key=lambda r: (r["task_id"], r["repeat"]))
        destination = folder / "results.json"
        write(destination, results)
        out.extend(
            [
                destination,
                *[
                    p
                    for p in sorted(folder.rglob("*"))
                    if p.is_file() and p != destination
                ],
            ]
        )
        if any(r["status"] == "INFRA_ERROR" for r in results):
            raise InfraError(
                f"{stage}: incomplete task results; inspect retained attempts before resume"
            )
