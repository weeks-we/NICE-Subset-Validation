"""Native Harbor tools driven by the same frozen local LoRA service."""

import asyncio
import json
from pathlib import Path
import shlex
import time
import uuid

from nice_validation.core import config, digest, require_commit, write
from nice_validation.harness import (
    EDITOR_PROGRAM,
    InfraError,
    TaskFailure,
    request_generation,
    patch_command,
)
from nice_validation.model import tree_identity
from nice_validation.protocol import arguments


async def pro_checkout(environment) -> dict:
    probe = await environment.exec(
        command=(
            "for d in /app /testbed; do "
            'if git -C "$d" rev-parse --is-inside-work-tree >/dev/null 2>&1; then '
            'git -C "$d" rev-parse --show-toplevel && '
            'git -C "$d" rev-parse HEAD && '
            'git -C "$d" status --porcelain --untracked-files=all; exit $?; fi; '
            "done; exit 1"
        ),
        timeout_sec=60,
    )
    lines = (probe.stdout or "").splitlines()
    if probe.return_code or len(lines) != 2 or not lines[0].startswith("/"):
        raise InfraError("pro_starting_checkout_missing_or_dirty")
    try:
        require_commit(lines[1], "Pro starting commit")
    except ValueError as exc:
        raise InfraError("pro_starting_commit_invalid") from exc
    return {"root": lines[0], "commit": lines[1]}


def environment_images(environment) -> dict:
    import docker
    from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

    project = _sanitize_docker_compose_project_name(environment.session_id)
    client = docker.from_env()
    try:
        containers = client.containers.list(
            filters={"label": "com.docker.compose.project=" + project}
        )
        proof = {
            container.attrs["Config"]["Labels"][
                "com.docker.compose.service"
            ]: container.image.id
            for container in containers
        }
        if not proof:
            raise InfraError("native_environment_image_proof_missing")
        return proof
    finally:
        client.close()


async def service_loop(
    environment,
    cfg: dict,
    adapter: Path,
    logs: Path,
    instruction: str,
    context,
    capture_patch: bool = False,
    patch_base: dict | None = None,
) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    if capture_patch:
        patch_base = patch_base or await pro_checkout(environment)
        write(logs / "patch_base.json", patch_base)
    messages = [
        {
            "role": "system",
            "content": "Complete the task using the declared tools. Shell invocations are independent; use absolute paths or explicit cd. Call submit when finished.",
        },
        {"role": "user", "content": instruction},
    ]
    budget = cfg["agent"]
    start, output_tokens, input_tokens = time.monotonic(), 0, 0
    stop = "ACTION_LIMIT"
    adapter_sha = tree_identity(adapter)
    try:
        for index in range(budget["max_actions"]):
            remaining = budget["wall_seconds"] - (time.monotonic() - start)
            if remaining <= 0:
                raise TaskFailure("WALL_LIMIT")
            allowance = min(
                budget["max_new_tokens"], budget["total_new_tokens"] - output_tokens
            )
            if allowance <= 0:
                raise TaskFailure("TOKEN_LIMIT")
            payload = {
                "messages": messages,
                "nice": False,
                "seed": int(digest([2026, index])[:8], 16),
                "max_tokens": allowance,
                "adapter_path": str(adapter.resolve()),
                "adapter_sha256": adapter_sha,
            }
            write(logs / f"call_{index:03d}.request.json", payload)
            response = await asyncio.to_thread(
                request_generation,
                budget["service_url"],
                payload,
                remaining,
                start + budget["wall_seconds"],
            )
            write(logs / f"call_{index:03d}.response.json", response)
            input_tokens += response["usage"]["input_tokens"]
            output_tokens += response["usage"]["output_tokens"]
            context.n_input_tokens, context.n_output_tokens = (
                input_tokens,
                output_tokens,
            )
            if response.get("parse_error"):
                raise TaskFailure(response["parse_error"])
            messages.append(response["message"])
            calls = response["message"].get("tool_calls", [])
            if not calls:
                stop = "FINAL_OUTPUT"
                break
            submitted = False
            for ci, call in enumerate(calls):
                if time.monotonic() >= start + budget["wall_seconds"]:
                    raise TaskFailure("WALL_LIMIT")
                try:
                    name, args = arguments(call)
                except ValueError as exc:
                    raise TaskFailure("INVALID_TOOL_CALL:" + str(exc)) from exc
                if name == "submit":
                    stop, submitted = "SUBMITTED", True
                    break
                command = (
                    args["command"]
                    if name == "bash"
                    else (
                        "printf '%s' "
                        + shlex.quote(json.dumps(args))
                        + " | python3 -c "
                        + shlex.quote(EDITOR_PROGRAM)
                    )
                )
                remaining = budget["wall_seconds"] - (time.monotonic() - start)
                if remaining <= 0:
                    raise TaskFailure("WALL_LIMIT")
                seconds = max(1, int(min(remaining, budget["tool_seconds"])))
                executed = f"timeout --signal=KILL {seconds}s bash -lc " + shlex.quote(
                    command
                )
                try:
                    value = await asyncio.wait_for(
                        environment.exec(
                            command=executed,
                            timeout_sec=seconds + 2,
                        ),
                        timeout=seconds + 3,
                    )
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    raise TaskFailure("TOOL_TIMEOUT") from exc
                result = (value.stdout or "") + (value.stderr or "")
                if value.return_code:
                    result += f"\n[exit_code={value.return_code}]"
                write(
                    logs / f"call_{index:03d}.tool_{ci:02d}.json",
                    {"call": call, "executed_command": executed, "result": result},
                )
                if value.return_code in {124, 137}:
                    raise TaskFailure("TOOL_TIMEOUT")
                observed = (
                    result
                    if len(result) <= 20000
                    else result[:10000]
                    + "\n[tool-output-policy-v1: middle omitted; full output saved]\n"
                    + result[-10000:]
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": observed}
                )
            if submitted:
                break
    except TaskFailure as exc:
        stop = str(exc)
    finally:
        failed = stop not in {"SUBMITTED", "FINAL_OUTPUT"}
        write(
            logs / "episode.json",
            {
                "stop_reason": stop,
                "task_failure": failed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "wall_seconds": time.monotonic() - start,
            },
        )
        context.metadata = {"stop_reason": stop, "task_failure": failed}
        if capture_patch:
            remote_patch = "/tmp/nice-model-" + uuid.uuid4().hex + ".patch"
            diff = await environment.exec(
                command=patch_command(
                    patch_base["root"], patch_base["commit"], remote_patch
                ),
                timeout_sec=60,
            )
            if diff.return_code:
                write(
                    logs / "patch_capture_failure.json",
                    {"return_code": diff.return_code, "stderr": diff.stderr},
                )
                write(
                    logs / "episode.json",
                    {
                        "stop_reason": "PATCH_COLLECTION_FAILED",
                        "task_failure": True,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "wall_seconds": time.monotonic() - start,
                    },
                )
                context.metadata = {
                    "stop_reason": "PATCH_COLLECTION_FAILED",
                    "task_failure": True,
                }
                (logs / "model.patch").write_bytes(b"")
            else:
                await environment.download_file(
                    source_path=remote_patch, target_path=logs / "model.patch"
                )


# Import the optional harness only when Harbor instantiates the native agent.
from harbor.agents.base import BaseAgent


class NiceAgent(BaseAgent):
    def __init__(
        self,
        *args,
        config_file: str,
        adapter_path: str,
        capture_patch: bool = False,
        expected_images: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cfg, self.adapter = config(config_file), Path(adapter_path)
        self.capture_patch = capture_patch
        self.expected_images = expected_images

    @staticmethod
    def name():
        return "nice-local-lora"

    def version(self):
        return "1.0.0"

    async def setup(self, environment):
        proof = environment_images(environment)
        if self.expected_images is not None and proof != self.expected_images:
            raise InfraError("native_environment_image_changed")
        write(self.logs_dir / "environment_images.json", proof)
        self.patch_base = (
            await pro_checkout(environment) if self.capture_patch else None
        )
        if self.patch_base:
            write(self.logs_dir / "patch_base.json", self.patch_base)

    async def run(self, instruction, environment, context):
        await service_loop(
            environment,
            self.cfg,
            self.adapter,
            self.logs_dir,
            instruction,
            context,
            self.capture_patch,
            self.patch_base,
        )


class NicePatchReplay(BaseAgent):
    def __init__(
        self,
        *args,
        patch_path: str,
        patch_sha256: str,
        expected_images: dict,
        expected_checkout: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_path, self.patch_sha = Path(patch_path), patch_sha256
        self.expected_images = expected_images
        self.expected_checkout = expected_checkout

    @staticmethod
    def name():
        return "nice-pristine-pro-regrade"

    def version(self):
        return "1.0.0"

    async def setup(self, environment):
        if environment_images(environment) != self.expected_images:
            raise InfraError("native_regrade_image_changed")
        if (
            not self.expected_checkout
            or await pro_checkout(environment) != self.expected_checkout
        ):
            raise InfraError("pro_regrade_starting_checkout_changed")

    async def run(self, instruction, environment, context):
        from nice_validation.core import sha

        if sha(self.patch_path) != self.patch_sha:
            raise InfraError("pro_captured_patch_changed")
        if not self.patch_path.stat().st_size:
            return
        await environment.upload_file(
            source_path=self.patch_path, target_path="/tmp/nice-model.patch"
        )
        result = await environment.exec(
            command="git -C "
            + shlex.quote(self.expected_checkout["root"])
            + " apply --binary /tmp/nice-model.patch",
            timeout_sec=120,
        )
        write(
            self.logs_dir / "patch_application.json",
            {
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if result.return_code:
            context.metadata = {"patch_application_failed": True}
