"""Native Harbor tools driven by the same frozen local LoRA service."""

import asyncio
import json
from pathlib import Path
import shlex
import time

from nice_validation.core import config, digest, write
from nice_validation.harness import (
    EDITOR_PROGRAM,
    InfraError,
    TaskFailure,
    request_generation,
)
from nice_validation.model import tree_identity
from nice_validation.protocol import arguments


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
) -> None:
    logs.mkdir(parents=True, exist_ok=True)
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
            diff = await environment.exec(
                command="cd /app 2>/dev/null || cd /testbed; git add -N .; git diff --binary HEAD",
                timeout_sec=60,
            )
            if diff.return_code:
                raise InfraError("pro_patch_capture_failed")
            (logs / "model.patch").write_text(diff.stdout or "", encoding="utf-8")


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

    async def run(self, instruction, environment, context):
        await service_loop(
            environment,
            self.cfg,
            self.adapter,
            self.logs_dir,
            instruction,
            context,
            self.capture_patch,
        )


class NicePatchReplay(BaseAgent):
    def __init__(
        self, *args, patch_path: str, patch_sha256: str, expected_images: dict, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.patch_path, self.patch_sha = Path(patch_path), patch_sha256
        self.expected_images = expected_images

    @staticmethod
    def name():
        return "nice-pristine-pro-regrade"

    def version(self):
        return "1.0.0"

    async def setup(self, environment):
        if environment_images(environment) != self.expected_images:
            raise InfraError("native_regrade_image_changed")

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
            command="cd /app 2>/dev/null || cd /testbed; git apply --binary /tmp/nice-model.patch",
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
