"""Fail early on official harness API drift, before downloading target weights."""

import inspect


SWEBENCH_REF = "v4.1.0"


def model_loader_api(cfg: dict, root) -> dict:
    """Resolve the actual config-to-class mapping without allocating model weights."""
    import transformers

    name = cfg["model"].get("loader")
    if not isinstance(name, str) or not name.startswith("AutoModelFor"):
        raise ValueError("Set an explicit official AutoModelFor* loader")
    loader = getattr(transformers, name, None)
    if loader is None or not hasattr(loader, "_model_mapping"):
        raise ValueError(f"Official model loader is unavailable: {name}")
    architecture = transformers.AutoConfig.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    try:
        model_class = loader._model_mapping[type(architecture)]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"{name} does not support actual configuration {type(architecture).__name__}"
        ) from exc
    declared = architecture.architectures or []
    if declared and model_class.__name__ not in declared:
        raise ValueError("Model loader and declared checkpoint architecture differ")
    return {
        "loader": name,
        "config_class": type(architecture).__name__,
        "model_class": model_class.__name__,
        "transformers": transformers.__version__,
        "weights_loaded": False,
    }


def harness_api(cfg: dict) -> dict:
    from swebench.harness import constants
    from swebench.harness.docker_build import close_logger
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swesmith.harness.utils import run_patch_in_container
    from swesmith.harness.grading import get_eval_report

    required = [
        "DOCKER_PATCH",
        "DOCKER_USER",
        "DOCKER_WORKDIR",
        "KEY_INSTANCE_ID",
        "KEY_PREDICTION",
        "UTF8",
        "LOG_REPORT",
        "LOG_TEST_OUTPUT",
        "LOG_INSTANCE",
        "RUN_EVALUATION_LOG_DIR",
    ]
    if any(not hasattr(constants, name) for name in required):
        raise ValueError(
            "SWE-bench/SWE-smith constant API mismatch; install the pinned compatible pair"
        )
    for function, fields in [
        (make_test_spec, {"namespace"}),
        (run_patch_in_container, {"commit", "patch"}),
        (get_eval_report, {"f2p_only"}),
    ]:
        if not fields <= set(inspect.signature(function).parameters):
            raise ValueError(f"Official harness API mismatch: {function.__name__}")
    if not callable(close_logger):
        raise ValueError("Official logger close API is missing")
    result = {"swebench_ref": SWEBENCH_REF, "swesmith_api": "PASS"}
    if cfg.get("transfer", {}).get("enabled"):
        from harbor.job import Job
        from harbor.models.job.config import JobConfig
        from harbor.environments.docker.docker import (
            _sanitize_docker_compose_project_name,
        )

        if not inspect.iscoroutinefunction(Job.create) or not callable(
            JobConfig.model_validate
        ):
            raise ValueError("Harbor asynchronous Job.create/JobConfig API mismatch")
        if not callable(_sanitize_docker_compose_project_name):
            raise ValueError("Harbor Docker image-proof API is missing")
        result["harbor_api"] = "PASS"
    return result
