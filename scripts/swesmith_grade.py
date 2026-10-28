"""Official SWE-smith utilities with an explicitly frozen checkout SHA."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--timeout", required=True, type=int)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.commit):
        raise ValueError("Grading requires a frozen 40-hex checkout SHA")
    from swebench.harness.constants import (
        RUN_EVALUATION_LOG_DIR,
        LOG_REPORT,
        LOG_TEST_OUTPUT,
        LOG_INSTANCE,
        APPLY_PATCH_FAIL,
    )
    from swebench.harness.docker_build import close_logger
    from swesmith.harness.utils import run_patch_in_container
    from swesmith.harness.grading import get_eval_report

    task = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))[0]
    prediction = json.loads(
        Path(args.predictions_path).read_text(encoding="utf-8").splitlines()[0]
    )
    if task["instance_id"] != prediction["instance_id"]:
        raise ValueError("Prediction belongs to another task")
    outcome = run_patch_in_container(
        task,
        args.run_id,
        RUN_EVALUATION_LOG_DIR,
        args.timeout,
        patch=prediction["model_patch"],
        commit=args.commit,
    )
    if outcome is None:
        raise RuntimeError("Official SWE-smith container execution failed")
    logger, timed_out = outcome
    folder = RUN_EVALUATION_LOG_DIR / args.run_id / task["instance_id"]
    try:
        if timed_out:
            report = {"resolved": False, "timed_out": True}
        else:
            output = folder / LOG_TEST_OUTPUT
            if not output.is_file():
                log = folder / LOG_INSTANCE
                if log.is_file() and APPLY_PATCH_FAIL in log.read_text(
                    encoding="utf-8"
                ):
                    report = {
                        "resolved": False,
                        "reason": "official_patch_application_failed",
                    }
                else:
                    raise RuntimeError("Official SWE-smith test output missing")
            else:
                report = get_eval_report(prediction, task, output, f2p_only=False)
            if not isinstance(report.get("resolved"), bool):
                raise RuntimeError("Official SWE-smith verdict missing")
        (folder / LOG_REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        close_logger(logger)


if __name__ == "__main__":
    main()
