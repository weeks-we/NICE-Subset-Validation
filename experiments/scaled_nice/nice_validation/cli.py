"""Unified CLI. Main and explicit mock profiles share the research design."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

from nice_validation.core import Run, config, doctor, read, utc, write


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nice",
        description="Single initial checkpoint NICE-style subset validation",
    )
    p.add_argument("--config", default="configs/main.yaml")
    p.add_argument("--output", help="Override output before freezing the configuration")
    commands = p.add_subparsers(dest="command", required=True)
    online = commands.add_parser("prepare-online")
    online.add_argument("--download-model", action="store_true")
    online.add_argument("--mode", choices=["stream", "download"], default="download")
    online.add_argument("--recover-lock", action="store_true")
    online_run = commands.add_parser("run-online")
    online_run.add_argument("--recover-lock", action="store_true")
    for name in [
        "doctor",
        "prepare-data",
        "prepare-environments",
        "rollout-nice",
        "score",
        "make-subsets",
        "audit",
        "status",
        "train-parallel",
        "fetch-model",
        "freeze-model",
        "init-adapter",
    ]:
        commands.add_parser(name)
    fetch = commands.add_parser("fetch-data")
    check = commands.add_parser("preflight")
    check.add_argument("--prepared", action="store_true")
    check.add_argument("--network", action="store_true")
    cleanup = commands.add_parser("cleanup-containers")
    cleanup.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag only list this frozen run's containers",
    )
    fetch.add_argument("--mode", choices=["stream", "download"], default="stream")
    fetch.add_argument("--limit", type=int)
    fetch.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag only preview matching source shards",
    )
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--task-file")
    train = commands.add_parser("train")
    train.add_argument("--subset", required=True)
    replace = commands.add_parser("replace-validation")
    replace.add_argument("--task", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    external = commands.add_parser("evaluate-transfer")
    external.add_argument("--model", required=True)
    commands.add_parser("prepare-transfer")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--allow-mock", action="store_true")
    analyze.add_argument(
        "--refresh",
        action="store_true",
        help="Archive previous analysis before regenerating after progress",
    )
    resume = commands.add_parser("resume")
    resume.add_argument("--recover-lock", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--profile", choices=["main", "mock"], required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return p


def recover_lock(run: Run) -> None:
    path = run.root / ".run.lock"
    if not path.exists():
        return
    value = read(path)
    if value["host"] != platform.node():
        raise ValueError(
            "Lock belongs to another host; confirm its process has stopped before manual recovery"
        )
    import psutil

    if psutil.pid_exists(value["pid"]):
        raise ValueError("Lock owner PID is still alive; refusing concurrent writers")
    run.event("resume", "stale_lock_recovered", previous_owner=value)
    path.unlink()


def refresh_analysis(run: Run) -> None:
    state = run.state()
    if not any(name in state for name in ["analyze", "audit"]):
        return
    destination = run.root / "history" / utc().replace(":", "-")
    destination.mkdir(parents=True, exist_ok=False)
    for name in [
        "tables",
        "figures",
        "summary.json",
        "report.md",
        "audit.json",
        "audit.md",
    ]:
        source = run.root / name
        if source.is_dir():
            shutil.copytree(source, destination / name)
        elif source.is_file():
            shutil.copy2(source, destination / name)
    write(destination / "previous_run_state.json", state)
    for name in ["analyze", "audit"]:
        state.pop(name, None)
    write(run.state_path, state)
    run.event("analyze", "previous_analysis_archived", path=str(destination))


def pipeline(run: Run) -> None:
    if run.cfg["mode"] == "real" and run.cfg.get("schedule") == "phased-independent":
        raise ValueError("Phased GPU ownership requires scripts/run_scaled_server.py")
    from nice_validation.analysis import analyze, audit
    from nice_validation.data import prepare
    from nice_validation.harness import episodes
    from nice_validation.scoring import score
    from nice_validation.smoke import smoke
    from nice_validation.subsets import make_subsets
    from nice_validation.training import train

    prepare(run)
    if run.cfg["mode"] == "mock":
        smoke(run)
    else:
        from nice_validation.smoke import require_real_smoke

        require_real_smoke(run)
    episodes(run, "initial", nice=True)
    score(run)
    make_subsets(run)
    episodes(run, "initial", nice=False)
    subsets = read(run.root / "manifests/subsets.json")
    if run.cfg.get("schedule") == "phased-independent" and run.cfg.get("resources"):
        from nice_validation.parallel_training import train_parallel

        train_parallel(run)
        for subset in subsets:
            episodes(run, subset["subset_id"], nice=False)
    else:
        for subset in subsets:
            train(run, subset["subset_id"])
            episodes(run, subset["subset_id"], nice=False)
    if run.cfg.get("transfer", {}).get("enabled"):
        from nice_validation.transfer import evaluate_transfer
        from nice_validation.transfer_analysis import analyze_transfer

        for model in ["initial"] + [
            s["subset_id"] for s in read(run.root / "manifests/subsets.json")
        ]:
            evaluate_transfer(run, model)
        analyze_transfer(run)
    refresh_analysis(run)
    analyze(run, allow_mock=run.cfg["mode"] == "mock")
    outcome = audit(run)
    if outcome["status"] == "FAIL":
        raise ValueError("Audit failed; inspect audit.json and retained evidence")


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        cfg = config(args.config)
        if args.output:
            cfg["output"] = args.output
        command = args.command
        if command in {"prepare-online", "run-online"}:
            from nice_validation.online import prepare_online, run_online

            if command == "prepare-online":
                prepare_online(cfg, args.download_model, args.mode, args.recover_lock)
            else:
                run_online(cfg, args.recover_lock)
            print(f"{command}: completed; output={cfg['output']}")
            return
        if command == "cleanup-containers":
            from nice_validation.cleanup import cleanup_containers

            print(
                json.dumps(
                    cleanup_containers(Path(cfg["output"]), args.execute),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if command == "preflight":
            from nice_validation.preflight import preflight

            result = preflight(cfg, args.prepared, args.network)
            write(
                Path(cfg["output"])
                / (
                    "preflight-prepared.json"
                    if args.prepared
                    else "preflight-bootstrap.json"
                ),
                result,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["status"] == "FAIL":
                raise ValueError(
                    "Preflight failed; inspect the listed missing prerequisites"
                )
            return
        if command == "doctor":
            result = doctor(cfg)
            path = Path(cfg["output"]) / "doctor.json"
            write(path, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if command == "fetch-data":
            from nice_validation.data import fetch_data

            if args.limit is not None and args.limit < cfg["experiment"]["candidates"]:
                raise ValueError(
                    f"Use a limit >={cfg['experiment']['candidates']}; cleaning may require more raw records"
                )
            result = fetch_data(cfg, args.mode, args.limit, False)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.execute:
                print("Downloading only the source files listed above.", flush=True)
                print(
                    json.dumps(
                        fetch_data(cfg, args.mode, args.limit, True),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            return
        if command in {"fetch-model", "freeze-model", "init-adapter"}:
            from nice_validation.model import (
                fetch_model,
                freeze_model,
                initialize_adapter,
            )

            function = {
                "fetch-model": fetch_model,
                "freeze-model": freeze_model,
                "init-adapter": initialize_adapter,
            }[command]
            print(json.dumps(function(cfg), ensure_ascii=False, indent=2))
            return
        if command == "serve":
            from nice_validation.service import serve

            serve(cfg, args.host, args.port)
            return
        run = Run(cfg)
        if command == "prepare-transfer":
            from nice_validation.transfer_sources import prepare_transfer

            prepare_transfer(run)
        elif command == "evaluate-transfer":
            from nice_validation.transfer import evaluate_transfer
            from nice_validation.transfer_analysis import analyze_transfer

            evaluate_transfer(run, args.model)
            analyze_transfer(run)
        elif command == "prepare-data":
            from nice_validation.data import prepare

            prepare(run)
        elif command == "prepare-environments":
            from nice_validation.environments import prepare_environments

            prepare_environments(run)
        elif command == "replace-validation":
            from nice_validation.environments import replace_validation

            print(
                json.dumps(
                    replace_validation(run, args.task), ensure_ascii=False, indent=2
                )
            )
        elif command == "smoke":
            from nice_validation.smoke import smoke

            smoke(run, args.task_file)
        elif command in {"rollout-nice", "evaluate"}:
            from nice_validation.harness import episodes

            episodes(
                run,
                "initial" if command == "rollout-nice" else args.model,
                nice=command == "rollout-nice",
            )
        elif command == "score":
            from nice_validation.scoring import score

            score(run)
        elif command == "make-subsets":
            from nice_validation.subsets import make_subsets

            make_subsets(run)
        elif command == "train-parallel":
            from nice_validation.parallel_training import train_parallel

            train_parallel(run)
        elif command == "train":
            from nice_validation.training import train

            train(run, args.subset)
        elif command == "analyze":
            from nice_validation.analysis import analyze

            if args.refresh:
                refresh_analysis(run)
            analyze(run, args.allow_mock)
        elif command == "audit":
            from nice_validation.analysis import audit

            outcome = audit(run)
            print(json.dumps(outcome, ensure_ascii=False, indent=2))
            if outcome["status"] == "FAIL":
                raise ValueError("Audit failed")
        elif command == "status":
            print(
                json.dumps(
                    {
                        "run_id": run.root.name,
                        "mode": cfg["mode"],
                        "stages": run.state(),
                        "cost_log": str(run.root / "logs/events.jsonl"),
                        "cost_estimate": str(run.root / "cost_estimate.json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif command in {"run", "resume"}:
            if command == "run" and (
                (args.profile == "mock") != (cfg["mode"] == "mock")
            ):
                raise ValueError("Profile and explicit config mode disagree")
            if command == "resume" and args.recover_lock:
                recover_lock(run)
            pipeline(run)
        print(f"{command}: completed; output={run.root}")
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
