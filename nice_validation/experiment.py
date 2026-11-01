"""Evaluation namespaces, validation-only checkpoint choice and full mock flow."""
import copy
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from nice_validation.core import read, write, sha
from nice_validation.experiment_training import ensure_jobs, validate_point


class EndpointPool:
    """Expand from validation GPUs to all GPUs without changing an active episode."""
    def __init__(self, endpoints, capacity=None, slots_per_endpoint=1):
        self.queue = queue.Queue()
        self.endpoints = set()
        self.capacity = capacity or len(endpoints)
        self.error = None
        self.lock = threading.Lock()
        self.slots_per_endpoint = slots_per_endpoint
        self.add(endpoints)

    def add(self, endpoints):
        with self.lock:
            for endpoint in endpoints:
                if endpoint not in self.endpoints:
                    self.endpoints.add(endpoint)
                    for _ in range(self.slots_per_endpoint):
                        self.queue.put(endpoint)

    @contextmanager
    def acquire(self):
        while True:
            if self.error is not None:
                raise RuntimeError("Training/service activation failed") from self.error
            try:
                endpoint = self.queue.get(timeout=.2)
                break
            except queue.Empty:
                continue
        try:
            yield endpoint
        finally:
            self.queue.put(endpoint)


def evaluate(run, split, model_id, endpoint_pool=None):
    from nice_validation.harness import _episode_job, InfraError
    if split not in {"selection", "benchmark", "nice_retest"} or not run.valid("prepare-data"):
        raise ValueError("Unknown/unprepared evaluation split")
    if run.cfg["mode"] == "real":
        from nice_validation.smoke import require_real_smoke
        require_real_smoke(run)
        frozen = read(run.root / "manifests/validation_freeze.json")
        if not run.valid("prepare-environments") or frozen["validation_sha256"] != sha(run.root / "manifests/validation.json") or frozen["environment_lock_sha256"] != sha(Path(run.cfg['agent']['image_revision_lock'])):
            raise ValueError("Complete immutable environment preparation first")
    if split in {"benchmark", "nice_retest"}:
        if not run.valid("model-selection"):
            raise ValueError("Freeze validation-only checkpoint choices before benchmark")
        allowed = {"initial", *[p["model_id"] for p in read(run.root / "selection.json")["selected"]]}
        if model_id not in allowed:
            raise ValueError("Final evaluation is restricted to initial and the four selected models")
    if model_id != "initial":
        matches = [p for p in read(run.root / "manifests/checkpoints.json") if p["model_id"] == model_id]
        if len(matches) != 1:
            raise ValueError("Unknown checkpoint model")
        validate_point(run, matches[0])
    manifest = {"selection": "sft_validation", "benchmark": "benchmark", "nice_retest": "validation"}[split]
    tasks = read(run.root / "manifests" / (manifest + ".json"))
    expected = run.cfg["experiment"][manifest]
    if len(tasks) != expected or len({r["instance_id"] for r in tasks}) != expected:
        raise ValueError("Evaluation task count/identity mismatch")
    stage = split + ":" + model_id
    with run.stage(stage) as artifacts:
        if artifacts is None:
            return
        folder = run.root / "evaluation" / split / model_id
        folder.mkdir(parents=True, exist_ok=True)
        pool = endpoint_pool or EndpointPool(list(dict.fromkeys(run.cfg["agent"]["service_endpoints"])),
                  capacity=run.cfg["agent"]["workers"], slots_per_endpoint=run.cfg["agent"].get("tasks_per_gpu", 1))

        def task_job(item):
            index, task = item
            with pool.acquire() as endpoint:
                view = copy.copy(run)
                view.cfg = copy.deepcopy(run.cfg)
                view.cfg["agent"]["service_url"] = endpoint
                view.cfg["_evaluation_scope"] = stage
                return _episode_job(view, stage, model_id, False, index, task, 0, folder)

        with ThreadPoolExecutor(max_workers=pool.capacity) as executor:
            results = list(executor.map(task_job, enumerate(tasks)))
        results.sort(key=lambda r: r["task_id"])
        destination = folder / "results.json"
        write(destination, results)
        artifacts.extend(p for p in folder.rglob("*") if p.is_file())
        if any(r["status"] == "INFRA_ERROR" for r in results):
            raise InfraError("Incomplete " + stage)


def rate(records, expected):
    if len(records) != expected or len({r["task_id"] for r in records}) != expected or any(
        r["status"] not in {"RESOLVED", "TASK_FAILED"} or type(r.get("resolved")) is not bool
        or r["status"] != ("RESOLVED" if r["resolved"] else "TASK_FAILED") for r in records
    ):
        raise ValueError("Incomplete/invalid task results cannot select or rank models")
    return sum(r["resolved"] for r in records)


def select_models(run):
    entries = read(run.root / "manifests/training_jobs.json")
    points = read(run.root / "manifests/checkpoints.json")
    if any(not run.valid("train:" + j["subset_id"]) for j in entries):
        raise ValueError("Complete all four training branches before selecting")
    if any(not run.valid("selection:" + p["model_id"]) for p in points):
        raise ValueError("Evaluate all twelve checkpoints on the same selection tasks")
    with run.stage("model-selection") as artifacts:
        if artifacts is None:
            return
        evaluated, selected = [], []
        for point in points:
            records = read(run.root / "evaluation/selection" / point["model_id"] / "results.json")
            expected_ids = {r["instance_id"] for r in read(run.root / "manifests/sft_validation.json")}
            if {r["task_id"] for r in records} != expected_ids or any(r["model_id"] != point["model_id"] or r["run_identity"] != run.identity for r in records):
                raise ValueError("Selection evaluation belongs to another model/task split/run")
            evaluated.append({**point, "resolved_count": rate(records, run.cfg["experiment"]["sft_validation"])})
        for job in entries:
            options = [p for p in evaluated if p["job_id"] == job["subset_id"]]
            # Same predeclared rule within every (group, learning-rate) branch.
            selected.append(min(options, key=lambda p: (-p["resolved_count"], p["exposures"])))
        result = {"mode": run.cfg["mode"], "run_identity": run.identity,
                  "rule": "maximize SWE-smith validation resolved count; ties prefer earlier training exposure",
                  "selection_tasks_sha256": sha(run.root / "manifests/sft_validation.json"),
                  "benchmark_used_for_selection": False, "evaluated": evaluated, "selected": selected}
        write(run.root / "selection.json", result)
        artifacts.append(run.root / "selection.json")


def pipeline(run):
    if run.cfg["mode"] != "mock":
        raise ValueError("Real execution requires the managed scripts/run_server.py entry")
    from nice_validation.data import prepare
    from nice_validation.smoke import smoke
    from nice_validation.harness import episodes
    from nice_validation.scoring import score
    from nice_validation.subsets import make_subsets
    from nice_validation.batch_tuning import tune
    from nice_validation.training import train
    from nice_validation.experiment_report import report
    prepare(run)
    smoke(run)
    episodes(run, "initial", nice=True)
    score(run)
    make_subsets(run)
    ensure_jobs(run)
    tune(run)
    evaluate(run, "selection", "initial")
    for job in read(run.root / "manifests/training_jobs.json"):
        train(run, job["subset_id"])
    for point in read(run.root / "manifests/checkpoints.json"):
        evaluate(run, "selection", point["model_id"])
    select_models(run)
    for model_id in ["initial", *[p["model_id"] for p in read(run.root / "selection.json")["selected"]]]:
        evaluate(run, "benchmark", model_id)
        evaluate(run, "nice_retest", model_id)
    report(run)
