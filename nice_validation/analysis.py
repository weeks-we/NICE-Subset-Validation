"""Task-paired descriptive analysis, reproducible tables and publication exports."""

from __future__ import annotations

import csv
import json
import itertools
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from nice_validation.core import Run, read, read_jsonl, write


FIGURES = [
    "candidate_distribution",
    "subset_distributions",
    "score_gain",
    "group_gains",
    "cost_gain",
    "overlap",
    "repository_composition",
]
TABLES = [
    "candidate_scores",
    "subset_summary",
    "task_results",
    "group_summary",
    "paired_tasks",
    "stage_costs",
    "failure_summary",
]
COLORS = {"low": "#0072B2", "middle": "#E69F00", "high": "#009E73", "random": "#CC79A7"}
MARKERS = {"low": "o", "middle": "s", "high": "^", "random": "D"}


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        fields or list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (dict, list))
                        else v
                    )
                    for k, v in row.items()
                }
            )


def task_rate(records: list[dict], task_ids: list[str]) -> dict:
    ids = [r["task_id"] for r in records]
    valid = [
        r
        for r in records
        if r.get("status") in {"RESOLVED", "TASK_FAILED"}
        and isinstance(r.get("resolved"), bool)
    ]
    complete = (
        bool(task_ids)
        and len(records) == len(task_ids)
        and set(ids) == set(task_ids)
        and len(ids) == len(set(ids))
        and len(valid) == len(task_ids)
    )
    return {
        "status": "completed" if complete else "incomplete",
        "resolved_count": sum(r["resolved"] for r in valid),
        "completed_tasks": len(valid),
        "denominator": len(task_ids),
        "rate": sum(r["resolved"] for r in valid) / len(task_ids) if complete else None,
    }


def ranking(x: list[float], y: list[float]) -> dict:
    from scipy.stats import rankdata

    if len(x) < 2:
        return {
            "spearman": None,
            "concordance": None,
            "comparable_pairs": 0,
            "tied_pairs": 0,
        }
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    cx, cy = rx - np.mean(rx), ry - np.mean(ry)
    denominator = float(np.sqrt(np.sum(cx * cx) * np.sum(cy * cy)))
    rho = float(np.sum(cx * cy) / denominator) if denominator else None
    comparable, tied, concordant = 0, 0, 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx == 0 or dy == 0:
            tied += 1
        else:
            comparable += 1
            concordant += int(dx == dy)
    return {
        "spearman": rho,
        "concordance": concordant / comparable if comparable else None,
        "comparable_pairs": comparable,
        "tied_pairs": tied,
        "concordant_pairs": concordant,
    }


def optional(root: Path, path: str, default):
    return read(root / path) if (root / path).is_file() else default


def evaluation_records(root: Path, folder: str) -> list[dict]:
    finals = sorted((root / folder).glob("*/rollout_*/result.json"))
    return (
        [read(p) for p in finals]
        if finals
        else optional(root, folder + "/results.json", [])
    )


def analyze(run: Run, allow_mock: bool = False) -> None:
    if run.cfg["mode"] == "mock" and not allow_mock:
        raise ValueError(
            "Mock results require --allow-mock and are never formal research evidence"
        )
    with run.stage("analyze") as out:
        if out is None:
            return
        root, mock = run.root, run.cfg["mode"] == "mock"
        candidates = optional(root, "scores/candidate_scores.json", [])
        subsets = optional(root, "manifests/subsets.json", [])
        tasks = optional(root, "manifests/validation.json", [])
        task_ids = [t["instance_id"] for t in tasks]
        initial = evaluation_records(root, "evaluation/initial")
        baseline = (
            task_rate(initial, task_ids)
            if task_ids
            else {"rate": None, "status": "incomplete"}
        )
        all_results, summary = list(initial), []
        all_train = []
        for subset in subsets:
            sid = subset["subset_id"]
            training = optional(root, f"training/{sid}/training.json", {})
            all_train.append(training)
            results = evaluation_records(root, f"evaluation/{sid}")
            all_results.extend(results)
            evaluation = task_rate(results, task_ids)
            delta = (
                evaluation["rate"] - baseline["rate"]
                if evaluation["rate"] is not None and baseline["rate"] is not None
                else None
            )
            summary.append(
                {
                    "subset_id": sid,
                    "group": subset["group"],
                    "sampling_seed": subset["sampling_seed"],
                    "member_count": len(subset["members"]),
                    "fresh_adam_cosine": subset["fresh_adam_cosine"]["mean"],
                    "raw_grad_cosine": subset["raw_grad_cosine"]["mean"],
                    "score_stats": subset["fresh_adam_cosine"],
                    "raw_score_stats": subset["raw_grad_cosine"],
                    "supervised_tokens": subset["supervised_tokens"],
                    "budget_target": subset["budget_target"],
                    "repository_counts": subset["repos"],
                    "training_status": training.get("status", "pending"),
                    "training_seconds": training.get("wall_seconds"),
                    "updates": training.get("updates"),
                    "exposures": training.get("exposures"),
                    "initial_adapter_sha256": training.get("initial_adapter_sha256"),
                    "gpu_hours": training.get("gpu_hours"),
                    "evaluation_status": evaluation["status"],
                    "resolved_count": evaluation["resolved_count"],
                    "completed_tasks": evaluation["completed_tasks"],
                    "denominator": len(task_ids),
                    "success_rate": None if mock else evaluation["rate"],
                    "delta": None if mock else delta,
                    "synthetic_demo_rate": evaluation["rate"] if mock else None,
                    "synthetic_demo_delta": delta if mock else None,
                    "plot_gain": delta,
                    "mode": run.cfg["mode"],
                }
            )
        usable = [s for s in summary if s["plot_gain"] is not None]
        if not mock and any(
            row.get("mode") != "real"
            for row in [
                *candidates,
                *subsets,
                *all_results,
                *[t for t in all_train if t],
            ]
        ):
            raise ValueError("Mock or unlabeled artifacts cannot enter formal analysis")
        correlations = {
            name: ranking([s[name] for s in usable], [s["plot_gain"] for s in usable])
            for name in ["fresh_adam_cosine", "raw_grad_cosine"]
        }
        events_path = root / "logs/events.jsonl"
        events = read_jsonl(events_path) if events_path.exists() else []
        for service_log in sorted((root / "logs").glob("service-*.jsonl")):
            events.extend(read_jsonl(service_log))
        costs = [e for e in events if e["event"] == "stage_cost"]
        failures = [
            e
            for e in events
            if e["event"] == "failed" or e.get("status") == "INFRA_ERROR"
        ]
        groups = []
        for group in COLORS:
            members = [s for s in usable if s["group"] == group]
            gains = [s["plot_gain"] for s in members]
            groups.append(
                {
                    "group": group,
                    "n": len(gains),
                    "mean_gain": float(np.mean(gains)) if gains else None,
                    "min_gain": min(gains) if gains else None,
                    "max_gain": max(gains) if gains else None,
                    "interpretation": "three overlapping sampling repeats; no independence claim",
                    "mode": run.cfg["mode"],
                }
            )
        paired = []
        for task in task_ids:
            pair = {"task_id": task}
            for model in ["initial"] + [s["subset_id"] for s in summary]:
                hits = [
                    r
                    for r in all_results
                    if r["task_id"] == task and r["model_id"] == model
                ]
                pair[model] = (
                    hits[0]["resolved"]
                    if len(hits) == 1 and hits[0]["status"] != "INFRA_ERROR"
                    else None
                )
            paired.append(pair)
        for name, rows in [
            ("candidate_scores", candidates),
            (
                "subset_summary",
                [{k: v for k, v in s.items() if k != "plot_gain"} for s in summary],
            ),
            ("task_results", all_results),
            ("group_summary", groups),
            ("paired_tasks", paired),
            ("stage_costs", costs),
            ("failure_summary", failures),
        ]:
            p = root / "tables" / f"{name}.csv"
            csv_write(p, rows)
            out.append(p)
        # Profile before selecting chart forms. Small n/group uses individual points.
        profile = {
            "candidate_count": len(candidates),
            "subset_count": len(summary),
            "group_counts": dict(Counter(s["group"] for s in summary)),
            "missing_gain_count": sum(s["plot_gain"] is None for s in summary),
            "shared_task_count": len(task_ids),
            "score_quantiles": (
                np.quantile(
                    [c["fresh_adam_cosine"] for c in candidates],
                    [0, 0.25, 0.5, 0.75, 1],
                ).tolist()
                if candidates
                else None
            ),
            "chart_rationale": "Histogram for candidate distribution; raw member points for subset distributions; scatter for score/gain; all three points per group; no KDE or significance tests.",
        }
        write(root / "figures" / "data_profile.json", profile)
        out.append(root / "figures" / "data_profile.json")
        out.extend(plots(run, candidates, subsets, summary))
        required = [
            "prepare-data",
            "rollout-nice",
            "score",
            "make-subsets",
            "evaluate:initial",
        ]
        required += [
            f"{stage}:{s['subset_id']}"
            for s in subsets
            for stage in ["train", "evaluate"]
        ]
        complete = (
            len(summary) == 12
            and len(usable) == 12
            and len(task_ids) == 30
            and all(run.valid(stage) for stage in required)
        )
        report = {
            "title": "Single initial checkpoint NICE-style scoring validation",
            "mode": run.cfg["mode"],
            "formal_evidence": not mock and complete,
            "status": (
                "synthetic_fixture"
                if mock
                else "completed" if complete else "incomplete"
            ),
            "baseline": {**baseline, "rate": None if mock else baseline["rate"]},
            "synthetic_demo_baseline_rate": baseline["rate"] if mock else None,
            "correlations": None if mock else correlations,
            "synthetic_demo_correlations": correlations if mock else None,
            "subsets": [
                {k: v for k, v in s.items() if k != "plot_gain"} for s in summary
            ],
            "main_episode_count": len(all_results)
            + len(optional(root, "rollouts/nice/results.json", [])),
            "episode_attempts": sum(e["event"] == "episode_attempt" for e in events),
            "successful_updates": sum(t.get("updates", 0) for t in all_train),
            "transfer_summary": (
                "transfer/summary.json"
                if run.cfg.get("transfer", {}).get("enabled")
                else None
            ),
            "transfer_episode_count": optional(root, "transfer/summary.json", {}).get(
                "transfer_episode_count", 0
            ),
            "limitations": [
                "SWE-smith validation is shared by scoring and same-task evaluation; held-out transfer evidence is reported separately.",
                "12 overlapping subsets and 30 shared tasks are not independent repeats.",
                "Three sampling seeds do not measure training seed stability.",
                "One resolved task changes the success rate by 3.33 percentage points.",
                "Logical tokens exclude checkpointing recomputation and are not FLOPs.",
            ],
            "conclusion": (
                "synthetic_fixture_no_research_conclusion"
                if mock
                else (
                    "descriptive_evidence_only" if complete else "insufficient_evidence"
                )
            ),
        }
        write(root / "summary.json", report)
        text = (
            "# SYNTHETIC FIXTURE — NO RESEARCH EVIDENCE\n\n"
            if mock
            else "# 单初始 checkpoint 的 NICE 型评分验证\n\n"
        )
        text += f"Status: {report['status']}. Main episodes: {report['main_episode_count']}; optimizer updates: {report['successful_updates']}.\n\n"
        if run.cfg.get("transfer", {}).get("enabled"):
            text += "Held-out transfer: see [transfer/report.md](transfer/report.md) and transfer/tables/summary.csv. Four fixed 50-task suites; never used for NICE scoring or training.\n\n"
        text += "Primary score: fresh_adam_cosine. Diagnostic: raw_grad_cosine.\n\n"
        text += "| subset | score | resolved/30 | gain |\n|---|---:|---:|---:|\n"
        for s in summary:
            gain = f"{s['plot_gain']:+.4f}" if s["plot_gain"] is not None else "null"
            text += f"| {s['subset_id']} | {s['fresh_adam_cosine']:.5f} | {s['resolved_count']}/30 ({s['evaluation_status']}) | {gain} |\n"
        text += "\n" + (
            "All rates and gains above are synthetic demo values.\n\n" if mock else ""
        )
        text += "\n".join("- " + line for line in report["limitations"])
        text += "\n\nSee tables/, figures/, audit.json and logs/events.jsonl for evidence.\n"
        (root / "report.md").write_text(text, encoding="utf-8")
        out.extend([root / "summary.json", root / "report.md"])


def plots(
    run: Run, candidates: list[dict], subsets: list[dict], summaries: list[dict]
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )
    paths, qa = [], []
    folder = run.root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    mock = run.cfg["mode"] == "mock"

    def save(fig, name):
        fig.suptitle(
            ("SYNTHETIC FIXTURE | " if mock else "") + name.replace("_", " "),
            fontsize=10,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = fig.bbox
            clipped = []
            # Matplotlib creates labels for ticks outside the visible limits;
            # these Text objects are not painted by Axis.draw().
            skipped = set()
            for axes in fig.axes:
                for axis in [axes.xaxis, axes.yaxis]:
                    lo, hi = sorted(axis.get_view_interval())
                    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                        if tick.get_loc() < lo - 1e-10 or tick.get_loc() > hi + 1e-10:
                            skipped.update([id(tick.label1), id(tick.label2)])
            for text in fig.findobj(matplotlib.text.Text):
                if id(text) not in skipped and text.get_visible() and text.get_text():
                    box = text.get_window_extent(renderer)
                    if (
                        box.width
                        and box.height
                        and (
                            box.x0 < -1
                            or box.y0 < -1
                            or box.x1 > bounds.x1 + 1
                            or box.y1 > bounds.y1 + 1
                        )
                    ):
                        clipped.append(text.get_text())
            glyph = [str(w.message) for w in caught if "Glyph" in str(w.message)]
            qa.append(
                {
                    "figure": name,
                    "missing_glyphs": glyph,
                    "clipped_text": clipped,
                    "status": "FAIL" if glyph or clipped else "PASS",
                }
            )
        for ext in ["png", "pdf"]:
            p = folder / f"{name}.{ext}"
            fig.savefig(p, dpi=300)
            paths.append(p)
        with Image.open(folder / f"{name}.png") as im:
            im.convert("L").save(folder / f"{name}.grayscale.png", dpi=(300, 300))
        paths.append(folder / f"{name}.grayscale.png")
        plt.close(fig)

    score_map = {r["data_id"]: r["fresh_adam_cosine"] for r in candidates}
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    if candidates:
        vals = sorted(score_map.values())
        ax.hist(vals, bins=25, color="#0072B2", edgecolor="white")
        for boundary in [99, 199]:
            if len(vals) > boundary:
                ax.axvline(vals[boundary], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="fresh Adam cosine", ylabel="Candidate count")
    save(fig, "candidate_distribution")

    fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
    for i, s in enumerate(subsets):
        vals = [score_map[m] for m in s["members"] if m in score_map]
        jitter = np.linspace(-0.15, 0.15, len(vals))
        ax.scatter(
            vals,
            i + jitter,
            s=10,
            c=COLORS[s["group"]],
            marker=MARKERS[s["group"]],
            alpha=0.7,
        )
    ax.set_yticks(range(len(subsets)), [s["subset_id"] for s in subsets])
    ax.set(xlabel="Member fresh Adam cosine", ylabel="Subset (30 members each)")
    save(fig, "subset_distributions")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    # Numeric labels reduce overlap; the right margin gives a complete ID key.
    label_key, annotations = [], []
    for i, s in enumerate(summaries):
        if s["plot_gain"] is None:
            continue
        ax.scatter(
            s["fresh_adam_cosine"],
            s["plot_gain"],
            color=COLORS[s["group"]],
            marker=MARKERS[s["group"]],
            s=36,
        )
        annotations.append(
            ax.annotate(
                str(i + 1),
                (s["fresh_adam_cosine"], s["plot_gain"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.5},
            )
        )
        label_key.append(f"{i+1}: {s['subset_id']}")
    ax.text(
        1.02, 0.98, "\n".join(label_key), transform=ax.transAxes, va="top", fontsize=8
    )
    ax.axhline(0, color="0.6", linewidth=0.7)
    ax.margins(0.15)
    ax.set(xlabel="Subset mean fresh Adam cosine", ylabel="Gain in resolved fraction")
    fig.canvas.draw()
    occupied = []
    for annotation in annotations:
        for offset in [
            (5, 5),
            (5, -13),
            (-14, 5),
            (-14, -13),
            (10, 20),
            (-18, 20),
            (20, -20),
            (-25, -20),
        ]:
            annotation.set_position(offset)
            box = annotation.get_window_extent(fig.canvas.get_renderer()).expanded(
                1.08, 1.08
            )
            if not any(box.overlaps(previous) for previous in occupied):
                occupied.append(box)
                break
    save(fig, "score_gain")

    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    for index, group in enumerate(COLORS):
        vals = [
            s["plot_gain"]
            for s in summaries
            if s["group"] == group and s["plot_gain"] is not None
        ]
        ax.scatter(
            index + np.linspace(-0.12, 0.12, len(vals)),
            vals,
            color=COLORS[group],
            marker=MARKERS[group],
            s=45,
        )
    ax.set_xticks(range(4), list(COLORS))
    ax.axhline(0, color="0.6", linewidth=0.7)
    ax.set(
        ylabel="Gain in resolved fraction",
        xlabel="Sampling group (3 overlapping subsets per group)",
    )
    save(fig, "group_gains")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
    for s in summaries:
        if s["plot_gain"] is None:
            continue
        for ax, key in zip(axs, ["supervised_tokens", "training_seconds"]):
            if s[key] is not None:
                ax.scatter(
                    s[key],
                    s["plot_gain"],
                    color=COLORS[s["group"]],
                    marker=MARKERS[s["group"]],
                )
    for ax, label in zip(axs, ["Supervised tokens", "Training wall time (s)"]):
        ax.set(xlabel=label, ylabel="Gain in resolved fraction")
        ax.axhline(0, color="0.6", linewidth=0.7)
    save(fig, "cost_gain")

    fig, ax = plt.subplots(figsize=(7.2, 6), layout="constrained")
    overlap = optional(
        run.root,
        "manifests/overlap.json",
        {"jaccard": [[0]], "subset_ids": ["not available"]},
    )
    im = ax.imshow(overlap["jaccard"], cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Jaccard overlap")
    ax.set_xticks(
        range(len(overlap["subset_ids"])),
        overlap["subset_ids"],
        rotation=60,
        ha="right",
    )
    ax.set_yticks(range(len(overlap["subset_ids"])), overlap["subset_ids"])
    save(fig, "overlap")

    fig, ax = plt.subplots(
        figsize=(7.2, max(3.6, len(subsets) * 0.3)), layout="constrained"
    )
    repos = sorted(set(repo for s in subsets for repo in s["repos"]))
    if repos and subsets:
        matrix = np.array(
            [[s["repos"].get(repo, 0) for repo in repos] for s in subsets]
        )
        im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0)
        fig.colorbar(im, ax=ax, label="Member count")
        ax.set_xticks(range(len(repos)), repos, rotation=90, ha="center", fontsize=8)
        ax.set_yticks(range(len(subsets)), [s["subset_id"] for s in subsets])
    save(fig, "repository_composition")
    write(folder / "layout_audit.json", qa)
    paths.append(folder / "layout_audit.json")
    return paths


def audit(run: Run) -> dict:
    with run.stage("audit") as out:
        if out is None:
            return read(run.root / "audit.json")
        checks = []

        def check(name, ok, evidence, warning=False):
            checks.append(
                {
                    "check": name,
                    "status": "PASS" if ok else "WARN" if warning else "FAIL",
                    "evidence": evidence,
                }
            )

        root = run.root
        candidates = optional(root, "manifests/candidates.json", [])
        tasks = optional(root, "manifests/validation.json", [])
        subsets = optional(root, "manifests/subsets.json", [])
        scores = optional(root, "scores/candidate_scores.json", [])
        check(
            "candidate_count_unique",
            len(candidates) == len({r["instance_id"] for r in candidates}) == 300,
            "manifests/candidates.json",
        )
        check(
            "validation_count_unique",
            len(tasks) == len({r["instance_id"] for r in tasks}) == 30,
            "manifests/validation.json",
        )
        check(
            "score_count_finite",
            len(scores) == 300
            and all(
                math.isfinite(r[k])
                for r in scores
                for k in ["fresh_adam_cosine", "raw_grad_cosine"]
            ),
            "scores/candidate_scores.json",
        )
        check(
            "random_groups_scored",
            sum(s["group"] == "random" for s in subsets) == 3
            and all(
                "fresh_adam_cosine" in s and "raw_grad_cosine" in s for s in subsets
            ),
            "manifests/subsets.json",
        )
        check(
            "subset_count_members",
            len(subsets) == 12 and all(len(set(s["members"])) == 30 for s in subsets),
            "manifests/subsets.json",
        )
        check(
            "no_identical_subsets",
            len({tuple(s["members"]) for s in subsets}) == len(subsets),
            "manifests/subsets.json",
        )
        check(
            "token_budgets",
            bool(subsets)
            and all(
                abs(s["supervised_tokens"] - s["budget_target"])
                <= 0.1 * s["budget_target"]
                for s in subsets
            ),
            "manifests/budget.json",
        )
        strata = optional(root, "manifests/strata.json", {})
        check(
            "strata_100_each",
            all(len(strata.get(g, [])) == 100 for g in ["low", "middle", "high"]),
            "manifests/strata.json",
        )
        score_by_id = {r["data_id"]: r for r in scores}
        ordered = sorted(scores, key=lambda s: (s["fresh_adam_cosine"], s["data_id"]))
        check(
            "strata_follow_primary_score",
            bool(ordered)
            and all(
                strata.get(g) == [r["data_id"] for r in ordered[start : start + 100]]
                for g, start in [("low", 0), ("middle", 100), ("high", 200)]
            ),
            "manifests/strata.json",
        )
        check(
            "members_from_declared_strata",
            bool(subsets)
            and all(
                set(s["members"]) <= set(strata.get(s["group"], [])) for s in subsets
            ),
            "manifests/subsets.json",
        )
        check(
            "frozen_prediction_means",
            bool(subsets)
            and all(
                abs(
                    s["fresh_adam_cosine"]["mean"]
                    - np.mean(
                        [
                            score_by_id[m]["fresh_adam_cosine"]
                            for m in s["members"]
                            if m in score_by_id
                        ]
                    )
                )
                < 1e-12
                for s in subsets
            ),
            "manifests/subsets.json",
        )
        training = [
            optional(root, f"training/{s['subset_id']}/training.json", {})
            for s in subsets
        ]
        check(
            "common_initial_adapter",
            len(training) == 12
            and len({t.get("initial_adapter_sha256") for t in training}) == 1
            and all(t.get("initial_adapter_sha256") for t in training),
            "training/*/training.json",
        )
        check(
            "exact_one_pass_updates",
            len(training) == 12
            and all(t.get("updates") == t.get("exposures") == 30 for t in training),
            "training/*/step_*.metrics.json",
        )
        check(
            "training_exposure_members",
            bool(training)
            and all(
                t.get("members") == s["members"] for t, s in zip(training, subsets)
            ),
            "training/*/training.json",
        )
        main_results = optional(root, "rollouts/nice/results.json", [])
        check(
            "nice_120_episodes",
            len(main_results) == 120
            and all(
                sum(r["task_id"] == t["instance_id"] for r in main_results) == 4
                for t in tasks
            ),
            "rollouts/nice/results.json",
        )
        for model in ["initial"] + [s["subset_id"] for s in subsets]:
            result = optional(root, f"evaluation/{model}/results.json", [])
            main_results += result
            check(
                "evaluation_denominator:" + model,
                task_rate(result, [t["instance_id"] for t in tasks])["status"]
                == "completed"
                and len(tasks) == 30,
                f"evaluation/{model}/results.json",
            )
        check(
            "510_main_episodes",
            len(main_results) == 510,
            "evaluation/*/results.json; rollouts/nice/results.json",
        )
        check(
            "no_unresolved_infrastructure",
            all(r.get("status") != "INFRA_ERROR" for r in main_results),
            "logs/events.jsonl",
        )
        diagnostics = optional(root, "scores/diagnostics.json", {})
        check(
            "score_cache_identity",
            bool(scores)
            and all(
                s["cache_identity"] == diagnostics.get("cache_identity") for s in scores
            ),
            "scores/diagnostics.json",
        )
        check(
            "reward_signal",
            diagnostics.get("zero_gradient_tasks", 30) < 30,
            "scores/diagnostics.json",
        )
        check(
            "reward_coverage",
            diagnostics.get("warning") != "LOW_REWARD_COVERAGE",
            "scores/diagnostics.json",
            warning=True,
        )
        check(
            "all_stage_artifacts_valid",
            all(
                run.valid(name)
                for name, state in run.state().items()
                if name != "audit" and state.get("status") == "completed"
            ),
            "run_state.json",
        )
        for name in TABLES:
            check(
                "table:" + name,
                (root / "tables" / f"{name}.csv").is_file(),
                f"tables/{name}.csv",
            )
        for name in FIGURES:
            check(
                "figure:" + name,
                all(
                    (root / "figures" / f"{name}.{ext}").is_file()
                    for ext in ["png", "pdf"]
                ),
                f"figures/{name}.*",
            )
        qa = optional(root, "figures/layout_audit.json", [])
        check(
            "figure_layout",
            len(qa) == 7 and all(q["status"] == "PASS" for q in qa),
            "figures/layout_audit.json",
        )
        if run.cfg.get("transfer", {}).get("enabled"):
            from nice_validation.transfer_sources import SUITES

            external = optional(root, "transfer/tasks.json", {})
            for suite in SUITES:
                selection = external.get(suite, [])
                check(
                    "transfer_sample:" + suite,
                    len(selection) == len({t["instance_id"] for t in selection}) == 50,
                    "transfer/tasks.json",
                )
                forbidden = {t["instance_id"] for t in selection}
                check(
                    "transfer_disjoint:" + suite,
                    not forbidden.intersection(
                        {t["instance_id"] for t in tasks + candidates}
                    ),
                    "transfer/tasks.json; manifests",
                )
                for model_id in ["initial"] + [s["subset_id"] for s in subsets]:
                    path = (
                        root / "transfer/evaluation" / suite / model_id / "results.json"
                    )
                    values = read(path) if path.is_file() else []
                    measure = task_rate(values, [t["instance_id"] for t in selection])
                    check(
                        f"transfer_denominator:{suite}:{model_id}",
                        measure["status"] == "completed"
                        and measure["denominator"] == 50,
                        str(path.relative_to(root)),
                    )
            transfer_value = optional(root, "transfer/summary.json", {})
            check(
                "transfer_report_complete",
                transfer_value.get("transfer_episode_count") == 2600
                and len(transfer_value.get("rows", [])) == 52,
                "transfer/summary.json",
            )
            transfer_qa = optional(root, "transfer/figures/layout_audit.json", [])
            check(
                "transfer_figures",
                len(transfer_qa) == 2
                and all(item["status"] == "PASS" for item in transfer_qa),
                "transfer/figures/layout_audit.json",
            )
        if run.cfg["mode"] == "mock":
            check(
                "formal_evidence",
                False,
                "Explicit synthetic fixture. No target model, Docker or GPU verification performed.",
                warning=True,
            )
        else:
            from nice_validation.smoke import require_real_smoke

            try:
                require_real_smoke(run)
                smoke_ready = True
            except ValueError:
                smoke_ready = False
            check(
                "no_mock_results",
                all(r.get("mode") == "real" for r in main_results),
                "evaluation/*/results.json",
            )
            check(
                "server_smoke",
                smoke_ready,
                "smoke/result.json",
            )
        outcome = (
            "FAIL"
            if any(c["status"] == "FAIL" for c in checks)
            else "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"
        )
        value = {"status": outcome, "mode": run.cfg["mode"], "checks": checks}
        write(root / "audit.json", value)
        (root / "audit.md").write_text(
            f"# Audit: {outcome}\n\n"
            + "\n".join(
                f"- {c['status']} {c['check']}: {c['evidence']}" for c in checks
            )
            + "\n",
            encoding="utf-8",
        )
        out.extend([root / "audit.json", root / "audit.md"])
        return value
