"""Separate descriptive transfer tables and raw-point figures for four held-out suites."""

from pathlib import Path
import warnings
import shutil

from nice_validation.analysis import COLORS, MARKERS, csv_write, ranking, task_rate
from nice_validation.core import Run, digest, read, sha, utc, write
from nice_validation.transfer_sources import SUITES


def transfer_summary(run: Run) -> dict:
    selections = read(run.root / "transfer/tasks.json")
    subsets_path = run.root / "manifests/subsets.json"
    subsets = read(subsets_path) if subsets_path.is_file() else []
    models = ["initial"] + [s["subset_id"] for s in subsets]
    result, paired, correlations = [], [], {}
    mock = run.cfg["mode"] == "mock"
    for suite in SUITES:
        task_ids = [t["instance_id"] for t in selections[suite]]
        model_rows = {}
        for model_id in models:
            path = run.root / "transfer/evaluation" / suite / model_id / "results.json"
            records = read(path) if path.is_file() else []
            model_rows[model_id] = records
            if not mock and any(r.get("mode") != "real" for r in records):
                raise ValueError("Mock transfer rows cannot enter formal analysis")
        baseline = task_rate(model_rows["initial"], task_ids)
        values = []
        for model_id in models:
            records = model_rows[model_id]
            measured = task_rate(records, task_ids)
            rate, base = measured["rate"], baseline["rate"]
            delta = rate - base if rate is not None and base is not None else None
            subset = next((s for s in subsets if s["subset_id"] == model_id), {})
            native = (
                [r.get("native_reward", float(r["resolved"])) for r in records]
                if measured["status"] == "completed"
                else []
            )
            row = {
                "benchmark": suite,
                "model_id": model_id,
                "group": subset.get("group", "initial"),
                "sampling_seed": subset.get("sampling_seed"),
                "denominator": 50,
                "status": measured["status"],
                "resolved_count": measured["resolved_count"],
                "completed_tasks": measured["completed_tasks"],
                "success_rate": None if mock else rate,
                "delta": None if mock else delta,
                "native_mean_reward": (
                    None if mock or not native else sum(native) / len(native)
                ),
                "synthetic_demo_rate": rate if mock else None,
                "synthetic_demo_delta": delta if mock else None,
                "nice_score": subset.get("fresh_adam_cosine", {}).get("mean"),
                "mode": run.cfg["mode"],
            }
            result.append(row)
            if model_id != "initial" and delta is not None:
                values.append((row["nice_score"], delta))
        correlations[suite] = {
            "formal": (
                None
                if mock or len(values) != 12
                else ranking([x for x, y in values], [y for x, y in values])
            ),
            "synthetic_demo": (
                ranking([x for x, y in values], [y for x, y in values])
                if mock
                else None
            ),
            "paired_subsets": len(values),
        }
        for task_id in task_ids:
            paired.append(
                {
                    "benchmark": suite,
                    "task_id": task_id,
                    **{
                        model: next(
                            (
                                r["resolved"]
                                for r in rows
                                if r["task_id"] == task_id
                                and r["status"] != "INFRA_ERROR"
                            ),
                            None,
                        )
                        for model, rows in model_rows.items()
                    },
                }
            )
    complete = len(result) == 52 and all(row["status"] == "completed" for row in result)
    return {
        "mode": run.cfg["mode"],
        "formal_evidence": not mock and complete,
        "status": (
            "synthetic_fixture" if mock else "completed" if complete else "incomplete"
        ),
        "rows": result,
        "paired_tasks": paired,
        "correlations": correlations,
        "transfer_episode_count": sum(row["completed_tasks"] for row in result),
        "planned_transfer_episodes": 2600,
        "planned_all_agent_episodes": 3110,
        "limitations": [
            "Four different task distributions; report each benchmark separately.",
            "50 tasks gives 2 percentage points per binary success; 12 overlapping subsets share the same tasks.",
            "No task outcomes used for sampling or training; public pretraining contamination is not ruled out.",
            "Custom agent and budgets; these subset scores are not full official leaderboard scores.",
        ],
    }


def plot_transfer(run: Run, rows: list[dict]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    folder = run.root / "transfer/figures"
    folder.mkdir(parents=True, exist_ok=True)
    paths, qa = [], []
    plt.rcParams.update(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    mock = run.cfg["mode"] == "mock"
    gain = "synthetic_demo_delta" if mock else "delta"
    for name in ["score_gain", "group_gain"]:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
        fig.suptitle(
            ("SYNTHETIC FIXTURE | " if mock else "")
            + "Held-out transfer: "
            + name.replace("_", " "),
            fontsize=10,
        )
        for ax, suite in zip(axes.flat, SUITES):
            selected = [
                r
                for r in rows
                if r["benchmark"] == suite
                and r["model_id"] != "initial"
                and r[gain] is not None
            ]
            ax.set_title(
                {
                    "verified": "SWE-bench Verified",
                    "pro": "Pro Public V2",
                    "live": "SWE-bench Live (Python)",
                    "terminal": "Terminal-Bench 2.0",
                }[suite]
            )
            ax.axhline(0, color="gray", lw=0.7)
            for row in selected:
                x = (
                    row["nice_score"]
                    if name == "score_gain"
                    else list(COLORS).index(row["group"])
                    + ({11: -0.14, 22: 0.0, 33: 0.14}[row["sampling_seed"]])
                )
                ax.scatter(
                    x,
                    row[gain],
                    color=COLORS[row["group"]],
                    marker=MARKERS[row["group"]],
                    s=26,
                )
            ax.set_ylabel("Resolved fraction gain")
            if name == "score_gain":
                ax.set_xlabel("Mean NICE score")
            else:
                ax.set_xticks(range(4), list(COLORS))
                ax.set_xlabel("3 overlapping subsets per group")
            if not selected:
                ax.text(
                    0.5,
                    0.5,
                    "Incomplete evaluation",
                    transform=ax.transAxes,
                    ha="center",
                )
        if name == "score_gain":
            from matplotlib.lines import Line2D

            axes.flat[0].legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color=COLORS[group],
                        marker=MARKERS[group],
                        linestyle="",
                        label=group,
                    )
                    for group in COLORS
                ],
                loc="best",
                fontsize=7,
                ncol=2,
                frameon=False,
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.canvas.draw()
            clipped = []
            for text in fig.findobj(matplotlib.text.Text):
                if text.get_visible() and text.get_text():
                    box = text.get_window_extent(fig.canvas.get_renderer())
                    if (
                        box.x0 < -1
                        or box.y0 < -1
                        or box.x1 > fig.bbox.x1 + 1
                        or box.y1 > fig.bbox.y1 + 1
                    ):
                        # Tick labels outside the displayed range are not drawn.
                        if (
                            text in [a.title for a in axes.flat]
                            or text is fig._suptitle
                            or text in [a.xaxis.label for a in axes.flat]
                            or text in [a.yaxis.label for a in axes.flat]
                        ):
                            clipped.append(text.get_text())
            missing = [str(w.message) for w in caught if "Glyph" in str(w.message)]
        qa.append(
            {
                "figure": name,
                "missing_glyphs": missing,
                "clipped_text": clipped,
                "status": "PASS" if not missing and not clipped else "FAIL",
            }
        )
        for ext in ["png", "pdf"]:
            path = folder / f"{name}.{ext}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        gray = folder / f"{name}.grayscale.png"
        with Image.open(folder / f"{name}.png") as image:
            image.convert("L").save(gray, dpi=(300, 300))
        paths.append(gray)
        plt.close(fig)
    path = folder / "layout_audit.json"
    write(path, qa)
    paths.append(path)
    return paths


def analyze_transfer(run: Run) -> None:
    if not run.cfg.get("transfer", {}).get("enabled"):
        return
    inputs = digest(
        {
            str(p.relative_to(run.root)): sha(p)
            for p in sorted((run.root / "transfer/evaluation").glob("*/*/results.json"))
        }
    )
    destination = run.root / "transfer/summary.json"
    state = run.state()
    previous = state.get("analyze-transfer", {})
    if (
        run.valid("analyze-transfer")
        and read(destination).get("input_identity") != inputs
    ):
        archive = run.root / "history" / ("transfer-" + utc().replace(":", "-"))
        for name in previous["artifacts"]:
            source = run.root / name
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write(archive / "previous_state.json", previous)
        state.pop("analyze-transfer", None)
        write(run.state_path, state)
    with run.stage("analyze-transfer") as artifacts:
        if artifacts is None:
            return
        value = transfer_summary(run)
        value["input_identity"] = inputs
        destination = run.root / "transfer/summary.json"
        write(destination, value)
        artifacts.append(destination)
        profile = {
            "benchmark_count": 4,
            "tasks_per_benchmark": 50,
            "model_rows": len(value["rows"]),
            "group_counts_per_benchmark": {
                suite: {
                    group: sum(
                        r["benchmark"] == suite and r["group"] == group
                        for r in value["rows"]
                    )
                    for group in COLORS
                }
                for suite in SUITES
            },
            "missing_gain_count": sum(
                r["model_id"] != "initial"
                and r["delta" if run.cfg["mode"] == "real" else "synthetic_demo_delta"]
                is None
                for r in value["rows"]
            ),
            "rationale": "Per-suite scatter and individual group points; no pooled benchmark score, mean bars or significance claims.",
        }
        profile_path = run.root / "transfer/data_profile.json"
        write(profile_path, profile)
        artifacts.append(profile_path)
        for name, rows in [
            ("summary", value["rows"]),
            ("paired_tasks", value["paired_tasks"]),
        ]:
            path = run.root / "transfer/tables" / f"{name}.csv"
            csv_write(path, rows)
            artifacts.append(path)
        artifacts.extend(plot_transfer(run, value["rows"]))
        path = run.root / "transfer/report.md"
        path.write_text(
            "# Held-out benchmark transfer\n\n"
            + (
                "SYNTHETIC FIXTURE; no research conclusion.\n\n"
                if run.cfg["mode"] == "mock"
                else ""
            )
            + "NICE scoring and same-task evaluation use held-out SWE-smith tasks. Four external suites use independently frozen samples of 50 tasks; initial and all 12 adapters share each sample.\n\n"
            + "\n".join("- " + item for item in value["limitations"])
            + "\n\nSee tables/summary.csv and paired_tasks.csv for per-suite rates, paired deltas and missing-task status.\n",
            encoding="utf-8",
        )
        artifacts.append(path)
