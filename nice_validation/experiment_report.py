"""Auditable checkpoint selection, paired held-out results and explicit mock outputs."""
import math
import warnings

from nice_validation.core import read, write, read_jsonl
from nice_validation.analysis import csv_write
from nice_validation.experiment import rate
from nice_validation.experiment_design import counts, jobs, points
from nice_validation.experiment_training import validate_point


def report(run):
    with run.stage("experiment-report") as artifacts:
        if artifacts is None:
            return
        root, cfg, e = run.root, run.cfg, run.cfg["experiment"]
        mock, design = cfg["mode"] == "mock", counts(cfg)
        candidates = read(root / "manifests/candidates.json")
        subsets = read(root / "manifests/subsets.json")
        entries = read(root / "manifests/training_jobs.json")
        milestones = read(root / "manifests/checkpoints.json")
        chosen = read(root / "selection.json")
        checks = []

        def check(name, okay, evidence):
            checks.append({"check": name, "status": "PASS" if okay else "FAIL", "evidence": evidence})

        check("source_mode", chosen["mode"] == cfg["mode"] and chosen["run_identity"] == run.identity,
              "selection.json; config/snapshot.json")
        split_ids = read(root / "manifests/split_ids.json")
        sizes = {"candidates": e["candidates"], "nice": e["validation"],
                 "sft_validation": e["sft_validation"], "benchmark": e["benchmark"], "development": 1}
        flat = [n for key in sizes for n in split_ids[key]]
        check("split_counts_unique_disjoint", all(len(split_ids[k]) == len(set(split_ids[k])) == n for k, n in sizes.items())
              and len(flat) == len(set(flat)), "manifests/split_ids.json")
        families = {k: set(v) for k, v in read(root / "manifests/split_family_ids.json").items()}
        import itertools
        check("split_explicit_family_closure_disjoint", all(not families[a].intersection(families[b])
              for a, b in itertools.combinations(sizes, 2)), "manifests/split_family_ids.json")
        scores = read(root / "scores/candidate_scores.json")
        ordered = sorted(scores, key=lambda r: (r["fresh_adam_cosine"], r["data_id"]))
        check("candidate_score_coverage", len(scores) == e["candidates"]
              and {r["data_id"] for r in scores} == {r["data_id"] for r in candidates}
              and all(math.isfinite(r[k]) for r in scores for k in ["fresh_adam_cosine", "raw_grad_cosine"]),
              "scores/candidate_scores.json")
        strata = read(root / "manifests/strata.json")
        check("ascending_score_halves", strata == {"low": [r["data_id"] for r in ordered[:len(ordered)//2]],
                                                   "high": [r["data_id"] for r in ordered[len(ordered)//2:]]},
              "manifests/strata.json")
        check("two_512_subsets", len(subsets) == 2 and {s["group"] for s in subsets} == {"low", "high"}
              and all(s["sampling_seed"] == 11 and len(s["members"]) == len(set(s["members"])) == e["subset_size"]
                      and set(s["members"]) <= set(strata[s["group"]])
                      and abs(s["supervised_tokens"] - s["budget_target"]) <= cfg["experiment"]["budget_tolerance"] * s["budget_target"] for s in subsets),
              "manifests/subsets.json; manifests/budget.json")
        check("four_matched_lr_jobs", entries == jobs(subsets, cfg), "manifests/training_jobs.json")
        check("twelve_exact_exposure_checkpoints", milestones == [p for j in entries for p in points(j, cfg)],
              "manifests/checkpoints.json")
        training = []
        from nice_validation.batch_tuning import training_config
        from nice_validation.experiment_training import training_identity
        expected_training_ids = {}
        for job in entries:
            path = root / "training" / job["subset_id"] / "training.json"
            row = read(path)
            training.append(row)
            training_cfg, plan_sha = training_config(run)
            training_cfg["learning_rate"] = job["learning_rate"]
            job_rows = [r for r in candidates if r["data_id"] in job["members"]]
            expected_id = training_identity(run, job, job_rows, training_cfg, plan_sha)
            expected_training_ids[job["subset_id"]] = expected_id
            check("training:" + job["subset_id"], run.valid("train:" + job["subset_id"])
                  and row["exposures"] == e["subset_size"]
                  and row["updates"] == e["subset_size"] // cfg["training"]["effective_batch"]
                  and row["learning_rate"] == job["learning_rate"] and set(row["members"]) == set(job["members"])
                  and row.get("training_identity") == expected_id, str(path.relative_to(root)))
        for point in milestones:
            record = validate_point(run, point)
            check("checkpoint_data_identity:" + point["model_id"],
                  record.get("training_identity") == expected_training_ids[point["job_id"]],
                  "training/" + point["model_id"] + "/checkpoint.json")
        nice = read(root / "rollouts/nice/results.json")
        check("nice_eight_rollouts_per_task", len(nice) == design["nice_episodes"]
              and all(sorted(r["repeat"] for r in nice if r["task_id"] == t) == list(range(e["rollouts"])) for t in split_ids["nice"])
              and all(r["run_identity"] == run.identity and r["status"] in {"RESOLVED", "TASK_FAILED"} for r in nice),
              "rollouts/nice/results.json")
        diagnostics = read(root / "scores/diagnostics.json")
        check("nonzero_nice_direction", diagnostics["zero_gradient_tasks"] < e["validation"], "scores/diagnostics.json")
        curves, benchmark, retest, task_results = [], [], [], []
        for scope, models, n in [("selection", ["initial", *[p["model_id"] for p in milestones]], e["sft_validation"]),
                                  ("benchmark", ["initial", *[p["model_id"] for p in chosen["selected"]]], e["benchmark"]),
                                  ("nice_retest", ["initial", *[p["model_id"] for p in chosen["selected"]]], e["validation"])]:
            for model_id in models:
                records = read(root / "evaluation" / scope / model_id / "results.json")
                correct = rate(records, n)
                expected = set(split_ids[{"selection": "sft_validation", "benchmark": "benchmark", "nice_retest": "nice"}[scope]])
                check(scope + ":" + model_id, run.valid(scope + ":" + model_id)
                      and {r["task_id"] for r in records} == expected
                      and all(r["mode"] == cfg["mode"] and r["run_identity"] == run.identity and r["model_id"] == model_id for r in records),
                      f"evaluation/{scope}/{model_id}/results.json")
                point = next((p for p in milestones if p["model_id"] == model_id), {})
                row = {"model_id": model_id, **point, "denominator": n,
                       "resolved_count": None if mock else correct, "success_rate": None if mock else correct / n,
                       "synthetic_demo_resolved_count": correct if mock else None,
                       "synthetic_demo_rate": correct / n if mock else None, "mode": cfg["mode"]}
                {"selection": curves, "benchmark": benchmark, "nice_retest": retest}[scope].append(row)
                task_results.extend({"scope": scope, **r} for r in records)
        recomputed = []
        for job in entries:
            options = []
            for p in milestones:
                if p["job_id"] == job["subset_id"]:
                    records = read(root / "evaluation/selection" / p["model_id"] / "results.json")
                    options.append({**p, "resolved_count": rate(records, e["sft_validation"])})
            recomputed.append(min(options, key=lambda p: (-p["resolved_count"], p["exposures"])))
        check("validation_only_four_selections", chosen["selected"] == recomputed
              and chosen["benchmark_used_for_selection"] is False and run.valid("model-selection"), "selection.json")
        check("expected_episode_count", len(task_results) + len(nice) == design["all_episodes"], "evaluation/; rollouts/nice/")
        check("no_old_external_suites", not (root / "transfer").exists(), "transfer.enabled=false")
        check("all_required_stages_immutable", all(run.valid(k) for k in ["prepare-data", "smoke", "score", "make-subsets", "training-jobs"]), "run_state.json")
        if any(r["status"] == "FAIL" for r in checks):
            write(root / "audit.json", {"status": "FAIL", "mode": cfg["mode"], "checks": checks})
            raise ValueError("New experiment audit failed; inspect audit.json")
        paired = []
        for task_id in split_ids["benchmark"]:
            paired.append({"task_id": task_id, **{r["model_id"]: r["resolved"] for r in task_results
                          if r["scope"] == "benchmark" and r["task_id"] == task_id}})
        comparisons = []
        for lr in cfg["training"]["learning_rates"]:
            low = next(p["model_id"] for p in chosen["selected"] if p["group"] == "low" and p["learning_rate"] == lr)
            high = next(p["model_id"] for p in chosen["selected"] if p["group"] == "high" and p["learning_rate"] == lr)
            a = sum(r[low] and not r[high] for r in paired)
            b = sum(r[high] and not r[low] for r in paired)
            comparisons.append({"learning_rate": lr, "low_model": low, "high_model": high,
                                "only_low_resolved": a, "only_high_resolved": b,
                                "high_minus_low": None if mock else (b - a) / e["benchmark"],
                                "synthetic_demo_high_minus_low": (b - a) / e["benchmark"] if mock else None,
                                "mode": cfg["mode"]})
        retest_tasks = [r for r in task_results if r["scope"] == "nice_retest"]
        retest_paired = [{"task_id": task_id, **{r["model_id"]: r["resolved"] for r in retest_tasks
                          if r["task_id"] == task_id}} for task_id in split_ids["nice"]]
        retest_comparisons = []
        for row in comparisons:
            low, high = row["low_model"], row["high_model"]
            a = sum(r[low] and not r[high] for r in retest_paired)
            b = sum(r[high] and not r[low] for r in retest_paired)
            retest_comparisons.append({**row, "only_low_resolved": a, "only_high_resolved": b,
                "denominator": e["validation"], "scope": "nice_retest", "independent_test": False,
                "high_minus_low": None if mock else (b-a) / e["validation"],
                "synthetic_demo_high_minus_low": (b-a) / e["validation"] if mock else None})
        events = read_jsonl(root / "logs/events.jsonl")
        for service_log in sorted((root / "logs").glob("service-*.jsonl")):
            events.extend(read_jsonl(service_log))
        for name, rows in [("candidate_scores", scores), ("subset_summary", subsets), ("training_summary", training),
                           ("checkpoint_validation", curves), ("benchmark_summary", benchmark), ("task_results", task_results),
                           ("nice_task_results", nice),
                           ("nice_retest_summary", retest), ("nice_retest_task_results", retest_tasks),
                           ("nice_retest_paired_tasks", retest_paired), ("nice_retest_high_low_comparison", retest_comparisons),
                           ("paired_tasks", paired), ("high_low_comparison", comparisons),
                           ("stage_costs", [r for r in events if r["event"] == "stage_cost"]),
                           ("failure_summary", [r for r in task_results if not r["resolved"]])]:
            path = root / "tables" / (name + ".csv")
            csv_write(path, rows)
            artifacts.append(path)
        plots(run, curves, benchmark, scores, artifacts)
        coverage = 1 - diagnostics["zero_gradient_tasks"] / e["validation"]
        def known_sum(rows, key):
            values = [r.get(key) for r in rows]
            return sum(values) if values and all(type(v) is int and v >= 0 for v in values) else None
        token_usage = {}
        for scope, rows in [("nice", nice), ("selection", [r for r in task_results if r["scope"] == "selection"]),
                            ("nice_retest", retest_tasks),
                            ("benchmark", [r for r in task_results if r["scope"] == "benchmark"])]:
            token_usage[scope] = {key: known_sum(rows, key) for key in ["input_tokens", "output_tokens", "logprob_replay_tokens"]}
        token_usage["training"] = {key: None if mock else known_sum(training, key)
                                   for key in ["logical_input_tokens", "supervised_tokens", "padding_tokens"]}
        summary = {"mode": cfg["mode"], "formal_evidence": not mock, "experiment_design": design,
                   "logical_token_usage": token_usage,
                   "main_episode_count": len(task_results) + len(nice),
                   "successful_updates": sum(r["updates"] for r in training),
                   "training_exposures": sum(r["exposures"] for r in training),
                   "nice_reward_coverage": coverage, "selected_models": chosen["selected"],
                   "benchmark": benchmark, "comparisons": comparisons,
                   "nice_retest": {"independent_test": False, "used_for_checkpoint_selection": False,
                                   "summary": retest, "comparisons": retest_comparisons},
                   "limitations": ["Only one data-sampling and training-order seed; no seed-stability claim.",
                      "The 20 Verified NICE tasks are disjoint from 100 held-out Verified benchmark tasks.",
                      "Final greedy retest on the 20 NICE tasks is selection-exposed diagnostic evidence, not an independent test; it never selects checkpoints.",
                      "SWE-smith selection validation chooses a checkpoint within each group and learning rate.",
                      "Checkpoint validation scores are adaptive selection evidence, not held-out benchmark scores.",
                      "Models share the same benchmark tasks; these are paired comparisons, not independent repeats.",
                      "Potential pretraining overlap is unknown; task-ID/family/prompt checks do not prove its absence.",
                      f"{cfg['agent']['max_new_tokens']}/{cfg['agent']['total_new_tokens']} output tokens, {cfg['agent']['context_length']} context tokens and wall-time budgets are project settings, not universal Verified benchmark defaults."]}
        write(root / "summary.json", summary)
        audit = {"status": "WARN" if mock or coverage < cfg["score"]["low_reward_coverage"] else "PASS",
                 "mode": cfg["mode"], "checks": checks,
                 "warnings": (["Synthetic fixture, not research evidence"] if mock else [])
                            + (["Low nonzero NICE validation-direction coverage"] if coverage < cfg["score"]["low_reward_coverage"] else [])}
        write(root / "audit.json", audit)
        (root / "audit.md").write_text("# Audit\n\n" + audit["status"] + "\n\n" + "\n".join(f"- {r['status']}: {r['check']}" for r in checks), encoding="utf-8")
        text = ("# SYNTHETIC FIXTURE — NO RESEARCH EVIDENCE\n\n" if mock else "# NICE 高低组选样实验报告\n\n")
        text += (f"候选 {e['candidates']} 条；高低各 {e['subset_size']} 条；NICE {e['validation']} 题 × {e['rollouts']} 次。\n\n"
                 f"两个学习率各训练高低两组，共 {len(entries)} 个分支、{len(milestones)} 个阶段权重。"
                 f"使用独立 SWE-smith {e['sft_validation']} 题，为每个分支选择阶段权重。\n\n"
                 f"最终 initial 加四份选中权重，共用 Verified {e['benchmark']} 题；总计 {design['all_episodes']} 个正式 episode。\n\n"
                 f"另对 NICE 原 {e['validation']} 题，以 initial+四份选中权重各做一次 greedy 复测。"
                 "该集合已参与 NICE 选样，仅作诊断，不是独立 benchmark；不用于重新选权重。"
                 "见 tables/nice_retest_summary.csv、nice_retest_task_results.csv、nice_retest_paired_tasks.csv 和 nice_retest_high_low_comparison.csv。"
                 "initial 也重新 greedy 运行，不能拿原先随机 NICE rollout 的成功率代替基线。\n\n"
                 f"非零 NICE 验证方向覆盖率：{coverage:.1%}。零方向题仍留在固定分母中。\n\n"
                 "优先阅读 tables/benchmark_summary.csv、high_low_comparison.csv、paired_tasks.csv；"
                 "选权重过程见 checkpoint_validation.csv 和 selection.json；失败原因见 task_results.csv 与原始 trace。\n\n"
                 "训练进度图连接同一分支的有序训练节点；benchmark 点图误差线为 95% Wilson 区间，"
                 "不是跨 seed 的误差。模型共用任务，比较应使用逐题配对结果。\n\n"
                 "## 局限\n\n" + "\n".join("- " + s for s in summary["limitations"]) + "\n")
        (root / "report.md").write_text(text, encoding="utf-8")
        artifacts.extend(root / n for n in ["summary.json", "audit.json", "audit.md", "report.md"])


def plots(run, curves, benchmark, scores, artifacts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    folder = run.root / "figures"
    folder.mkdir(exist_ok=True)
    mock = run.cfg["mode"] == "mock"
    key = "synthetic_demo_rate" if mock else "success_rate"
    prefix = "SYNTHETIC FIXTURE | " if mock else ""
    qa = []

    def save(fig, name):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.canvas.draw()
            renderer, bounds = fig.canvas.get_renderer(), fig.bbox
            clipped = []
            skipped = set()
            for ax in fig.axes:
                for axis in [ax.xaxis, ax.yaxis]:
                    low, high = sorted(axis.get_view_interval())
                    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                        if tick.get_loc() < low - 1e-10 or tick.get_loc() > high + 1e-10:
                            skipped.update([id(tick.label1), id(tick.label2)])
            for text in fig.findobj(matplotlib.text.Text):
                if id(text) not in skipped and text.get_visible() and text.get_text():
                    box = text.get_window_extent(renderer)
                    if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1):
                        clipped.append(text.get_text())
            glyphs = [str(w.message) for w in caught if "Glyph" in str(w.message)]
        qa.append({"figure": name, "missing_glyphs": glyphs, "clipped_text": clipped,
                   "status": "FAIL" if glyphs or clipped else "PASS"})
        for ext in ["png", "pdf", "svg"]:
            path = folder / (name + "." + ext)
            fig.savefig(path, dpi=300)
            artifacts.append(path)
        with Image.open(folder / (name + ".png")) as image:
            path = folder / (name + ".grayscale.png")
            image.convert("L").save(path, dpi=(300, 300))
            artifacts.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, layout="constrained")
    for ax, lr in zip(axes, run.cfg["training"]["learning_rates"]):
        for group, color, marker in [("low", "#0072B2", "o"), ("high", "#D55E00", "^")]:
            rows = sorted([r for r in curves if r.get("group") == group and r.get("learning_rate") == lr], key=lambda r: r["fraction"])
            ax.plot([r["fraction"] for r in rows], [r[key] for r in rows], color=color, marker=marker,
                    linestyle="--" if group == "low" else "-", label=group)
        ax.set(title=f"lr={lr:g}", xlabel="Fraction of one SFT epoch", ylim=(0, 1.02), xticks=[.25, .5, 1])
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.23), ncol=2, frameon=False)
    axes[0].set_ylabel("SWE-smith selection resolved fraction")
    fig.suptitle(prefix + "Checkpoint validation")
    save(fig, "checkpoint_validation")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    for index, row in enumerate(benchmark):
        n, value = row["denominator"], row[key]
        z = 1.959963984540054
        center = (value + z*z/(2*n)) / (1+z*z/n)
        radius = z * math.sqrt(value*(1-value)/n + z*z/(4*n*n)) / (1+z*z/n)
        low, high = center - radius, center + radius
        color = "#333333" if index == 0 else "#0072B2" if row["group"] == "low" else "#D55E00"
        ax.errorbar(index, value, yerr=[[max(0, value-low)], [max(0, high-value)]], fmt="o" if index == 0 or row.get("group") == "low" else "^", color=color, capsize=3)
    ax.set_xticks(range(len(benchmark)), ["initial" if r["model_id"] == "initial" else f"{r['group']}\nlr={r['learning_rate']:g}\n{r['fraction']:.0%} epoch" for r in benchmark])
    ax.set(ylabel="Held-out Verified resolved fraction", ylim=(0, 1.02), title=prefix + "Benchmark (95% Wilson intervals)")
    save(fig, "benchmark")
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    ax.hist([r["fresh_adam_cosine"] for r in scores], bins=25, color="#0072B2", edgecolor="white")
    values = sorted(r["fresh_adam_cosine"] for r in scores)
    ax.axvline(values[len(values)//2], color="black", linestyle="--")
    ax.set(xlabel="NICE fresh-Adam cosine", ylabel="Candidate count", title=prefix + "Score halves")
    save(fig, "candidate_distribution")
    path = folder / "layout_audit.json"
    write(path, qa)
    artifacts.append(path)
    if any(r["status"] != "PASS" for r in qa):
        raise ValueError("Figure layout audit failed")
