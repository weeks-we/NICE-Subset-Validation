"""Package small reports, tables, figures and IDs; exclude model/data payloads."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.core import read, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, target = Path(args.run), Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    files = [
        root / n
        for n in [
            "report.md",
            "summary.json",
            "audit.json",
            "audit.md",
            "doctor.json",
            "cost_estimate.json",
            "selection.json",
            "figures/layout_audit.json",
        ]
    ]
    for folder, pattern in [
        ("tables", "*.csv"),
        ("figures", "*.png"),
        ("figures", "*.pdf"),
        ("config", "*.json"),
        ("manifests", "*.json"),
        ("training_batch", "*.json"),
        ("scores", "diagnostics.json"),
        ("scores", "parallel_summary.json"),
        ("transfer/tables", "*.csv"),
        ("transfer/figures", "*.png"),
        ("transfer/figures", "*.pdf"),
    ]:
        files.extend((root / folder).glob(pattern))
    files = sorted(
        {
            p
            for p in files
            if p.is_file()
            and p.name
            not in {
                "candidates.json",
                "validation.json",
                "validation_reserve.json",
                "mapping.json",
                "development_task.json",
                "sft_validation.json",
                "benchmark.json",
            }
        }
    )
    candidates = read(root / "manifests/candidates.json")
    task_ids = [t["instance_id"] for t in read(root / "manifests/validation.json")]
    for name in ["report.md", "summary.json", "selection.json", "native_images.json"]:
        path = root / "transfer" / name
        if path.is_file():
            files.append(path)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(root).as_posix())
        archive.writestr(
            "candidate_ids.json",
            json.dumps(
                [
                    {
                        k: r[k]
                        for k in ["data_id", "instance_id", "traj_id", "source_sha256"]
                    }
                    for r in candidates
                ],
                indent=2,
            ),
        )
        archive.writestr("validation_ids.json", json.dumps(task_ids, indent=2))
        transfer_path = root / "transfer/tasks.json"
        if transfer_path.is_file():
            archive.writestr(
                "transfer_ids.json",
                json.dumps(
                    {
                        suite: [row["instance_id"] for row in rows]
                        for suite, rows in read(transfer_path).items()
                    },
                    indent=2,
                ),
            )
        archive.writestr(
            "bundle_checksums.json",
            json.dumps(
                {p.relative_to(root).as_posix(): sha(p) for p in files}, indent=2
            ),
        )
    print(f"{target} SHA256={sha(target)} bytes={target.stat().st_size}")


if __name__ == "__main__":
    main()
