"""Run only under Live's isolated interpreter, avoiding a conflicting swebench import."""

import argparse
import importlib.metadata
import json
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--namespace", default="starryzhang")
    args = parser.parse_args()
    dist = importlib.metadata.distribution("swebench")
    source = json.loads(dist.read_text("direct_url.json") or "{}")
    files = {
        str(file): hashlib.sha256(Path(dist.locate_file(file)).read_bytes()).hexdigest()
        for file in dist.files or []
        if str(file).replace("\\", "/").startswith("swebench/")
        and str(file).endswith(".py")
    }
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    }
    result = {
        "version": dist.version,
        "source": source,
        "source_files": files,
        "packages": packages,
    }
    if args.task:
        from swebench.harness.test_spec.test_spec import make_test_spec

        task = json.loads(Path(args.task).read_text(encoding="utf-8"))
        spec = make_test_spec(task, namespace=args.namespace)
        result["image"] = spec.instance_image_key
    print(json.dumps(result))


if __name__ == "__main__":
    main()
