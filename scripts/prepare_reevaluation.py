"""Inventory original 120 task environments and test the new internal API route."""
import argparse
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.core import write
from nice_validation.reevaluation_deploy import independent_paths, inventory, probe_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-image", help="Existing representative task image, never downloaded")
    parser.add_argument("--network", help="Existing dedicated Docker network")
    parser.add_argument("--endpoint", help="Container-reachable inference service root URL")
    parser.add_argument("--execute-server", action="store_true")
    args = parser.parse_args()
    independent_paths(args.old_run, args.output)
    if not args.execute_server:
        print("Plan only: read original 120 task manifests, verify adapters/base, inspect local images; optional container network probe. No training/rollout/download.")
        return
    if platform.system() != "Linux":
        parser.error("Execute only on the authorized Linux server")
    if any([args.probe_image, args.network, args.endpoint]) and not all([args.probe_image, args.network, args.endpoint]):
        parser.error("Provide --probe-image, --network and --endpoint together")
    report = inventory(args.old_run)
    write(args.output / "resource_inventory.json", report)
    if report["issues"]:
        raise SystemExit("Resources missing/changed; inspect resource_inventory.json. No downloads attempted.")
    if args.probe_image:
        try:
            result = probe_network(args.probe_image, args.network, args.endpoint)
        except Exception as exc:
            write(args.output / "network_probe.json", {"status": "FAIL", "reason": str(exc)})
            raise
        write(args.output / "network_probe.json", result)
    print("Resource inventory complete. Formal evaluation is NOT started; follow docs/HANDOFF.md reevaluation addendum.")


if __name__ == "__main__":
    main()
