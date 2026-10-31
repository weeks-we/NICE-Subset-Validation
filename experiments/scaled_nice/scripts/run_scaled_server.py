"""Independent four-GPU experiment with an overridable smaller code model."""

from pathlib import Path
import sys

from run_server import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    defaults = {
        "--num-gpus": "4",
        "--base-config": str(root / "configs/scaled.yaml"),
        "--config-dir": str(root / "configs/scaled-server"),
        "--port": "8875",
    }
    for option, value in defaults.items():
        if not any(arg == option or arg.startswith(option + "=") for arg in sys.argv):
            sys.argv.extend([option, value])
    main()
