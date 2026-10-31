from pathlib import Path
import os
import sys
import uuid

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Arrow encodes absolute cache paths into Windows lock filenames. Keep test-only
# artifacts short inside the enclosing workspace when this project is nested.
project_root = Path(__file__).resolve().parents[1]
workspace_root = (
    project_root.parents[1]
    if project_root.parent.name == "experiments"
    else project_root
)
TEST_RUNS = workspace_root / "runs" / "s2"


@pytest.fixture
def tmp_path():
    # This managed Windows sandbox denies pytest's mode=0700 temporary dirs.
    # Inherit the workspace ACL and retain evidence under the ignored runs/ tree.
    path = TEST_RUNS / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    return path.resolve()
