from pathlib import Path
import os
import sys
import uuid

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tmp_path():
    # This managed Windows sandbox denies pytest's mode=0700 temporary dirs.
    # Inherit the workspace ACL and retain evidence under the ignored runs/ tree.
    path = Path("runs/test-work") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path.resolve()
