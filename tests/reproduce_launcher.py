"""Test-only launcher for scripts/reproduce_full.*: `-m module args` under the
smoke-test config (tiny crops, 1 epoch, small HMM grid) plus a few
permutations and bootstrap resamples, so a full reproduction runs in minutes.

Used through EMHA_PYTHON_PREFIX by tests/test_reproduce.py. Refuses any data
root that is not a synthetic fixture, so it can never shrink a real run.
"""

import contextlib
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import smoke_test  # noqa: E402
from src.utils.config import config  # noqa: E402
from src.utils.protocol_guard import is_synthetic_root  # noqa: E402

TEST_SIZES = (
    (config.permutation, "n_lr", 3),
    (config.permutation, "n_cnn", 1),
    (config.evaluation, "n_bootstrap", 200),
    (config.report, "n_bootstrap", 50),
    (config.gradcam, "n_examples", 2),
)


def main(argv: list) -> None:
    if len(argv) < 2 or argv[0] != "-m":
        raise SystemExit("usage: reproduce_launcher.py -m <module> [args...]")
    root = Path(os.environ.get("EMHA_DATA_ROOT", ""))
    if not is_synthetic_root(root):
        raise SystemExit(f"refusing: {root} is not a synthetic data root")
    module, rest = argv[1], argv[2:]
    with contextlib.ExitStack() as stack:
        smoke_test._overrides(stack, root)
        for obj, name, value in TEST_SIZES:
            stack.enter_context(patch.object(obj, name, value))
        sys.argv = [module, *rest]
        runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
