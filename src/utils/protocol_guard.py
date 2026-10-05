"""
Refuse to produce model results on real data before the protocol is frozen.

CLAUDE.md Non-Negotiable 8 and EVALUATION_PROTOCOL.md: no model may be run on
real participants' data before the git tag `protocol-frozen` exists. Every
Stage F/G runner calls require_frozen_or_synthetic() before loading data.

A data root counts as synthetic when it contains SYNTHETIC_MARKER, which
only src/utils/synthetic.py writes. Synthetic roots always pass.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROTOCOL_TAG = "protocol-frozen"
SYNTHETIC_MARKER = ".synthetic_root"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ProtocolNotFrozenError(RuntimeError):
    """A model was about to run on real data before the protocol tag."""


def is_synthetic_root(data_root) -> bool:
    return (Path(data_root) / SYNTHETIC_MARKER).is_file()


def protocol_frozen(repo_root: Path = _REPO_ROOT) -> bool:
    """True if git reports the protocol-frozen tag in repo_root."""
    try:
        out = subprocess.run(
            ["git", "tag", "--list", PROTOCOL_TAG],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return out.stdout.strip() == PROTOCOL_TAG


def require_frozen_or_synthetic(cfg, what: str) -> None:
    """Raise ProtocolNotFrozenError unless the data root is synthetic or frozen."""
    if is_synthetic_root(cfg.paths.data_root):
        return
    if protocol_frozen():
        return
    raise ProtocolNotFrozenError(
        f"{what} would run on real data ({cfg.paths.data_root}) but the git tag "
        f"'{PROTOCOL_TAG}' does not exist. Freeze EVALUATION_PROTOCOL.md first "
        "(CLAUDE.md Non-Negotiable 8), or point EMHA_DATA_ROOT at a synthetic "
        "root (src/utils/synthetic.py)."
    )
