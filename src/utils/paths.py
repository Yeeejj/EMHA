"""
Manifest path resolution across data-root mounts.

Packaged manifests (src.utils.package_dataset) store absolute paths under
PackageConfig.mount_root. Kaggle may mount the same dataset elsewhere (e.g.
/kaggle/input/datasets/<user>/<slug>), so a path that does not exist but
starts with mount_root is re-rooted at PathsConfig.data_root. Existing paths,
local relative paths and anything outside mount_root are returned unchanged.
read_manifest is the one loader every module uses for manifests whose path
columns (PackageConfig.path_columns) are opened.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from src.utils.config import config


def resolve_data_path(path, cfg=config) -> str:
    """path if it exists; else re-rooted from mount_root to data_root."""
    text = str(path)
    if Path(text).exists():
        return text
    mount = PurePosixPath(cfg.package.mount_root)
    posix = PurePosixPath(text.replace("\\", "/"))
    try:
        rel = posix.relative_to(mount)
    except ValueError:
        return text
    return str(Path(cfg.paths.data_root).joinpath(*rel.parts))


def read_manifest(path, cfg=config) -> pd.DataFrame:
    """Read a manifest CSV and resolve its path columns (see module docstring)."""
    frame = pd.read_csv(path, dtype={"participant_id": str})
    for col in cfg.package.path_columns:
        if col in frame.columns:
            frame[col] = [
                p if pd.isna(p) else resolve_data_path(p, cfg) for p in frame[col]
            ]
    return frame
