"""
Merge CNN-path permutation draws cached on another machine (Deviation 3).

Copies every perm_<k>.csv from SRC into DST, where both are runs folders of
the form results/final/permutation_runs/<analysis>/seed_<seed>/. The two
folders' run_config.json must be identical (same analysis, seed, folds SHA-256
and model settings), so draws made under different settings are never mixed.
A draw present in both folders is left as it is in DST and reported; draws
are never overwritten. SRC is only read.

    python -m scripts.merge_draws SRC DST [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CONFIG = "run_config.json"


def merge(src: Path, dst: Path, dry_run: bool = False) -> list[str]:
    """Copy SRC's draws into DST; return the copied file names."""
    src, dst = Path(src), Path(dst)
    src_cfg = src / CONFIG
    if not src_cfg.is_file():
        raise FileNotFoundError(f"{src_cfg} not found; SRC is not a runs folder")
    settings = json.loads(src_cfg.read_text())
    dst_cfg = dst / CONFIG
    if dst_cfg.is_file():
        previous = json.loads(dst_cfg.read_text())
        if previous != settings:
            changed = sorted(
                k
                for k in set(settings) | set(previous)
                if previous.get(k) != settings.get(k)
            )
            raise RuntimeError(
                f"{src} and {dst} were made with different settings ({changed}); "
                "their draws cannot be merged."
            )
    copied, skipped = [], []
    for path in sorted(src.glob("perm_*.csv")):
        target = dst / path.name
        if target.is_file():
            skipped.append(path.name)
            continue
        copied.append(path.name)
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            if not dst_cfg.is_file():
                shutil.copy2(src_cfg, dst_cfg)
            tmp = target.with_suffix(".tmp")
            shutil.copy2(path, tmp)
            tmp.replace(target)
    verb = "would copy" if dry_run else "copied"
    print(f"{verb} {len(copied)} draws: {copied}")
    if skipped:
        print(f"already in {dst}, left unchanged: {skipped}")
    present = sorted(p.name for p in dst.glob("perm_*.csv")) if dst.is_dir() else []
    print(f"{dst} now holds {len(present)} draws")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("src", type=Path, help="runs folder to copy draws from")
    parser.add_argument("dst", type=Path, help="runs folder to copy draws into")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        merge(args.src, args.dst, args.dry_run)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
