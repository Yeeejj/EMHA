"""
Private Kaggle dataset upload (slug emha-crops) — Stage G.

Builds <out_dir>/<slug>/ (default dist/emha-crops/) with DATASET's layout,
so EMHA_DATA_ROOT=/kaggle/input/emha-crops works with no code changes:

    raw-3Page/D1..D4/<crop>.png        raw crops (PackageConfig.include_crops)
    raw-4Page/<cell>/<crop>.png        (handcrafted features need them)
    processed/<participant_id>/<cell>.png
    metadata/labels.csv participants.csv folds.csv      byte-identical
    metadata/crop_manifest.csv processed_manifest.csv qc_log.csv
                                       filtered to the packaged participants,
                                       path columns rewritten under
                                       PackageConfig.mount_root
    dataset-metadata.json              Kaggle CLI metadata (title, id, license)
    MANIFEST.txt                       file counts, SHA-256 of folds/labels,
                                       upload command

Only files listed in crop_manifest.csv / processed_manifest.csv are copied,
each checked against its manifest SHA-256 (raw crops). Never copied: page
scans, the tabulation export (questionnaire_export.csv), raw_manifest.csv,
anything under src/tabulation. The build fails on any item-level column
(ReportConfig.item_columns) or name / email / notes column; an all-blank
notes column is dropped instead. It also fails unless every packaged
participant is qc_passed with all 24 raw crops and every non-dropped
processed crop on disk, and if a labeled participant still awaits QC.
audit_package re-checks the finished folder. Sources are only read; an
existing output folder is never overwritten.

With --subset pilot only pilot participants' crops and manifest rows are
packaged; labels/participants/folds stay byte-identical so their SHA-256
match the frozen protocol (runners take --subset pilot).

Upload (private by default):

    kaggle datasets create -p dist/emha-crops -r zip

    python -m src.utils.package_dataset [--subset pilot] [--out-dir dist]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.data.crop_manifest import ALL_CELLS
from src.preprocessing.pipeline import _load_qc_dropped
from src.training.splits import restrict_to_subset
from src.utils.config import _REPO_ROOT, config

QC_PASSED, QC_FAILED = "qc_passed", "qc_failed"
METADATA_JSON, MANIFEST_TXT = "dataset-metadata.json", "MANIFEST.txt"
CROP_TREES = ("raw-3Page", "raw-4Page")


class PackageError(RuntimeError):
    """The package would be incomplete or would contain forbidden data."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── checks ───────────────────────────────────────────────────────────────────


def forbidden_columns(columns, cfg=config) -> list:
    """Item-level, name-, email- or notes-like column names."""
    items = {c.lower() for c in cfg.report.item_columns}
    patterns = cfg.package.forbidden_column_patterns
    return [
        c
        for c in columns
        if str(c).lower() in items or any(p in str(c).lower() for p in patterns)
    ]


def clean_columns(df: pd.DataFrame, name: str, cfg=config) -> pd.DataFrame:
    """Drop all-blank droppable columns; raise on any forbidden column left."""
    for col in cfg.package.droppable_blank_columns:
        if col in df.columns:
            text = df[col].fillna("").astype(str).str.strip()
            if (text != "").any():
                rows = df.loc[text != "", "participant_id"].astype(str).tolist()
                raise PackageError(
                    f"{name}: column {col!r} has text for {rows[:10]}; free text "
                    "may identify a participant, so it is never packaged"
                )
            df = df.drop(columns=col)
    bad = forbidden_columns(df.columns, cfg)
    if bad:
        raise PackageError(f"{name}: forbidden columns {bad} (item / name / email)")
    return df


def _check_source(path: Path, cfg=config) -> None:
    """Refuse the tabulation export, other never-copy files, src/tabulation."""
    path = Path(path).resolve()
    tab = (_REPO_ROOT / cfg.package.tabulation_dir).resolve()
    if path.name in cfg.package.never_copy or tab in path.parents:
        raise PackageError(f"{path} must never be packaged")


def _mount_path(path: str, cfg=config) -> str:
    """A manifest path (any OS, repo-relative or absolute) under mount_root."""
    p = Path(str(path).replace("\\", "/")).resolve()
    root = Path(cfg.paths.data_root).resolve()
    try:
        rel = p.relative_to(root)
    except ValueError as exc:
        raise PackageError(f"{path} is outside the data root {root}") from exc
    return f"{cfg.package.mount_root.rstrip('/')}/{rel.as_posix()}"


def _local(path: str) -> Path:
    return Path(str(path).replace("\\", "/"))


# ── participants ─────────────────────────────────────────────────────────────


def package_ids(subset: str | None, cfg=config) -> list:
    """qc_passed labeled participants (in the subset); refuse pending QC."""
    meta = cfg.paths.metadata_dir
    labels = pd.read_csv(meta / "labels.csv", dtype={"participant_id": str})
    parts = pd.read_csv(meta / "participants.csv", dtype={"participant_id": str})
    labeled = restrict_to_subset(sorted(labels["participant_id"]), subset, cfg)
    status = parts.set_index("participant_id")["status"].reindex(labeled)
    pending = sorted(status[~status.isin([QC_PASSED, QC_FAILED])].index)
    if pending:
        raise PackageError(
            f"{len(pending)} of {len(labeled)} labeled participants are not "
            f"through QC (e.g. {pending[:5]}); run python -m "
            "src.data.crop_manifest --apply-qc-status first"
        )
    ids = sorted(status[status == QC_PASSED].index)
    if not ids:
        raise PackageError(f"no qc_passed labeled participants (subset {subset})")
    return ids


def _crop_rows(ids: list, cfg=config) -> tuple:
    """(crop_manifest rows, processed_manifest rows) for ids; refuse gaps."""
    meta = cfg.paths.metadata_dir
    crops = pd.read_csv(meta / "crop_manifest.csv", dtype={"participant_id": str})
    proc = pd.read_csv(meta / "processed_manifest.csv", dtype={"participant_id": str})
    dropped = _load_qc_dropped(meta)
    crops = crops[crops["participant_id"].isin(ids)]
    proc = proc[proc["participant_id"].isin(ids)]

    problems = []
    for pid, part in crops.groupby("participant_id"):
        if set(part["cell"]) != set(ALL_CELLS):
            problems.append(f"{pid}: crop_manifest lacks 24 cells")
    missing_ids = set(ids) - set(crops["participant_id"])
    problems += [f"{pid}: no crop_manifest rows" for pid in sorted(missing_ids)]
    for pid in ids:
        need = {c for c in ALL_CELLS if (pid, c) not in dropped}
        have = set(proc.loc[proc["participant_id"] == pid, "cell"])
        if need - have:
            problems.append(
                f"{pid}: {len(need - have)} processed crops not in manifest"
            )
    for col, frame in (("path", crops), ("processed_path", proc)):
        gone = [p for p in frame[col] if not _local(p).is_file()]
        problems += [f"missing file {p}" for p in gone[:20]]
    if problems:
        raise PackageError(
            f"{len(problems)} completeness problem(s), e.g. {problems[:5]}; run "
            "python -m src.preprocessing.pipeline for missing processed crops"
        )
    return crops, proc


# ── build ────────────────────────────────────────────────────────────────────


def kaggle_username(cfg=config) -> str:
    """PackageConfig.kaggle_user, else KAGGLE_USERNAME, else kaggle.json."""
    if cfg.package.kaggle_user:
        return cfg.package.kaggle_user
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cred_dir = os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle"
    cred = Path(cred_dir) / "kaggle.json"
    if cred.is_file():
        user = json.loads(cred.read_text()).get("username")  # never the key
        if user:
            return user
    raise PackageError(
        "Kaggle username not found: set KAGGLE_USERNAME, add ~/.kaggle/kaggle.json, "
        "or set PackageConfig.kaggle_user"
    )


def _copy(src: Path, dest: Path, expected_sha: str | None = None) -> None:
    _check_source(src)
    if expected_sha is not None and sha256_of(src) != expected_sha:
        raise PackageError(f"{src} differs from its crop_manifest SHA-256")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)  # read-only on the source


def _rel(path: str, cfg=config) -> Path:
    """Data-root-relative location of a manifest path inside the package."""
    return Path(_mount_path(path, cfg)[len(cfg.package.mount_root) :].lstrip("/"))


def _write_metadata(pkg: Path, ids: list, crops, proc, cfg=config) -> None:
    meta_src, meta_dst = cfg.paths.metadata_dir, pkg / "metadata"
    meta_dst.mkdir(parents=True)
    for name in cfg.package.metadata_files:
        src = meta_src / name
        if not src.is_file():
            raise PackageError(f"{src} not found")
        _check_source(src)
        frame = pd.read_csv(src, dtype={"participant_id": str})
        if name in cfg.package.verbatim_files:
            clean_columns(frame, name, cfg)  # check only; copy byte-identical
            shutil.copyfile(src, meta_dst / name)
            continue
        frame = {"crop_manifest.csv": crops, "processed_manifest.csv": proc}.get(
            name, frame[frame["participant_id"].isin(ids)]
        )
        frame = clean_columns(frame.copy(), name, cfg)
        if not cfg.package.include_crops and name != "processed_manifest.csv":
            frame = frame.drop(columns="path")
        for col in cfg.package.path_columns:
            if col in frame.columns:
                frame[col] = [_mount_path(p, cfg) for p in frame[col]]
        frame.to_csv(meta_dst / name, index=False)


def _manifest_text(built: Path, final: Path, subset, ids, user, cfg=config) -> str:
    """MANIFEST.txt from the files in `built`, naming the final folder."""
    counts = {
        top: sum(1 for p in (built / top).rglob("*") if p.is_file())
        for top in (*CROP_TREES, "processed", "metadata")
        if (built / top).is_dir()
    }
    lines = [
        f"Kaggle dataset : {user}/{cfg.package.slug} (private)",
        f"Subset         : {subset or 'all'}",
        f"Participants   : {len(ids)}",
        f"Mount root     : {cfg.package.mount_root} (EMHA_DATA_ROOT)",
        "Files:",
        *[f"  {top:12s} {n}" for top, n in counts.items()],
        f"  {'total':12s} {sum(counts.values())}",
        "SHA-256:",
        *[
            f"  metadata/{n:16s} {sha256_of(built / 'metadata' / n)}"
            for n in ("folds.csv", "labels.csv")
        ],
        "Excluded: page scans, questionnaire_export.csv (item-level answers),",
        "  raw_manifest.csv, src/tabulation, name/email/notes columns.",
        "Upload (private unless --public is added):",
        f"  kaggle datasets create -p {final.as_posix()} -r {cfg.package.dir_mode}",
    ]
    return "\n".join(lines) + "\n"


def build_package(subset: str | None, out_dir: Path) -> Path:
    """Write <out_dir>/<slug>/ and return it. Never overwrites an existing one."""
    cfg = config
    pkg = Path(out_dir) / cfg.package.slug
    if pkg.exists():
        raise FileExistsError(
            f"{pkg} already exists; move or delete it yourself to rebuild"
        )
    user = kaggle_username(cfg)
    ids = package_ids(subset, cfg)
    crops, proc = _crop_rows(ids, cfg)

    tmp = Path(out_dir) / f".{cfg.package.slug}.partial"
    if tmp.exists():
        raise FileExistsError(
            f"{tmp} is left from an interrupted build; delete it yourself first"
        )
    try:
        if cfg.package.include_crops:
            for src, sha in zip(crops["path"], crops["sha256"]):
                _copy(_local(src), tmp / _rel(src, cfg), sha)
        for src in proc["processed_path"]:
            _copy(_local(src), tmp / _rel(src, cfg))
        _write_metadata(tmp, ids, crops, proc, cfg)
        (tmp / METADATA_JSON).write_text(
            json.dumps(
                {
                    "title": cfg.package.title,
                    "id": f"{user}/{cfg.package.slug}",
                    "licenses": [{"name": cfg.package.license}],
                },
                indent=2,
            )
        )
        (tmp / MANIFEST_TXT).write_text(
            _manifest_text(tmp, pkg, subset, ids, user, cfg)
        )
        audit_package(tmp, cfg)
        tmp.rename(pkg)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return pkg


def audit_package(pkg: Path, cfg=config) -> None:
    """Re-check a built folder: layout, no scans, no forbidden data."""
    pkg = Path(pkg)
    cells = set(ALL_CELLS)
    for path in (p for p in pkg.rglob("*") if p.is_file()):
        rel = path.relative_to(pkg)
        top = rel.parts[0]
        if path.name in cfg.package.never_copy:
            raise PackageError(f"{rel} must never be packaged")
        if len(rel.parts) == 1:
            ok = rel.name in (METADATA_JSON, MANIFEST_TXT)
        elif top == "metadata":
            ok = len(rel.parts) == 2 and rel.name in cfg.package.metadata_files
            if ok and forbidden_columns(pd.read_csv(path, nrows=0).columns, cfg):
                raise PackageError(f"{rel} has forbidden columns")
        elif top in CROP_TREES:
            # crops only: <tree>/<cell>/EMHA-..._<cell>.png; scans sit at the top
            ok = (
                len(rel.parts) == 3
                and rel.parts[1] in cells
                and rel.stem.endswith(f"_{rel.parts[1]}")
            )
        elif top == "processed":
            ok = len(rel.parts) == 3 and rel.stem in cells
        else:
            ok = False
        if not ok:
            raise PackageError(f"unexpected file in package: {rel}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the private Kaggle upload.")
    parser.add_argument("--subset", choices=("pilot",), default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=_REPO_ROOT / config.package.out_dir
    )
    args = parser.parse_args()
    try:
        pkg = build_package(args.subset, args.out_dir)
    except (PackageError, FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print((pkg / MANIFEST_TXT).read_text(), end="")
    print(f"Package written: {pkg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
