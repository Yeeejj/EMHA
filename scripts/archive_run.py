"""
Archive a finished run's results/ and models/ for Google Drive — Stage G.

archive(run_dir, dest) copies <run_dir>/results and <run_dir>/models into
<dest>/<YYYY-MM-DD>_<short commit>/, writes MANIFEST.txt there (one
"<sha256>  <relative path>" line per file, after a commented header with
the full commit and whether the working tree was dirty), and zips that
folder to <dest>/<YYYY-MM-DD>_<short commit>.zip. Sources are only read;
an existing archive folder or zip is never overwritten.

verify_archive(zip_path) re-hashes every file inside the zip against its
MANIFEST.txt and returns the list of mismatches (empty when intact).

    python -m scripts.archive_run [--run-dir .] [--dest archives]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import zipfile
from datetime import date
from pathlib import Path

ARCHIVED_DIRS = ("results", "models")
MANIFEST_NAME = "MANIFEST.txt"
_HASH_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _commit_info(run_dir: Path) -> tuple[str, str, bool]:
    """(full hash, short hash, dirty) of the repo holding run_dir."""
    full = _git(["rev-parse", "HEAD"], run_dir)
    short = _git(["rev-parse", "--short", "HEAD"], run_dir)
    dirty = bool(_git(["status", "--porcelain"], run_dir))
    return full, short, dirty


def _manifest_lines(folder: Path) -> list[str]:
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    return [
        f"{_sha256(p)}  {p.relative_to(folder).as_posix()}"
        for p in files
        if p.name != MANIFEST_NAME
    ]


def archive(run_dir: Path, dest: Path) -> Path:
    """Copy results/ and models/ into a dated, commit-named folder; return its zip."""
    run_dir, dest = Path(run_dir).resolve(), Path(dest).resolve()
    missing = [d for d in ARCHIVED_DIRS if not (run_dir / d).is_dir()]
    if missing:
        raise FileNotFoundError(f"{run_dir} has no {', '.join(missing)}/ to archive")

    full, short, dirty = _commit_info(run_dir)
    name = f"{date.today().isoformat()}_{short}"
    folder, zip_path = dest / name, dest / f"{name}.zip"
    for existing in (folder, zip_path):
        if existing.exists():
            raise FileExistsError(f"{existing} already exists; refusing to overwrite")

    folder.mkdir(parents=True)
    for d in ARCHIVED_DIRS:
        shutil.copytree(run_dir / d, folder / d)

    header = [
        f"# commit {full}",
        f"# working_tree {'dirty' if dirty else 'clean'}",
        f"# created {date.today().isoformat()}",
    ]
    (folder / MANIFEST_NAME).write_text(
        "\n".join(header + _manifest_lines(folder)) + "\n", encoding="utf-8"
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, (Path(name) / p.relative_to(folder)).as_posix())
    return zip_path


def verify_archive(zip_path: Path) -> list[str]:
    """Paths in zip_path whose SHA-256 differs from or is absent in MANIFEST.txt."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        manifest_name = next(n for n in names if n.endswith(f"/{MANIFEST_NAME}"))
        prefix = manifest_name[: -len(MANIFEST_NAME)]
        expected = {}
        for line in zf.read(manifest_name).decode("utf-8").splitlines():
            if line and not line.startswith("#"):
                digest, rel = line.split("  ", 1)
                expected[rel] = digest
        actual = {
            n[len(prefix) :]: hashlib.sha256(zf.read(n)).hexdigest()
            for n in names
            if n != manifest_name
        }
    keys = sorted(set(expected) | set(actual))
    return [k for k in keys if expected.get(k) != actual.get(k)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    parser.add_argument("--dest", type=Path, default=Path("archives"))
    args = parser.parse_args(argv)
    zip_path = archive(args.run_dir, args.dest)
    bad = verify_archive(zip_path)
    if bad:
        raise SystemExit(f"Manifest mismatch in {zip_path}: {bad}")
    print(f"Archived to {zip_path} (manifest verified)")


if __name__ == "__main__":
    main()
