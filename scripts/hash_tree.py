"""
Snapshot and compare SHA-256 checksums of an output tree (Deviation 6, L14).

snapshot(root, tops) hashes every file under root/<top> for each top and
returns (relative posix path, sha256, size) rows sorted by path. write()
stores them as "sha256  size  path" lines after "# " header lines; read()
loads such a file back as {path: sha256}. compare() re-snapshots the tree
and sorts each difference into changed, missing, new_outside_allowed and
new_allowed, so the before/after check around v3 is the same code both
times. The tree is only read; an existing snapshot file is never overwritten.

    python -m scripts.hash_tree snapshot ROOT --tops results models --out FILE
    python -m scripts.hash_tree compare ROOT --before FILE --tops results models
        [--allow-new PREFIX ...]

compare exits 1 if any file changed, went missing or appeared outside the
allowed prefixes, else 0.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

_HASH_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(root: Path, tops: list[str]) -> list[tuple[str, str, int]]:
    """Return (relative posix path, sha256, size) for every file under the tops."""
    root = Path(root)
    rows = []
    for top in tops:
        base = root / top
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                rows.append((rel, _sha256(path), path.stat().st_size))
    return sorted(rows)


def write(rows: list[tuple[str, str, int]], out: Path, header: str) -> None:
    """Write "# " header lines, then one "sha256  size  path" line per file."""
    lines = [f"# {line}" for line in header.splitlines()]
    lines += [f"{sha}  {size}  {rel}" for rel, sha, size in rows]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read(path: Path) -> dict[str, str]:
    """Load a snapshot file as {relative path: sha256}."""
    hashes = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, _size, rel = line.split(None, 2)
        hashes[rel] = sha
    return hashes


def compare(
    root: Path,
    before_file: Path,
    tops: list[str],
    allowed_new_prefixes: list[str],
) -> dict[str, list[str]]:
    """Compare the tree now against a snapshot; return the sorted differences."""
    before = {
        rel: sha
        for rel, sha in read(before_file).items()
        if rel.split("/", 1)[0] in tops
    }
    now = {rel: sha for rel, sha, _size in snapshot(root, tops)}
    new = sorted(set(now) - set(before))
    allowed = [p for p in new if p.startswith(tuple(allowed_new_prefixes))]
    return {
        "changed": sorted(p for p in before if p in now and now[p] != before[p]),
        "missing": sorted(set(before) - set(now)),
        "new_outside_allowed": [p for p in new if p not in allowed],
        "new_allowed": allowed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="hash the tree and write a snapshot")
    snap.add_argument("root", type=Path)
    snap.add_argument("--tops", nargs="+", required=True)
    snap.add_argument("--out", type=Path, required=True)
    comp = sub.add_parser("compare", help="compare the tree against a snapshot")
    comp.add_argument("root", type=Path)
    comp.add_argument("--before", type=Path, required=True)
    comp.add_argument("--tops", nargs="+", required=True)
    comp.add_argument("--allow-new", nargs="*", default=[])
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        if args.out.exists():
            print(f"ERROR: {args.out} already exists; snapshots are never overwritten")
            return 1
        rows = snapshot(args.root, args.tops)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        header = (
            f"hash_tree snapshot of {args.root.as_posix()}\n"
            f"tops: {' '.join(args.tops)}\n"
            f"utc: {stamp}\n"
            f"files: {len(rows)}"
        )
        write(rows, args.out, header)
        print(f"wrote {len(rows)} checksums to {args.out}")
        return 0

    if not args.before.is_file():
        print(f"ERROR: {args.before} not found")
        return 1
    diff = compare(args.root, args.before, args.tops, args.allow_new)
    for key, paths in diff.items():
        print(f"{key}: {len(paths)}")
    for key in ("changed", "missing", "new_outside_allowed"):
        for path in diff[key]:
            print(f"  {key}: {path}")
    failed = any(diff[k] for k in ("changed", "missing", "new_outside_allowed"))
    print("FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
