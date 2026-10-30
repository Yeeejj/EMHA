"""scripts.drive_check against an in-memory fake of the Drive v3 API."""

import re
import sys
import types
from pathlib import Path

import pytest

from scripts import drive_check

REL = "EMHA/run_outputs/seed_42"


class _Media:
    def __init__(self, path, mimetype, resumable):
        self.path = path


class _Req:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class _Files:
    def __init__(self, rows, page):
        self.rows, self.page = rows, page

    def list(self, q, fields, pageSize=100, pageToken=None):
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        name = re.search(r"name = '([^']+)'", q)
        rows = [r for r in self.rows if r["parent"] == parent]
        if name:
            rows = [r for r in rows if r["name"] == name.group(1) and r["folder"]]
        start = int(pageToken or 0)
        out = {
            "files": [
                {"id": r["id"], "name": r["name"], "size": str(r["size"])}
                for r in rows[start : start + self.page]
            ]
        }
        if start + self.page < len(rows):
            out["nextPageToken"] = str(start + self.page)
        return _Req(lambda: out)

    def create(self, body, media_body, fields):
        def run():
            self.rows.append(
                {
                    "id": f"new{len(self.rows)}",
                    "name": body["name"],
                    "parent": body["parents"][0],
                    "size": Path(media_body.path).stat().st_size,
                    "folder": False,
                }
            )
            return {"id": "x"}

        return _Req(run)


class _Svc:
    def __init__(self, rows, page=2):
        self._files = _Files(rows, page)

    def files(self):
        return self._files


@pytest.fixture(autouse=True)
def _fake_googleapiclient(monkeypatch):
    http = types.ModuleType("googleapiclient.http")
    http.MediaFileUpload = _Media
    package = types.ModuleType("googleapiclient")
    monkeypatch.setitem(sys.modules, "googleapiclient", package)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http)


def _tree(draws=()):
    """Fake Drive holding EMHA/run_outputs/seed_42 with the given draw files."""
    rows, parent = [], "root"
    for i, part in enumerate(REL.split("/")):
        rows.append(_row(f"f{i}", part, parent, 0, True))
        parent = f"f{i}"
    for name, size in draws:
        rows.append(_row(f"p{len(rows)}", name, parent, size, False))
    return rows


def _row(id_, name, parent, size, folder):
    return {"id": id_, "name": name, "parent": parent, "size": size, "folder": folder}


@pytest.fixture
def staging(tmp_path):
    for k in (5, 6, 8):
        (tmp_path / f"perm_{k:04d}.csv").write_text("x" * (100 + k))
    (tmp_path / "run_config.json").write_text("{}")
    return tmp_path


def test_missing_draws_are_uploaded_and_counted(staging):
    rows = _tree((f"perm_{k:04d}.csv", 9) for k in range(5))
    lines, ok = drive_check.check_draws(_Svc(rows), staging, REL, 20)
    assert ok
    assert [s for s in lines if "uploaded" in s] == [
        "drive server: uploaded perm_0005.csv (105 bytes)",
        "drive server: uploaded perm_0006.csv (106 bytes)",
        "drive server: uploaded perm_0008.csv (108 bytes)",
    ]
    assert lines[-1] == f"drive server: 8/20 present in {REL}"


def test_second_check_uploads_nothing(staging):
    rows = _tree()
    drive_check.check_draws(_Svc(rows), staging, REL, 20)
    lines, ok = drive_check.check_draws(_Svc(rows), staging, REL, 20)
    assert ok and not any("uploaded" in s for s in lines)
    assert lines[-1] == f"drive server: 3/20 present in {REL}"


def test_wrong_size_on_server_fails(staging):
    rows = _tree([("perm_0005.csv", 1)])
    lines, ok = drive_check.check_draws(_Svc(rows), staging, REL, 20)
    assert not ok and "['perm_0005.csv']" in lines[-1]


def test_duplicate_draw_on_server_fails(staging):
    rows = _tree([("perm_0001.csv", 9), ("perm_0001.csv", 9)])
    lines, ok = drive_check.check_draws(_Svc(rows), staging, REL, 20)
    assert not ok and "['perm_0001.csv']" in lines[-1]


def test_non_draw_files_are_ignored(staging):
    rows = _tree([("run_config.json", 2), ("perm_0009.tmp", 3)])
    lines, ok = drive_check.check_draws(_Svc(rows), staging, REL, 20)
    assert ok and lines[-1] == f"drive server: 3/20 present in {REL}"


def test_missing_folder_raises(staging):
    with pytest.raises(RuntimeError, match="0 folders named 'nope'"):
        drive_check.check_draws(_Svc(_tree()), staging, "EMHA/nope", 20)


def _confirm(rows, path, **kw):
    naps = []
    out = drive_check.confirm_draw(
        _Svc(rows), "f2", path, wait_s=60, poll_s=15, sleep=naps.append, **kw
    )
    return out, naps


def test_confirm_draw_already_uploaded(staging):
    rows = _tree([("perm_0005.csv", 105)])
    (line, ok), naps = _confirm(rows, staging / "perm_0005.csv")
    assert ok and "confirmed" in line and naps == []
    assert len([r for r in rows if r["name"] == "perm_0005.csv"]) == 1


def test_confirm_draw_uploads_after_waiting(staging):
    rows = _tree()
    (line, ok), naps = _confirm(rows, staging / "perm_0005.csv")
    assert ok and "uploaded through the API (105 bytes)" in line
    assert naps == [15, 15, 15, 15]
    assert [r["size"] for r in rows if r["name"] == "perm_0005.csv"] == [105]


def test_confirm_draw_sees_a_late_mount_upload(staging):
    rows = _tree()

    def nap(_):
        if not any(r["name"] == "perm_0005.csv" for r in rows):
            rows.append(_row("late", "perm_0005.csv", "f2", 105, False))

    line, ok = drive_check.confirm_draw(
        _Svc(rows), "f2", staging / "perm_0005.csv", wait_s=60, poll_s=15, sleep=nap
    )
    assert ok and "confirmed" in line
    assert len([r for r in rows if r["name"] == "perm_0005.csv"]) == 1


def test_confirm_draw_never_overwrites_a_wrong_size(staging):
    rows = _tree([("perm_0005.csv", 7)])
    (line, ok), _ = _confirm(rows, staging / "perm_0005.csv")
    assert not ok and "MISMATCH" in line
    assert [r["size"] for r in rows if r["name"] == "perm_0005.csv"] == [7]


def test_drive_upload_names_the_file(tmp_path):
    rows = _tree()
    path = tmp_path / "v3_draws_drive_check.log"
    path.write_text("ok\n")
    drive_check.drive_upload(_Svc(rows), "f2", path, "text/plain")
    assert rows[-1]["name"] == path.name and rows[-1]["parent"] == "f2"
