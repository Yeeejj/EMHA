"""scripts/hash_tree.py: snapshot and compare of an output tree."""

from scripts.hash_tree import compare, main, read, snapshot, write

TOPS = ["results", "models"]
ALLOWED = ["results/per_task_v3/", "models/cnn_v3/"]


def _tree(root):
    (root / "results" / "final").mkdir(parents=True)
    (root / "models" / "cnn").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "results" / "final" / "RESULTS.txt").write_text("v1 results\n")
    (root / "results" / "final" / "preds.csv").write_text("a,b\n1,2\n")
    (root / "models" / "cnn" / "fold_0.pt").write_bytes(b"\x00\x01\x02")
    (root / "logs" / "run.log").write_text("outside the tops\n")
    return root


def _snap(tmp_path):
    root = _tree(tmp_path / "out")
    before = tmp_path / "before.txt"
    assert main(["snapshot", str(root), "--tops", *TOPS, "--out", str(before)]) == 0
    return root, before


def _compare(root, before):
    argv = ["compare", str(root), "--before", str(before), "--tops", *TOPS]
    return main([*argv, "--allow-new", *ALLOWED])


def test_snapshot_read_round_trip(tmp_path):
    root = _tree(tmp_path / "out")
    rows = snapshot(root, TOPS)
    assert [r[0] for r in rows] == [
        "models/cnn/fold_0.pt",
        "results/final/RESULTS.txt",
        "results/final/preds.csv",
    ]
    assert rows[0][2] == 3
    out = tmp_path / "snap.txt"
    write(rows, out, "line one\nline two")
    text = out.read_text().splitlines()
    assert text[:2] == ["# line one", "# line two"]
    assert read(out) == {rel: sha for rel, sha, _size in rows}


def test_unchanged_tree_passes(tmp_path, capsys):
    root, before = _snap(tmp_path)
    assert _compare(root, before) == 0
    assert "OK" in capsys.readouterr().out


def test_changed_byte_is_changed(tmp_path):
    root, before = _snap(tmp_path)
    (root / "models" / "cnn" / "fold_0.pt").write_bytes(b"\x00\x01\x03")
    diff = compare(root, before, TOPS, ALLOWED)
    assert diff["changed"] == ["models/cnn/fold_0.pt"]
    assert _compare(root, before) == 1


def test_deleted_file_is_missing(tmp_path):
    root, before = _snap(tmp_path)
    (root / "results" / "final" / "preds.csv").unlink()
    diff = compare(root, before, TOPS, ALLOWED)
    assert diff["missing"] == ["results/final/preds.csv"]
    assert diff["changed"] == []
    assert _compare(root, before) == 1


def test_new_file_under_allowed_prefix_passes(tmp_path):
    root, before = _snap(tmp_path)
    (root / "results" / "per_task_v3").mkdir()
    (root / "results" / "per_task_v3" / "preds.csv").write_text("x\n")
    diff = compare(root, before, TOPS, ALLOWED)
    assert diff == {
        "changed": [],
        "missing": [],
        "new_outside_allowed": [],
        "new_allowed": ["results/per_task_v3/preds.csv"],
    }
    assert _compare(root, before) == 0


def test_new_file_elsewhere_fails(tmp_path, capsys):
    root, before = _snap(tmp_path)
    (root / "results" / "final" / "extra.csv").write_text("x\n")
    (root / "logs" / "other.log").write_text("not under a top\n")
    diff = compare(root, before, TOPS, ALLOWED)
    assert diff["new_outside_allowed"] == ["results/final/extra.csv"]
    assert _compare(root, before) == 1
    assert "new_outside_allowed: results/final/extra.csv" in capsys.readouterr().out


def test_snapshot_never_overwrites(tmp_path):
    root, before = _snap(tmp_path)
    original = before.read_text()
    (root / "results" / "final" / "extra.csv").write_text("x\n")
    assert main(["snapshot", str(root), "--tops", *TOPS, "--out", str(before)]) == 1
    assert before.read_text() == original
