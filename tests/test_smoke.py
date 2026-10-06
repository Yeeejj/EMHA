"""pytest wrapper for smoke_test.py: stages A-C (the full run is the script)."""

import hashlib

import pytest

import smoke_test
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root


def _hashes(root):
    return {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for d in ("raw-3Page", "raw-4Page")
        for p in sorted((root / d).rglob("*"))
        if p.is_file()
    }


def _snapshot():
    return (
        config.paths.data_root,
        config.paths.output_root,
        dict(config.crop.expected_size_px),
        dict(config.preprocessing.canvas_size),
        config.labeling.min_per_class,
        config.training.epochs,
    )


def test_parse_stages():
    assert smoke_test.parse_stages("A-H") == list("ABCDEFGH")
    assert smoke_test.parse_stages("a-c") == list("ABC")
    assert smoke_test.parse_stages("F") == ["F"]
    assert smoke_test.parse_stages("A,C-D") == list("ACD")
    for bad in ("C-A", "Z", "A-Z"):
        with pytest.raises(ValueError):
            smoke_test.parse_stages(bad)


def test_stages_a_to_c_pass_with_real_sample_read_only(tmp_path, capsys):
    # stand-in for a real data root: raw crops at the real sizes, no marker use
    real = make_synthetic_root(tmp_path / "real", n_participants=10, raw=True)
    before, config_before = _hashes(real), _snapshot()
    assert smoke_test.main("A-C", real_sample=real) == 0
    out = capsys.readouterr().out
    for stage in ("A", "B", "C"):
        assert f"[PASS] {stage} " in out
    assert "real sample ['001', '002', '003']" in out
    assert "ALL STAGES PASSED" in out
    assert _hashes(real) == before
    assert _snapshot() == config_before  # every override restored


def test_stops_at_first_failure(monkeypatch, capsys):
    def broken(ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(smoke_test, "stage_b", broken)
    assert smoke_test.main("A-C") == 1
    out = capsys.readouterr().out
    assert "[FAIL] B" in out and "RuntimeError: boom" in out
    assert "[PASS] C" not in out and "[FAIL] C" not in out
