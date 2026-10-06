"""Tests for the export= and raw= options of src.utils.synthetic."""

import pandas as pd

from src.analysis.questionnaire_report import _load_items, cronbach_alpha
from src.data import crop_manifest, ingest
from src.data.labeler import LabelLoader
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

N = 8


def _root(tmp_path, use_root, **kw):
    return use_root(make_synthetic_root(tmp_path / "root", n_participants=N, **kw))


def test_export_reproduces_labels_through_the_labeler(tmp_path, use_root):
    root = _root(tmp_path, use_root, export=True)
    expected = pd.read_csv(
        root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    loaded = LabelLoader.load()
    assert list(loaded["participant_id"]) == list(expected["participant_id"])
    assert list(loaded["label"]) == list(expected["label"])
    assert (loaded["total_score"] == expected["total_score"]).all()
    LabelLoader.write(loaded)
    rewritten = pd.read_csv(
        root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    assert rewritten.equals(expected)
    assert cronbach_alpha(_load_items(loaded["participant_id"])) > 0


def test_raw_crops_and_scans_pass_ingest_and_crop_verification(tmp_path, use_root):
    _root(tmp_path, use_root, raw=True, signal=True)
    scans = ingest.scan_raw(config.paths.raw3_dir, config.paths.raw4_dir)
    assert ingest.validate_scans(scans) == []
    assert (scans["kind"] == "crop").sum() == 24 * N
    manifest = crop_manifest.build_manifest()
    assert len(manifest) == 24 * N
    assert crop_manifest.verify(manifest) == []
    assert crop_manifest._propose_qc_status(manifest) == set(manifest["participant_id"])


def test_new_options_leave_existing_outputs_unchanged(tmp_path):
    plain = make_synthetic_root(tmp_path / "a", n_participants=N, signal=True)
    extra = make_synthetic_root(
        tmp_path / "b", n_participants=N, signal=True, export=True, raw=True
    )
    for rel in ("metadata/labels.csv", "results/features/handcrafted_participant.csv"):
        assert (plain / rel).read_bytes() == (extra / rel).read_bytes()
