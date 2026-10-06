"""scripts/archive_run.py on a synthetic results/models folder."""

import subprocess
import zipfile

import pytest

from scripts.archive_run import MANIFEST_NAME, archive, verify_archive


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "run"
    (root / "results" / "predictions").mkdir(parents=True)
    (root / "results" / "predictions" / "lr.csv").write_text(
        "analysis,model,feature_set,fold,participant_id,label,prob_sad,pred,"
        "in_middle_band\nprimary,lr,all,0,001,SAD,0.7,SAD,False\n"
    )
    (root / "results" / "RESULTS.txt").write_text("synthetic\n")
    (root / "models").mkdir()
    (root / "models" / "cnn_fold0.pt").write_bytes(bytes(range(256)) * 4)
    for args in (
        ["init", "-q"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q"]
        + ["--allow-empty", "-m", "init"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True)
    return root


def test_zip_manifest_hashes_match(run_dir, tmp_path):
    zip_path = archive(run_dir, tmp_path / "archives")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert zip_path.name.endswith(f"_{commit}.zip")
    assert verify_archive(zip_path) == []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    stem = zip_path.stem
    assert {
        f"{stem}/{MANIFEST_NAME}",
        f"{stem}/results/RESULTS.txt",
        f"{stem}/results/predictions/lr.csv",
        f"{stem}/models/cnn_fold0.pt",
    } <= names


def test_verify_detects_tampering(run_dir, tmp_path):
    zip_path = archive(run_dir, tmp_path / "archives")
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for n in src.namelist():
            data = src.read(n)
            dst.writestr(n, b"changed" if n.endswith("RESULTS.txt") else data)
    assert verify_archive(tampered) == ["results/RESULTS.txt"]


def test_refuses_to_overwrite(run_dir, tmp_path):
    archive(run_dir, tmp_path / "archives")
    with pytest.raises(FileExistsError):
        archive(run_dir, tmp_path / "archives")


def test_requires_results_and_models(tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(FileNotFoundError, match="models"):
        archive(tmp_path, tmp_path / "archives")
