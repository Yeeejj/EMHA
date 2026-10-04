"""
Labeler — Stage B. Reads the tabulation export verbatim and writes
DATASET/metadata/labels.csv.

No code path here computes a label or score from item responses (CLAUDE.md
Non-Negotiable 3): label and total_score are read directly from the
export's label_column/score_column. Only bookkeeping (boundary_distance,
in_middle_band, in_primary_analysis) is derived, and it never changes a
label.

Run from the project root:

    python -m src.data.labeler
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from src.utils.config import config

# Trailing run of digits in an id, e.g. "P001" -> "001", "P7" -> "7".
ID_DIGITS_PATTERN = re.compile(r"(\d+)$")
VALID_ID_PATTERN = re.compile(r"^\d{3}$")

LABELS_FIELDS = [
    "participant_id",
    "total_score",
    "label",
    "boundary_distance",
    "in_middle_band",
    "in_primary_analysis",
]


def _normalize_id(raw: str) -> str:
    """Extract the trailing digit run and zero-pad to 3 digits.

    Handles "P001", "P7", bare "7", etc. uniformly. Returns "" if no
    digits are found.
    """
    match = ID_DIGITS_PATTERN.search(str(raw).strip())
    if match is None:
        return ""
    return match.group(1).zfill(3)


class LabelLoader:
    """Loads, validates, and writes the authoritative label file.

    Labels are never changed: label and total_score are read verbatim from
    the tabulation export's label_column/score_column. Only bookkeeping
    (boundary_distance, in_middle_band, in_primary_analysis) is derived.
    """

    @staticmethod
    def load() -> pd.DataFrame:
        """Read the tabulation export and return participant_id/total_score/label.

        IDs are normalized to bare 3-digit strings; every id that needed
        stripping/padding is reported. Rows with a blank label (not yet
        encoded in the sheet) are excluded, not flagged as an error.
        """
        lbl = config.labeling
        source_csv = Path(
            lbl.source_csv or (config.paths.metadata_dir / "questionnaire_export.csv")
        )

        if not source_csv.is_file():
            print(f"ERROR: tabulation export not found: {source_csv}")
            sys.exit(1)

        raw = pd.read_csv(source_csv, dtype=str, keep_default_na=False)

        rows = []
        skipped_unlabeled = 0
        for _, row in raw.iterrows():
            original_id = str(row.get(lbl.id_column, "")).strip()
            normalized_id = _normalize_id(original_id)
            if not normalized_id:
                print(f"WARNING: no digits found in id {original_id!r}; skipping row.")
                continue
            if normalized_id != original_id:
                print(f"NOTE: id {original_id!r} normalized to {normalized_id!r}.")

            raw_label = str(row.get(lbl.label_column, "")).strip()
            if not raw_label:
                skipped_unlabeled += 1
                continue
            mapped_label = lbl.label_map.get(raw_label, raw_label)

            raw_score = str(row.get(lbl.score_column, "")).strip()
            try:
                total_score = float(raw_score)
            except ValueError:
                total_score = float("nan")

            rows.append(
                {
                    "participant_id": normalized_id,
                    "total_score": total_score,
                    "label": mapped_label,
                }
            )

        if skipped_unlabeled:
            print(f"INFO: {skipped_unlabeled} row(s) have no label yet; excluded.")

        return pd.DataFrame(rows, columns=["participant_id", "total_score", "label"])

    @staticmethod
    def validate(df: pd.DataFrame, participants: pd.DataFrame | None = None) -> list:
        """Return human-readable problems with df; empty list means clean.

        Always checked: duplicate participant_id, malformed participant_id
        (!= ^\\d{3}$), label outside {HAPPY, SAD}, missing (NaN) total_score.

        Checked only if participants is given: ids present in df but absent
        from participants, and participants with at least one crop
        (has_p3_scan or has_p4_scan) but no label.
        """
        errors = []
        ids = df["participant_id"].astype(str)

        for pid in ids[ids.duplicated()].unique():
            errors.append(f"duplicate participant_id: {pid!r}")

        for pid in ids[~ids.str.match(VALID_ID_PATTERN)]:
            errors.append(
                f"invalid participant_id format: {pid!r} (expected ^\\d{{3}}$)"
            )

        bad_labels = df.loc[
            ~df["label"].isin(["HAPPY", "SAD"]), ["participant_id", "label"]
        ]
        for _, row in bad_labels.iterrows():
            errors.append(
                f"label outside {{HAPPY, SAD}} for {row['participant_id']!r}: "
                f"{row['label']!r}"
            )

        missing_score = df.loc[df["total_score"].isna(), "participant_id"]
        for pid in missing_score:
            errors.append(f"missing total_score for participant_id {pid!r}")

        if participants is not None:
            p_ids = set(participants["participant_id"].astype(str))
            label_ids = set(ids)

            for pid in sorted(label_ids - p_ids):
                errors.append(
                    f"participant_id {pid!r} in export but not in participants.csv"
                )

            has_crop = participants["has_p3_scan"] | participants["has_p4_scan"]
            participant_ids = participants["participant_id"].astype(str)
            missing_label = participants.loc[
                has_crop & ~participant_ids.isin(label_ids), "participant_id"
            ]
            for pid in missing_label:
                errors.append(f"participant {pid!r} has crops but no label")

        return errors

    @staticmethod
    def write(df: pd.DataFrame) -> Path:
        """Write labels.csv with derived bookkeeping columns. Labels are never changed.

        boundary_distance    = abs(total_score - cutoff)
        in_middle_band        = True for the middle_band_fraction of
                                 participants with the smallest boundary_distance
        in_primary_analysis   = True for everyone under full_sample;
                                 True for everyone NOT in_middle_band under
                                 extreme_groups
        """
        lbl = config.labeling
        out = df.copy()
        out["boundary_distance"] = (out["total_score"] - lbl.cutoff).abs()

        n_middle = round(len(out) * lbl.middle_band_fraction)
        middle_ids = set(
            out.sort_values("boundary_distance", kind="stable").head(n_middle)[
                "participant_id"
            ]
        )
        out["in_middle_band"] = out["participant_id"].isin(middle_ids)

        if lbl.analysis_design == "full_sample":
            out["in_primary_analysis"] = True
        else:
            out["in_primary_analysis"] = ~out["in_middle_band"]

        out = out[LABELS_FIELDS]

        output_csv = Path(lbl.output_csv or (config.paths.metadata_dir / "labels.csv"))
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        return output_csv


def main() -> int:
    df = LabelLoader.load()

    participants = None
    participants_path = config.paths.metadata_dir / "participants.csv"
    if participants_path.is_file():
        from src.data.collector import ParticipantRegistry

        participants = ParticipantRegistry.load(participants_path)
    else:
        print(
            f"NOTE: {participants_path} not found; skipping participants cross-check."
        )

    errors = LabelLoader.validate(df, participants=participants)
    if errors:
        print(f"Validation problems ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
        return 1

    out_path = LabelLoader.write(df)
    written = pd.read_csv(out_path, dtype={"participant_id": str})

    counts = written["label"].value_counts()
    primary = written[written["in_primary_analysis"]]
    primary_counts = primary["label"].value_counts()

    print(f"Rows written          : {len(written)}")
    print(f"HAPPY / SAD (all)     : {counts.get('HAPPY', 0)} / {counts.get('SAD', 0)}")
    print(
        f"HAPPY / SAD (primary) : {primary_counts.get('HAPPY', 0)} / "
        f"{primary_counts.get('SAD', 0)}"
    )
    print(f"Labels written        : {out_path}")

    lbl = config.labeling
    if primary_counts.get("HAPPY", 0) < lbl.min_per_class:
        print(f"ERROR: HAPPY primary count below min_per_class={lbl.min_per_class}")
        return 1
    if primary_counts.get("SAD", 0) < lbl.min_per_class:
        print(f"ERROR: SAD primary count below min_per_class={lbl.min_per_class}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
