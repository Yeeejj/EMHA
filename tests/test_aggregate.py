"""Tests for src.training.aggregate: crop -> participant aggregation."""

import math

import pandas as pd
import pytest

from src.training.aggregate import METHODS, aggregate_crops, fuse_task_families
from src.utils.config import config


def _crops(probs, families=None, pid="001", **extra):
    families = families or ["word"] * len(probs)
    return pd.DataFrame(
        {"participant_id": pid, "task_family": families, "prob_sad": probs, **extra}
    )


def _one(df, method="mean_prob"):
    out = aggregate_crops(df, method)
    assert len(out) == 1
    return out.iloc[0]


# ── aggregate_crops ───────────────────────────────────────────────────────────


def test_brief_example_mean_prob_is_half_and_sad_at_threshold():
    row = _one(_crops([0.2, 0.4, 0.9]))
    assert row["prob_sad"] == pytest.approx(0.5)
    assert row["pred"] == "SAD"  # prob_sad >= threshold -> SAD
    assert row["n_crops"] == 3


def test_default_method_is_primary_mean_prob():
    assert config.aggregate.method == "mean_prob"
    assert config.aggregate.threshold == 0.5
    df = _crops([0.1, 0.3])
    pd.testing.assert_frame_equal(aggregate_crops(df), aggregate_crops(df, "mean_prob"))


def test_mean_logit():
    row = _one(_crops([0.2, 0.4, 0.9]), "mean_logit")
    logits = [math.log(p / (1 - p)) for p in (0.2, 0.4, 0.9)]
    expected = 1 / (1 + math.exp(-sum(logits) / 3))
    assert row["prob_sad"] == pytest.approx(expected)
    assert row["pred"] == ("SAD" if expected >= 0.5 else "HAPPY")


def test_mean_logit_clips_zero_and_one():
    row = _one(_crops([0.0, 1.0, 1.0]), "mean_logit")
    assert 0.5 < row["prob_sad"] < 1.0


def test_majority_vote_share_and_no_tie():
    row = _one(_crops([0.6, 0.7, 0.1]), "majority_vote")
    assert row["prob_sad"] == pytest.approx(2 / 3)
    assert row["pred"] == "SAD"
    assert not row["tie_broken"]


@pytest.mark.parametrize(
    "probs,expected",
    [
        ([0.9, 0.6, 0.1, 0.2], "HAPPY"),  # 2-2 tie, mean 0.45
        ([0.55, 0.55, 0.45, 0.45], "SAD"),  # 2-2 tie, mean 0.50 (>= threshold)
        ([0.51, 0.51, 0.0, 0.0], "HAPPY"),  # 2-2 tie, mean 0.255
        ([0.95, 0.9, 0.3, 0.4], "SAD"),  # 2-2 tie, mean 0.6375
    ],
)
def test_majority_vote_tie_broken_by_mean_prob(probs, expected):
    row = _one(_crops(probs), "majority_vote")
    assert row["prob_sad"] == 0.5  # the tie stays visible
    assert row["tie_broken"]
    assert row["pred"] == expected


def test_crop_exactly_at_threshold_votes_sad():
    row = _one(_crops([0.5, 0.0, 0.5]), "majority_vote")
    assert row["prob_sad"] == pytest.approx(2 / 3)


def test_one_row_per_participant_with_family_means():
    df = pd.concat(
        [
            _crops([0.2, 0.4], ["word", "word"], pid="001"),
            _crops([0.8], ["cursive"], pid="001"),
            _crops([0.9, 0.7], ["drawing", "word"], pid="002"),
        ]
    )
    out = aggregate_crops(df).set_index("participant_id")
    assert list(out.index) == ["001", "002"]
    assert out.loc["001", "prob_sad"] == pytest.approx((0.2 + 0.4 + 0.8) / 3)
    assert out.loc["001", "prob_sad_word"] == pytest.approx(0.3)
    assert out.loc["001", "prob_sad_cursive"] == pytest.approx(0.8)
    assert math.isnan(out.loc["001", "prob_sad_drawing"])
    assert [c for c in out.columns if c.startswith("prob_sad_")] == [
        "prob_sad_drawing",
        "prob_sad_word",
        "prob_sad_cursive",
    ]


def test_pass_through_columns_kept_and_must_be_constant():
    df = _crops([0.2, 0.8], label="SAD", fold=3, model="lr", in_middle_band=False)
    row = _one(df)
    assert (row["label"], row["fold"], row["model"]) == ("SAD", 3, "lr")
    with pytest.raises(ValueError, match="vary within a participant"):
        aggregate_crops(_crops([0.2, 0.8], fold=[1, 2]))


@pytest.mark.parametrize("bad", [[0.2, float("nan")], [0.2, 1.2], [-0.1]])
def test_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        aggregate_crops(_crops(bad))


def test_rejects_unknown_method_and_missing_columns():
    with pytest.raises(ValueError, match="method must be"):
        aggregate_crops(_crops([0.5]), "median")
    with pytest.raises(ValueError, match="lacks columns"):
        aggregate_crops(pd.DataFrame({"participant_id": ["001"], "prob_sad": [0.5]}))
    assert METHODS[0] == "mean_prob"


def test_participant_ids_kept_as_strings():
    out = aggregate_crops(_crops([0.4], pid=7))
    assert out["participant_id"].tolist() == ["7"]


# ── fuse_task_families ────────────────────────────────────────────────────────


def _family_df():
    return pd.concat(
        [
            _crops([0.2, 0.4], ["word", "word"], pid="001"),  # word 0.3
            _crops([0.9], ["cursive"], pid="001"),  # cursive 0.9
            _crops([0.6], ["drawing"], pid="001"),  # drawing 0.6
        ]
    )


def test_fuse_equal_weights_by_default():
    row = fuse_task_families(_family_df()).iloc[0]
    assert row["prob_sad"] == pytest.approx((0.3 + 0.9 + 0.6) / 3)
    assert row["pred"] == "SAD"
    assert row["n_families"] == 3


def test_fuse_differs_from_pooled_crop_mean():
    pooled = aggregate_crops(_family_df()).iloc[0]["prob_sad"]
    fused = fuse_task_families(_family_df()).iloc[0]["prob_sad"]
    assert pooled == pytest.approx((0.2 + 0.4 + 0.9 + 0.6) / 4)
    assert fused != pytest.approx(pooled)


def test_fuse_explicit_weights_and_subset():
    row = fuse_task_families(_family_df(), {"word": 3.0, "cursive": 1.0}).iloc[0]
    assert row["prob_sad"] == pytest.approx((3 * 0.3 + 0.9) / 4)
    assert row["n_families"] == 2
    assert "prob_sad_drawing" not in row.index


def test_fuse_rescales_weights_for_missing_family():
    df = pd.concat([_family_df(), _crops([0.1], ["word"], pid="002")])
    out = fuse_task_families(df).set_index("participant_id")
    assert out.loc["002", "prob_sad"] == pytest.approx(0.1)
    assert out.loc["002", "n_families"] == 1


def test_fuse_rejects_bad_weights():
    with pytest.raises(ValueError, match="not in input"):
        fuse_task_families(_family_df(), {"word": 1.0, "speech": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        fuse_task_families(_family_df(), {"word": -1.0, "cursive": 2.0})
    with pytest.raises(ValueError, match="no weighted family"):
        df = pd.concat([_family_df(), _crops([0.1], ["drawing"], pid="002")])
        fuse_task_families(df, {"word": 1.0, "drawing": 0.0})


def test_fuse_accepts_participant_level_family_rows():
    df = pd.DataFrame(
        {
            "participant_id": ["001", "001"],
            "task_family": ["word", "cursive"],
            "prob_sad": [0.2, 0.6],
            "fold": [0, 0],
        }
    )
    row = fuse_task_families(df).iloc[0]
    assert row["prob_sad"] == pytest.approx(0.4)
    assert row["fold"] == 0
