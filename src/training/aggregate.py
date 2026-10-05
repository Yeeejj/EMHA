"""
Crop -> participant aggregation, identical for every model — Stage E.

aggregate_crops turns crop-level probabilities into one row per participant:

    mean_prob      prob_sad = mean of crop prob_sad            (PRIMARY rule)
    mean_logit     prob_sad = sigmoid(mean of logit(prob_sad)) (secondary)
    majority_vote  prob_sad = share of crops voting SAD        (secondary)

All crops of a participant are pooled, whatever their task family (CLAUDE.md:
"mean of crop probabilities is the primary rule"), so families with more
crops weigh more (word 15, cursive 5, drawing 4 of 24). The output also
carries each family's plain mean probability (prob_sad_<family>) for
inspection, whatever the method.

pred is "SAD" when prob_sad >= AggregateConfig.threshold, else "HAPPY"
(a probability exactly at the threshold is SAD).

majority_vote ties: a crop votes SAD when its prob_sad >= threshold. If
exactly half the crops vote SAD (vote share == 0.5, only possible with an
even crop count), the tie is broken by mean_prob over the same crops:
pred = SAD iff mean prob_sad >= threshold. prob_sad stays the vote share
(0.5), so the tie is visible in the output (tie_broken=True).

fuse_task_families combines families with explicit weights: each family is
first averaged per participant, then families are combined by weighted mean.
Default weights are EQUAL. Any other weights must be chosen on inner
validation participants only -- never on outer test folds (CLAUDE.md
Non-Negotiable 8); this module cannot check where weights came from, so
callers pass them explicitly and must log their source. A participant with
no crops in a weighted family is fused over the families it has, with the
weights rescaled to sum to 1 (n_families records how many were used).

Pass-through columns: any of analysis, model, feature_set, fold, label,
in_middle_band present in the input is kept, and must be constant within a
participant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import config

METHODS = ("mean_prob", "mean_logit", "majority_vote")
PRIMARY_METHOD = "mean_prob"
FAMILIES = ("drawing", "word", "cursive")
PASS_THROUGH = ("analysis", "model", "feature_set", "fold", "label", "in_middle_band")
SAD, HAPPY = "SAD", "HAPPY"


def _check_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"participant_id", "task_family", "prob_sad"} - set(df.columns)
    if missing:
        raise ValueError(f"input lacks columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("input has no rows")
    probs = df["prob_sad"]
    if probs.isna().any() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("prob_sad must be in [0, 1] with no missing values")
    out = df.copy()
    out["participant_id"] = out["participant_id"].astype(str)
    return out


def _pass_through(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PASS_THROUGH if c in df.columns]
    if not cols:
        return pd.DataFrame(index=pd.Index(sorted(df["participant_id"].unique())))
    grouped = df.groupby("participant_id")[cols]
    counts = grouped.nunique(dropna=False)
    varying = [c for c in cols if (counts[c] > 1).any()]
    if varying:
        raise ValueError(f"columns vary within a participant: {varying}")
    return grouped.first()


def _pred(prob: pd.Series, threshold: float) -> pd.Series:
    return np.where(prob >= threshold, SAD, HAPPY)


def aggregate_crops(df: pd.DataFrame, method: str = "mean_prob") -> pd.DataFrame:
    """One row per participant: prob_sad, pred, n_crops, prob_sad_<family>.

    df: crop rows with participant_id, task_family, prob_sad (plus optional
    pass-through columns). method: one of METHODS; mean_prob is primary.
    Sorted by participant_id.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    acfg = config.aggregate
    crops = _check_input(df)
    by_pid = crops.groupby("participant_id")["prob_sad"]
    mean_prob = by_pid.mean()

    out = pd.DataFrame(index=mean_prob.index)
    out["n_crops"] = by_pid.size()
    tie = pd.Series(False, index=out.index)

    if method == "mean_prob":
        out["prob_sad"] = mean_prob
        pred = _pred(out["prob_sad"], acfg.threshold)
    elif method == "mean_logit":
        p = crops["prob_sad"].clip(acfg.logit_eps, 1 - acfg.logit_eps)
        logit = np.log(p / (1 - p))
        mean_logit = logit.groupby(crops["participant_id"]).mean()
        out["prob_sad"] = 1 / (1 + np.exp(-mean_logit))
        pred = _pred(out["prob_sad"], acfg.threshold)
    else:  # majority_vote
        votes = (crops["prob_sad"] >= acfg.threshold).astype(float)
        share = votes.groupby(crops["participant_id"]).mean()
        out["prob_sad"] = share
        tie = share == 0.5
        pred = np.where(
            tie, _pred(mean_prob, acfg.threshold), _pred(share, acfg.threshold)
        )

    out["pred"] = pred
    out["method"] = method
    if method == "majority_vote":
        out["tie_broken"] = tie

    family_means = crops.pivot_table(
        index="participant_id", columns="task_family", values="prob_sad", aggfunc="mean"
    )
    for family in [f for f in FAMILIES if f in family_means.columns] + sorted(
        set(family_means.columns) - set(FAMILIES)
    ):
        out[f"prob_sad_{family}"] = family_means[family]

    out = _pass_through(crops).join(out)
    out.index.name = "participant_id"
    return out.reset_index()


def fuse_task_families(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """One row per participant: weighted mean of per-family mean prob_sad.

    df: long rows with participant_id, task_family, prob_sad (crop-level or
    already one row per participant x family). weights: {family: weight};
    None = equal weights over the families present in df. Non-default
    weights must come from inner validation only (see module docstring).
    """
    acfg = config.aggregate
    rows = _check_input(df)
    present = sorted(rows["task_family"].unique())
    if weights is None:
        weights = {family: 1.0 for family in present}
    unknown = set(weights) - set(present)
    if unknown:
        raise ValueError(f"weights for families not in input: {sorted(unknown)}")
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    rows = rows[rows["task_family"].isin(list(weights))]

    family_mean = rows.pivot_table(
        index="participant_id", columns="task_family", values="prob_sad", aggfunc="mean"
    )
    w = pd.Series(weights, dtype=float).reindex(family_mean.columns)
    available = family_mean.notna()
    weight_sum = available.mul(w, axis=1).sum(axis=1)
    if (weight_sum <= 0).any():
        bad = sorted(weight_sum.index[weight_sum <= 0])
        raise ValueError(f"participants with no weighted family: {bad[:10]}")
    fused = family_mean.fillna(0).mul(w, axis=1).sum(axis=1) / weight_sum

    out = pd.DataFrame(index=family_mean.index)
    out["prob_sad"] = fused
    out["pred"] = _pred(out["prob_sad"], acfg.threshold)
    out["n_families"] = available.sum(axis=1)
    for family in family_mean.columns:
        out[f"prob_sad_{family}"] = family_mean[family]
    out = _pass_through(rows).join(out, how="right")
    out.index.name = "participant_id"
    return out.reset_index()
