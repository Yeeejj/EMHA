"""
Per-class GaussianHMM classifier with Platt calibration — Stage F.

    clf = HMMClassifier(cfg.hmm, seed)
    clf.fit(train_sequences, train_labels)          # list of (T_i, F) arrays
    clf.fit_calibrator(clf.decision_scores(val_seqs), val_labels)
    proba = clf.predict_proba(test_sequences)       # (n, 2): [HAPPY, SAD]

fit (training sequences only):
  1. StandardScaler (if standardize) then PCA(min(pca_components, F, frames))
     fit on the pooled training frames; both are re-applied at predict time.
  2. For each class (HAPPY = 0, SAD = 1; config.data.label_to_index), an
     hmmlearn GaussianHMM is fit n_restarts times with random_state =
     seed + restart; the restart with the best training log-likelihood is
     kept. Every restart's convergence (monitor_.converged) is logged; a
     winning restart that did not converge is logged as a WARNING.
     EM stops when the total log-likelihood gains less than
     HMMConfig.tol x (number of training frames of that class): tol is per
     frame, because hmmlearn's tol is absolute on the summed log-likelihood
     and 1e-3 would never trigger with thousands of frames. A restart counts
     as converged only if it stopped on that tolerance, not on n_iter.
  3. Topology "ergodic": hmmlearn's default initialisation. "left_right":
     startprob_ = [1, 0, ...] and a banded transmat_ (stay or move one state
     right; last state absorbing) set by hand with init_params="cm" and
     params="cmt" -- EM re-estimates the transitions but entries that start
     at zero stay zero (hmmlearn tutorial, "left-right HMM").

decision_scores: length-normalised log-likelihood difference plus the
log prior ratio of the training labels,

    score = LL_sad(x) / T - LL_happy(x) / T + log(n_sad / n_happy).

score > 0 favours SAD. fit_calibrator fits a 1-D logistic regression
(Platt scaling, no regularisation) of the labels on held-out scores --
inner-validation participants only, never test participants. predict_proba
before fit_calibrator warns and falls back to sigmoid(score), which is not
calibrated.

save / load persist the scaler, PCA, both HMMs, priors, and the calibrator
(joblib).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TOPOLOGIES = ("ergodic", "left_right")


class UncalibratedWarning(UserWarning):
    """predict_proba used before fit_calibrator."""


def left_right_params(n_states: int) -> tuple:
    """(startprob, transmat) of a banded left-right HMM."""
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        transmat[i, i] = transmat[i, i + 1] = 0.5
    transmat[-1, -1] = 1.0
    return startprob, transmat


class LeftRightGaussianHMM(hmm.GaussianHMM):
    """GaussianHMM whose banded transmat survives states with no exits.

    With short sequences the last states can be entered only on a final
    frame, so EM sees no transition out of them; hmmlearn then leaves an
    all-zero row and rejects the model. After every M-step such rows are
    restored to their previous (banded) values, keeping the structural
    zeros of the left-right topology.
    """

    def _do_mstep(self, stats):
        previous = self.transmat_.copy()
        super()._do_mstep(stats)
        dead = ~np.isclose(self.transmat_.sum(axis=1), 1.0)
        if dead.any():
            self.transmat_[dead] = previous[dead]


class HMMClassifier:
    """Two GaussianHMMs (HAPPY, SAD) on scaled + PCA-reduced frame sequences."""

    def __init__(self, cfg=None, seed: int | None = None):
        from src.utils.config import config

        self.cfg = cfg or config.hmm
        if self.cfg.topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}")
        self.seed = config.training.seed if seed is None else seed
        label_to_index = config.data.label_to_index
        self.happy, self.sad = label_to_index["HAPPY"], label_to_index["SAD"]
        self.classes_ = np.array(sorted((self.happy, self.sad)))
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.models: dict = {}
        self.log_prior_ratio = 0.0
        self.calibrator: LogisticRegression | None = None
        self.restart_log: dict = {}

    # ── frames ────────────────────────────────────────────────────────────

    def _transform(self, sequences) -> list:
        out = []
        for seq in sequences:
            x = np.asarray(seq, dtype=float)
            if x.ndim != 2 or len(x) == 0:
                raise ValueError("each sequence must be a non-empty (T, F) array")
            if self.scaler is not None:
                x = self.scaler.transform(x)
            out.append(self.pca.transform(x))
        return out

    def _fit_frames(self, sequences) -> None:
        frames = np.concatenate([np.asarray(s, dtype=float) for s in sequences])
        if self.cfg.standardize:
            self.scaler = StandardScaler().fit(frames)
            frames = self.scaler.transform(frames)
        n_comp = min(self.cfg.pca_components, frames.shape[1], frames.shape[0])
        self.pca = PCA(n_components=n_comp, svd_solver="full").fit(frames)

    # ── HMMs ──────────────────────────────────────────────────────────────

    def _new_hmm(
        self, n_states: int, random_state: int, n_frames: int
    ) -> hmm.GaussianHMM:
        kwargs = dict(
            n_components=n_states,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            tol=self.cfg.tol * n_frames,
            min_covar=self.cfg.min_covar,
            random_state=random_state,
        )
        if self.cfg.topology == "left_right":
            model = LeftRightGaussianHMM(init_params="cm", params="cmt", **kwargs)
            model.startprob_, model.transmat_ = left_right_params(n_states)
            return model
        return hmm.GaussianHMM(**kwargs)

    def _fit_class(self, name: str, sequences: list):
        X = np.concatenate(sequences)
        lengths = [len(s) for s in sequences]
        n_states = max(1, min(self.cfg.n_states, min(lengths)))
        best, best_ll, runs = None, -np.inf, []
        for restart in range(self.cfg.n_restarts):
            seed = self.seed + restart
            model = self._new_hmm(n_states, seed, len(X))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, lengths)
            ll = float(model.score(X, lengths))
            history = list(model.monitor_.history)
            # stopped on the tolerance (last gain < tol), not on the n_iter cap;
            # hmmlearn 0.3 keeps the whole log-likelihood history
            converged = len(history) >= 2 and history[-1] - history[-2] < model.tol
            runs.append(
                {
                    "restart": restart,
                    "seed": seed,
                    "log_likelihood": ll,
                    "converged": converged,
                    "n_iter": model.monitor_.iter,
                }
            )
            logger.info(
                "HMM %s restart %d (seed %d): LL %.3f, converged=%s",
                name,
                restart,
                seed,
                ll,
                converged,
            )
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll, winner = model, ll, runs[-1]
        if best is None:
            raise RuntimeError(f"no finite HMM fit for class {name}")
        level = logging.INFO if winner["converged"] else logging.WARNING
        logger.log(
            level,
            "HMM %s winning restart %d (seed %d): LL %.3f, converged=%s after %d "
            "iterations (%d/%d restarts converged)",
            name,
            winner["restart"],
            winner["seed"],
            best_ll,
            winner["converged"],
            winner["n_iter"],
            sum(r["converged"] for r in runs),
            len(runs),
        )
        self.restart_log[name] = {"winner": winner["restart"], "runs": runs}
        return best

    def fit(self, sequences: list, labels) -> "HMMClassifier":
        """Fit scaler, PCA, and one HMM per class on training sequences only."""
        y = np.asarray(labels).astype(int)
        if len(y) != len(sequences):
            raise ValueError("one label per sequence")
        if set(y) != {self.happy, self.sad}:
            raise ValueError("training needs sequences of both classes")
        self._fit_frames(sequences)
        reduced = self._transform(sequences)
        self.models = {}
        for name, cls in (("HAPPY", self.happy), ("SAD", self.sad)):
            class_seqs = [s for s, lab in zip(reduced, y) if lab == cls]
            self.models[name] = self._fit_class(name, class_seqs)
        n_sad, n_happy = int(np.sum(y == self.sad)), int(np.sum(y == self.happy))
        self.log_prior_ratio = float(np.log(n_sad / n_happy))
        self.calibrator = None
        return self

    # ── scores and probabilities ──────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.models:
            raise RuntimeError("HMMClassifier is not fitted")

    def decision_scores(self, sequences) -> np.ndarray:
        """Length-normalised LL_sad - LL_happy + log prior ratio, per sequence."""
        self._check_fitted()
        out = []
        for x in self._transform(sequences):
            t = len(x)
            ll_sad = self.models["SAD"].score(x) / t
            ll_happy = self.models["HAPPY"].score(x) / t
            out.append(ll_sad - ll_happy + self.log_prior_ratio)
        return np.asarray(out, dtype=float)

    def fit_calibrator(self, scores, labels) -> "HMMClassifier":
        """Platt scaling on held-out (inner-validation) scores and labels."""
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = (np.asarray(labels).astype(int) == self.sad).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration needs held-out scores of both classes")
        self.calibrator = LogisticRegression(penalty=None).fit(s, y)
        return self

    def predict_proba(self, sequences) -> np.ndarray:
        """(n, 2) probabilities, columns ordered as self.classes_ (HAPPY, SAD)."""
        scores = self.decision_scores(sequences)
        if self.calibrator is None:
            warnings.warn(
                "HMMClassifier.predict_proba called before fit_calibrator; "
                "returning uncalibrated sigmoid(score).",
                UncalibratedWarning,
                stacklevel=2,
            )
            p_sad = 1.0 / (1.0 + np.exp(-scores))
        else:
            p_sad = self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        proba = np.empty((len(scores), 2))
        proba[:, list(self.classes_).index(self.sad)] = p_sad
        proba[:, list(self.classes_).index(self.happy)] = 1.0 - p_sad
        return proba

    def predict(self, sequences) -> np.ndarray:
        """Class index per sequence (SAD iff decision score > 0)."""
        scores = self.decision_scores(sequences)
        return np.where(scores > 0, self.sad, self.happy)

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "cfg": asdict(self.cfg),
                "seed": self.seed,
                "scaler": self.scaler,
                "pca": self.pca,
                "models": self.models,
                "log_prior_ratio": self.log_prior_ratio,
                "calibrator": self.calibrator,
                "restart_log": self.restart_log,
            },
            path,
        )

    def load(self, path) -> "HMMClassifier":
        """Load a saved classifier into self (returns self)."""
        from src.utils.config import HMMConfig

        state = joblib.load(Path(path))
        self.cfg = HMMConfig(**state["cfg"])
        self.seed = state["seed"]
        self.scaler, self.pca = state["scaler"], state["pca"]
        self.models = state["models"]
        self.log_prior_ratio = state["log_prior_ratio"]
        self.calibrator = state["calibrator"]
        self.restart_log = state["restart_log"]
        return self
