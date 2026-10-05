# RECOVERY_PLAN.md

Written 2026-10-06 (maintenance; no features). Nothing has been merged,
rebased, pruned, or deleted. Every action below waits for your confirmation.

## 0. Update 2026-10-06: integration (G4 + F7 into main)

**Build-order change (logged):** G4 (`run_log.py`) was built and merged
**before** H1–H5, at your request, instead of at step 6 of the original
order in section 3. No H module was stubbed to make that possible: runners
log point metrics with the CI columns empty, and nothing fills them until H1.

| Item | Where | State |
|---|---|---|
| G4 `src/utils/run_log.py` | `worktree-run-log` dcdf6a3, merged into local `main` | Done. F1 `run_baselines`, F4 `run_cnn`, F6 `run_hybrid` call `log_predictions` and take `stage=` / `--stage` |
| F7 `src/training/run_ensemble.py` | `f7-ensemble` 6c3218f, merged into local `main` | **Single committed home.** Calls `log_predictions` and takes `stage=` / `--stage` |
| F7 duplicates | uncommitted copies in `main` and `pipeline-runner` working trees | Stashed (`git stash push -u -m f7-duplicates-2026-10-06 -- <2 paths>`), not deleted |
| G5 `src/pipeline_runner.py` | `worktree-pipeline-runner` c6428c4 (now committed, not merged) | Still paused |

Open items created by this change:

- **G5 `stage=`:** when G5 is merged, `pipeline_runner` must pass
  `stage=` (`pilot` / `full`; `smoke` comes from a synthetic root) to
  `run_baselines`, `run_cnn`, `run_hybrid` and `run_ensemble`. Without it the
  runners fall back to `run_log.default_stage()`.
- **H1 `log_run`:** the H1 evaluator calls `src.utils.run_log.log_run(...)`
  once per (model, feature_set) with `macro_f1_ci_low` / `macro_f1_ci_high`
  filled from its bootstrap. It is the only caller allowed to fill them
  (`tests/test_run_log.py::test_runners_never_compute_or_pass_a_ci`).
- **smoke_test.py `PENDING["G"]`** (`src.utils.run_log`) can be removed in
  the uncommitted smoke edits, now that G4 exists.

**Next task: H1** (section 4). Its precondition is met: `main` now contains
`run_ensemble` and `run_log`.

## 1. Where things stand

### Worktrees

| Worktree | Branch | Last commit | Uncommitted |
|---|---|---|---|
| `D:\EMHA_Thesis\EMHA-1` | `main` | 6f7b716, 2026-10-05 18:05, CNN-HMM on layer3 sequences | 17 entries |
| `.claude\worktrees\pipeline-runner` (locked) | `worktree-pipeline-runner` | 6f7b716 (same as main) | 21 entries |
| `.claude\worktrees\archive-run` (locked) | `worktree-archive-run` (also on origin) | 4d65559, 2026-10-06 01:29, add archive_run | none |
| `C:\...\Temp\claude\...\scratchpad\wt_hy` | detached HEAD | 6f7b716 | none (stale scratch worktree) |
| `.claude\worktrees\recovery-plan` | `worktree-recovery-plan` | this file only | n/a |

- `origin/main` is at 757b1bc (2026-09-20). Local `main` is several commits ahead and not pushed.
- The uncommitted files on `main` are byte-identical to the same files in
  `pipeline-runner` (F8 `src/app/`, G3 notebooks, `run_ensemble.py`,
  `package_dataset.py`, `EVALUATION_PROTOCOL.md`, and edits to `splits.py`,
  `synthetic.py`, `smoke_test.py`, `test_splits.py`).
- `pipeline-runner` adds only three things on top of that:
  `src/pipeline_runner.py`, `tests/test_pipeline_runner.py`, and a
  `RunnerConfig` block in `config.py`.
- This work exists in no commit. Losing either working copy loses F8 and G3.

### Build-order audit

| ID | Module | Status | Tests | Lives on |
|---|---|---|---|---|
| H1 | `src/training/evaluator.py` | **Partial (legacy).** Phase-10 file; `run_evaluation` and `_run_cross_validation` raise `LegacyPipelineError`. It has no `run(analysis, subset)`, no bootstrap CIs, and no run_log write. | **none** (`test_evaluator.py` missing) | main (committed) |
| H2 | `src/analysis/permutation.py` | **Missing** | none | nowhere |
| H3 | `src/analysis/secondary.py` | **Missing** | none | nowhere |
| H4 | `src/analysis/gradcam.py` | **Missing.** Grad-CAM helpers still sit inside legacy `evaluator.py` (lines 207–352). | none | nowhere |
| H5 | `src/analysis/report.py` | **Missing** | none | nowhere |
| G4 | `src/utils/run_log.py` | **Done** (update 2026-10-06; was missing) | `test_run_log.py` 17 cases | main (merged) |
| G5 | `src/pipeline_runner.py` | Exists (380 lines), uncommitted | `test_pipeline_runner.py`: 13 passed, 1 failed in a file run; the failing `test_end_to_end_on_synthetic_fixture` **passes alone**, so it is order-dependent (state leaks between tests) | pipeline-runner (untracked) |
| G7 | `scripts/archive_run.py` | Exists (131 lines), committed | `test_archive_run.py`: 4 passed | worktree-archive-run |
| F8 | `src/app/predict.py` | Exists (374 lines), uncommitted | `test_predict.py`: 12 passed (main), 12 passed (pipeline-runner) | main + pipeline-runner (untracked) |
| G3 | `notebooks/colab_bootstrap.ipynb`, `notebooks/kaggle_bootstrap.ipynb` | Exist, uncommitted | `test_notebooks.py`: 7 passed (main), 7 passed (pipeline-runner) | main + pipeline-runner (untracked) |

Each test file ran on its own with `pytest -q`, one at a time, with TEMP on D:.

### Stand-ins for H1–H5 created by Stage G / F8 work

No module fakes an H result: there are no mocks returning metrics and no
placeholder `run()` functions. The stand-ins are guards and "not built"
fallbacks, listed below:

| File:line | What it does | Replace when |
|---|---|---|
| `src/pipeline_runner.py:15-19, 130-145` (pipeline-runner) | Step table calls `evaluator.run`, `permutation.run`, `secondary.run`, `gradcam.run`, `report.run` | H1–H5 land (each must expose `run(analysis, subset[, device])`) |
| `src/pipeline_runner.py:233-241` (pipeline-runner) | `resolve()` raises `StepNotBuiltError` for a missing module **or a module with no `run()`**. The legacy `evaluator.py` therefore stops the run as "has no run() function yet" and does not crash. | H1 lands |
| `src/utils/config.py:537-541` (pipeline-runner) | `RunnerConfig.default_steps` lists the five H steps | Keep; verify names after H1–H5 |
| `src/app/predict.py:26, 76, 242-247` (main + pipeline-runner) | `_gradcam_module()` returns None when `src.analysis.gradcam` is absent, so the app shows no Grad-CAM | H4 lands; then drop the fallback message at `:355` |
| `src/app/predict.py:355` | Prints "not available (H4 Grad-CAM not built)" | H4 lands |
| `smoke_test.py:15, 60` (main + pipeline-runner) | `PENDING = {"G": ("src.utils.run_log",), "H": ("src.analysis.report",)}`; stages G and H SKIP | G4 and H5 land; remove both entries |
| `notebooks/kaggle_bootstrap.ipynb:131-135` | Stops with "src.pipeline_runner (G5) is not built yet" | G5 is merged (the guard can stay as a safety check) |
| `src/training/evaluator.py:1-9, 207-352, 405-416` (main) | Legacy file; the H1 work must replace it, not wrap it. Move the Grad-CAM helpers to H4. | H1/H4 |

### G4 rule check (only H1 fills `macro_f1_ci_low` / `macro_f1_ci_high`)

- `run_log.py` does not exist, so the rule cannot be checked against it yet.
- No runner computes a CI. A search for `bootstrap|ci_low|ci_high|percentile`
  across `run_*.py`, `aggregate.py`, `trainer.py`, `pipeline_runner.py`,
  `archive_run.py` and `predict.py` matched only `ReportConfig.n_bootstrap = 1000`
  in `config.py:323`. That is config only and is not used by any runner.
- `run_baselines.py:208-215` computes **point** metrics (accuracy, macro-F1,
  balanced accuracy, ROC-AUC) for its console summary. That is allowed, but
  when G4 lands, `run_baselines` must write these with the CI columns empty.

## 2. Branches to keep paused

- `worktree-pipeline-runner` (G5): paused until H1–H5 exist. Do not add more steps.
- `worktree-archive-run` (G7): paused; complete and green, merge after G5.
- F8 / G3 work (uncommitted on main and pipeline-runner): no more edits until H4 exists.
- `wt_hy` detached scratch worktree: leave alone; prune only on your say-so.
- No Stage G run (smoke on Colab, pilot, full) until H1–H5 are merged.

## 3. Merge order

0. Secure the uncommitted shared work (F-stage `run_ensemble`,
   `package_dataset`, splits, synthetic, smoke edits, F8, G3) in a commit on
   a branch. `EVALUATION_PROTOCOL.md` stays uncommitted per the
   protocol-freeze plan unless you decide otherwise.
   **Done for F7 only** (2026-10-06): `run_ensemble` is committed and merged;
   the rest of step 0 is still open.
1. H1 `src/training/evaluator.py`: replaces the legacy file; `run(analysis, subset)`;
   appends its rows with `run_log.log_run(...)`, filling `macro_f1_ci_low/high`.
2. H2 `src/analysis/permutation.py`
3. H3 `src/analysis/secondary.py`
4. H4 `src/analysis/gradcam.py` (absorbs the helpers from legacy evaluator)
5. H5 `src/analysis/report.py`
6. ~~G4 `src/utils/run_log.py`~~: **done early** (2026-10-06, see section 0)
7. G5 `src/pipeline_runner.py` (+ `RunnerConfig`); fix the order-dependent e2e test
   first; pass `stage=` to every runner
8. G7 `scripts/archive_run.py`
9. F8 `src/app/predict.py`: remove the H4 fallback
10. G3 bootstrap notebooks

After each merge, run `pytest -q`, `black --check .` and `flake8`, and stop for your confirmation.

## 4. Single next task

**Build H1, the participant-level evaluator, in `src/training/evaluator.py`.**
Replace the quarantined legacy file. Expose `run(analysis, subset)`, which reads every
model's `predictions.csv` (the standard prediction schema) and reports, per
model and analysis set, accuracy, macro-F1, balanced accuracy and ROC-AUC with
bootstrap CIs (`ReportConfig.n_bootstrap`), next to the majority baseline.
Also add `tests/test_evaluator.py` on the synthetic fixture. It appends one
`RUN_LOG.csv` row per (model, feature_set) through `run_log.log_run(...)`
with the macro-F1 CIs filled.

Precondition met (2026-10-06): branch H1 from `main`, which now contains
`run_ensemble` and `run_log`.
