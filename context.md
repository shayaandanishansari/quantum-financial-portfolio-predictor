# Project Context & Handoff

> Detailed handoff for resuming work in a fresh session. Captures what this project is, every change
> made and why, the exact configuration, and the precise results. Last updated 2026-05-30.

---

## 1. Project & assignment

- **Course:** CS 4084 – Quantum Computing (Spring 2026). Assignment = implement a base paper/codebase
  plus a *proposed methodology*, compare results, and write an IEEE-format paper + GitHub submission.
- **Base repo:** fork of **Vincent Gurgul's** [qrl-dpo-public](https://github.com/VincentGurgul/qrl-dpo-public)
  — Quantum Reinforcement Learning for Dynamic Portfolio Optimization.
- **Our contribution:** add a **stacked-angle (data re-uploading) encoding** path and turn the quantum
  models into a **controlled "encoding is the only variable" comparison** (stacked angle vs amplitude),
  including the engineering needed to make it train and a fair comparison harness.
- **Deliverables:** paper (IEEE) + GitHub repo with professional README. Paper/GitHub due **31 May 2026**.

## 2. The problem & data

- **Task:** given a 30-day × 15-asset window of log-returns (450 values, flattened), output 15 portfolio
  weights maximizing risk-adjusted return (an actor–critic RL setup, DDPG-style).
- **Data:** `data/price_data.parquet.gzip` — **3,536 daily rows × 15 assets**, log-returns,
  **2011-08-09 → 2025-08-29**. Tickers (`config.py`): AAPL, MSFT, JNJ, XOM, JPM, SPY, QQQ, IWM, XLV,
  XLF, TLT, LQD, GLD, USO, EFA.
- **Evaluation:** `TimeSeriesCrossValidation` (`utilities/data_processing.py`) → sklearn
  `TimeSeriesSplit(n_splits = CV_SPLITS + 1)`, **drops fold 0**, expanding window, no shuffle. With
  `CV_SPLITS=3`: 3 folds, ~707-day test block each; per-fold train sets ≈ 1415 / 2122 / 2829 rows,
  each split 80/20 train/val. **Gotcha:** `RLDataset` uses **non-overlapping** windows (stride =
  window_size = 30), so only ~36/55/74 training windows per fold. Metrics: SPO (static one-shot
  allocation) and DPO (dynamic, rebalance every 30 days), each reporting annualized profit and Sharpe.

## 3. Base codebase (commit `94998cb [INIT]`)

- A **single** quantum model `'QDPG'`: `encoding='amplitude'`, `input_transformation=radial_to_linear`
  (4× feature expansion 450→1,800), `batch_size=1`, `num_weights=60`, `num_qubits` auto (~11–12),
  `forecast_window=7`. Critic was the **same class as the actor** (i.e. a quantum critic).
- Baselines: Equal Weights, Mean-Variance Optimization, classical DDPG, Deep Q-Learning, Quantum Q-Learning.
- Training loop was **sequential** — one window at a time (`state.flatten()`), `replay_buffer.sample(1)`
  hardcoded (so `batch_size` was effectively ignored for the gradient step).

## 4. What was changed & why

### (a) Controlled encoding comparison — `MAIN.py`
Replaced the single amplitude `'QDPG'` block with **two** matched blocks:
- `'QDPG (Stacked Angle Encoding)'`: `encoding='stacked_angle'`, `input_transformation=normalized_arcsin`.
- `'QDPG'`: `encoding='amplitude'`, `input_transformation=standardize`.

Both identical otherwise: `num_qubits=len(tickers)=15`, `num_weights=NUM_WEIGHTS` (=30), classical
critic `NeuralNetwork(hidden_sizes=(30,))`, `batch_size=32`, `forecast_window=0` (→ 450 inputs),
`rotation_axes='y'`, same optimizer/LRs/gamma/risk/l2, `early_stopping=True`, `patience=10`, `SEED=68`.
**Only `encoding` (and its appropriate preprocessing) differs.** Also set `CV_SPLITS=3` and added the
`NUM_WEIGHTS = 30` shared constant near `GLOBAL_CONFIG`.

> The `StackedAngleEncoding` / `AmplitudeEncoding` classes already existed in the base
> `predictors/quantum_neural_network.py` — the contribution is wiring them into a controlled
> experiment, not implementing the encoders.

### (b) Batched / vectorized training loop — `ddpg/ddpg_functions.py`
Rewrote the loop to operate on full batches: `state.view(b, -1)`, batched actor/critic forward passes,
push each transition then `sample(min(batch_size, len(buffer)))`, losses reduced with `.mean()`, and
**gradient clipping** (`clip_grad_norm_(max_norm=1.0)`) on both networks. Cuts gradient steps ~32× and
stabilizes training; this is the main reason the 15-qubit sims are tractable.

### (c) Independent critic — `ddpg/ddpg_functions.py`
Added `critic_predictor` / `critic_predictor_kwargs` kwargs (default to the actor class for backward
compat). Both QDPG models now use a **classical** MLP critic, holding the critic constant across both
encodings so it isn't a confound.

### (d) Numerical guards
`standardize` → `/(std+1e-8)`; `min_max_scale_to_range` → `/(... +1e-8)`; `normalized_arcsin` →
`arcsin(x.clamp(-1+1e-6, 1-1e-6))`; DDPG short-sell activation → `/(sum+1e-8)`; `reduce_negatives` →
`/(sum+1e-8)`; `print_results` → `np.nanmean/nanmin/nanmax`; `arima_forecast` → NaN fallback to last value.

### (e) THE KEY FIX — bounded angles for stacked encoding
First run used `standardize` for **both** models. **Amplitude trained; stacked angle did not** (Sharpe
≈ 0, SPO folds swung −28.7% → +27.1%). Cause: raw z-scores (~±4, unbounded) fed directly as `RY` angles
and accumulated over **30 stacked passes per qubit** → over-rotation, untrainable. **Fix:** stacked
angle uses `normalized_arcsin` (min-max → arcsin → bounds each angle to [−π/2, π/2]); amplitude keeps
`standardize` (it L2-normalizes into a state, so angle bounds are irrelevant). Lesson: *identical
preprocessing ≠ fair* — each encoding needs the preprocessing it was designed for.

### (f) PCA-angle path — explored then dropped
An intermediate version used PCA (450→15) + `'angle'` encoding (`PCATransform` /
`PCAWithNormalization` in `input_transformations.py`). Superseded by stacked angle; the PCA classes
remain in the file but are **no longer used** by `MAIN.py`. (We considered re-adding PCA-angle as a
separate benchmark baseline but **intentionally skipped it** for now.)

## 5. Exact current configuration

| Setting | Value |
|---|---|
| `LOOKBACK_WINDOW` | 30 |
| `forecast_window` | **0** (block-level override; `GLOBAL_CONFIG['FORECAST_WINDOW']` is 7 but unused by the QDPG blocks) |
| `CV_SPLITS` | 3 |
| `NUM_WEIGHTS` | 30 (shared by both QDPG models) |
| `num_qubits` | 15 (= `len(tickers)`, both) |
| `batch_size` | 32 (both) |
| `SEED` | 68 |
| `rotation_axes` | `'y'` (RY gates) |
| ansatz | `ParameterizedQuantumCircuit`, reverse_linear CNOT entanglement |
| critic | `NeuralNetwork(hidden_sizes=(30,))` (classical, both) |
| optimizer | SGD; actor_lr ≈ 0.0994, critic_lr ≈ 0.00180, l2_lambda ≈ 3.21e-6, risk_preference ≈ −0.929, gamma ≈ 0.00983 |
| early stopping | `True`, `patience=10`, `num_epochs=50` |
| stacked-angle preprocessing | `normalized_arcsin` (bounded angles) |
| amplitude preprocessing | `standardize` |
| simulator | PennyLane `default.qubit`, `shots=None`, `diff_method='best'` (adjoint) |

## 6. Results — full detail

Source: `results_logs/2026-05-30_18-48-55.log`. CV_SPLITS=3, SEED=68. The "std" printed by
`print_results` is actually the **interquartile range (IQR = P75 − P25)**, not standard deviation —
relabel it in the paper.

### QDPG (Stacked Angle Encoding) — runtime 19m07s
| Metric | Mean | IQR | Min | Max |
|---|---|---|---|---|
| SPO profit p.a. | 22.7157% | 23.3678 | 5.0587% | 51.7943% |
| SPO Sharpe | 0.8481 | 0.8138 | 0.0279 | 1.6555 |
| **DPO profit p.a.** | **29.5691%** | 6.4915 | 21.2068% | 34.1899% |
| **DPO Sharpe** | **0.7688** | 0.1190 | 0.6791 | 0.9170 |

### QDPG (Amplitude) — runtime 6m02s
| Metric | Mean | IQR | Min | Max |
|---|---|---|---|---|
| SPO profit p.a. | 8.7244% | 7.1830 | 2.0379% | 16.4039% |
| SPO Sharpe | 0.2410 | 0.3839 | −0.0904 | 0.6774 |
| DPO profit p.a. | 11.0923% | 11.0512 | 0.4426% | 22.5449% |
| DPO Sharpe | 0.1295 | 0.2977 | −0.2090 | 0.3864 |

### Baselines (same run)
| Model | SPO profit / Sharpe | DPO profit / Sharpe | time |
|---|---|---|---|
| Equal Weights | 11.5186% / 0.5844 | — (SPO only) | 0s |
| Mean-Variance Optimization | 15.2244% / 0.4590 | 17.0625% / 0.5926 | 1.31s |
| Classical DDPG | 0.8186% / −0.0347 | 0.9436% / −0.0334 | 2m14s |
| Deep Q-Learning | 13.4950% / 0.5281 | 15.1660% / 0.3811 | 36.36s |
| Quantum Q-Learning | **did not complete in this log** (header only) | | |

## 7. Interpretation

- **Stacked angle wins decisively on DPO** (29.6%, Sharpe 0.77, all 3 folds 21–34%) — best model in the
  run, beating MVO (17%), DQL (15%), Equal Weights (11.5%), and amplitude QDPG (11%).
- **Why:** (1) stacked angle is a *data re-uploading* circuit (30 re-uploads + entanglement) → far more
  expressive once it trains; (2) amplitude **L2-normalizes**, discarding the window's magnitude
  (volatility scale) and temporal order — information that matters for this task; stacked angle keeps
  both as bounded per-value rotations. Matches "Scenario A" in `encoding_comparison_v2.html`.

## 8. Caveats (state these in the paper)

- **3-fold CV is noisy**, especially SPO (stacked-angle SPO spanned 5%→52%). **DPO is the robust
  metric** — report it as the headline. More folds would tighten SPO (and likely lower its mean toward
  the steadier DPO value); earlier amplitude runs scored ~15% with more folds → results are
  dataset-budget dependent.
- The printed spread is **IQR**, not std — relabel or recompute true std for the paper.
- Amplitude's fast `StatePrep` is a **simulator** convenience; on real quantum hardware amplitude
  encoding is exponentially expensive — the speed ranking would flip.
- The win reflects the whole stacked-angle **strategy** (deeper re-uploading circuit), not narrowly
  "angle beats amplitude."

## 9. Repo state & open items

- **Uncommitted:** the angle-bounding change (`MAIN.py` → `normalized_arcsin`; the guards in
  `predictors/input_transformations.py`) is a **working-tree edit on top of HEAD `46b82bb`** — not yet
  committed. Commit before/after the paper as desired.
- **Skipped:** the PCA-angle baseline model (decided to leave out for now).
- **Incomplete:** Quantum Q-Learning did not finish in the latest results log.
- **Design docs:** `encoding_comparison_v2.html` (the fair-comparison spec — note: still says
  "standardize only", superseded by the bounded-angle fix), `qdpg_pipeline.html` (data→circuit pipeline),
  `changes_report.html` (base→current change report, already updated for the angle bounding).
- **Key source files:** `MAIN.py` (the two QDPG blocks), `ddpg/ddpg_functions.py` (batched DDPG +
  independent critic), `predictors/quantum_neural_network.py` (encoders + QNN), `predictors/input_transformations.py`
  (`standardize`, `normalized_arcsin`, PCA classes), `utilities/data_processing.py` (CV + RLDataset),
  `utilities/metrics.py` (RLEvaluator + SPO/DPO).
