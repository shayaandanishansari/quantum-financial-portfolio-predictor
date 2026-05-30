# Quantum Reinforcement Learning for Portfolio Optimization

This is a fork of [Vincent Gurgul's](https://github.com/VincentGurgul) original codebase: [VincentGurgul/qrl-dpo-public](https://github.com/VincentGurgul/qrl-dpo-public).

---

## Overview

This project implements **Quantum Reinforcement Learning (QRL)** for Dynamic Portfolio Optimization (DPO) using **Variational Quantum Circuits (VQCs)** as Quantum Neural Networks (QNNs) in Python 3.12.6.

The core problem: given the last 30 days of log-returns across 15 assets (a 450-value window), output a portfolio weight vector that maximizes risk-adjusted return. Several approaches are compared:

| Model | Type | Description |
|---|---|---|
| Equal Weights | Baseline | Fixed 1/N allocation |
| Mean-Variance Optimization | Classical | Closed-form quadratic solver (Markowitz) |
| DDPG | Classical RL | MLP actor–critic |
| QDPG | Quantum RL | VQC actor with **amplitude encoding** |
| QDPG (Stacked Angle Encoding) | Quantum RL | VQC actor with **stacked angle (data re-uploading) encoding** *(this fork)* |
| Deep Q-Learning | Classical RL | MLP Q-network |
| Quantum Q-Learning | Quantum RL | VQC Q-network |

Performance is evaluated on a 15-asset portfolio (US equities, ETFs, bonds, commodities) over 2011–2025 using **3-fold expanding-window time-series cross-validation** (no shuffling; ~707-day out-of-sample test block per fold), reporting annualized profit and Sharpe ratio under both static (SPO) and dynamic (DPO, rebalancing every 30 days) strategies.


## Contribution: A Fair Stacked-Angle vs Amplitude Encoding Comparison

The original QDPG uses **amplitude encoding** to load the 450-feature input (15 assets × 30 days) into the quantum circuit. Amplitude encoding L2-normalizes the input into quantum probability amplitudes via a single state-preparation step — a process that **discards the overall magnitude (volatility scale) of the window** and the temporal ordering of the days.

This fork adds **`QDPG (Stacked Angle Encoding)`** and turns the two quantum models into a **controlled comparison where the encoding strategy is the only variable**:

- **Stacked angle encoding** loads the 450 values as **30 sequential passes** of 15 `RY` rotations, with CNOT entanglement after each pass (a *data re-uploading* circuit). Temporal order and per-value magnitude are preserved.
- Both models are held identical otherwise: **15 qubits**, **`NUM_WEIGHTS = 30`** (a single shared constant), an **identical classical MLP critic** (`hidden_sizes=(30,)`), **`batch_size = 32`**, **`forecast_window = 0`** (so both see exactly 450 inputs), and identical training hyperparameters and seed.

### Key finding: angle encoding needs *bounded* inputs

A first attempt fed **both** models raw z-scores (`standardize`). Amplitude trained fine, but stacked angle behaved like an untrained network (Sharpe ≈ 0, wildly variable). The cause: unbounded z-scores (~±4) used directly as rotation angles, accumulated over 30 stacked passes, **over-rotate** the circuit into a regime where it can't learn. The fix is to give the angle circuit **bounded** inputs via `normalized_arcsin` (min-max → arcsin → [−π/2, π/2]); amplitude keeps `standardize` since it normalizes into a state and doesn't care about angle bounds.

The takeaway: **identical preprocessing is not the same as a fair comparison** — each encoding must receive the preprocessing it was designed for (bounded angles for angle encoding, a unit-norm state for amplitude).

### Changes from the original codebase

- `MAIN.py` — two controlled QDPG blocks (stacked-angle + amplitude); shared `NUM_WEIGHTS` constant; `CV_SPLITS` set to 3; bounded-angle (`normalized_arcsin`) transform for the angle model.
- `ddpg/ddpg_functions.py` — **batched / vectorized** training loop (was one window at a time) with gradient clipping; `critic_predictor` / `critic_predictor_kwargs` to allow an independent critic network type.
- `predictors/input_transformations.py` — `normalized_arcsin` bounding (+ `+1e-8` and arcsin-clamp numerical guards); PCA transform classes from an earlier angle-encoding exploration (retained, no longer used).
- `predictors/quantum_neural_network.py` — `transformed_input_size` parameter for dimension-changing transforms.


## Results

3-fold expanding-window time-series CV, `SEED=68`. **DPO** (dynamic rebalancing) is the realistic, more stable metric and is reported as the headline; **SPO** is the static one-shot allocation. Spread shown is the **interquartile range (IQR, P75−P25)**, not standard deviation.

| Model | DPO profit p.a. % (min→max) | DPO Sharpe | SPO profit p.a. % | SPO Sharpe |
|---|---|---|---|---|
| **QDPG (Stacked Angle)** | **29.57 (21.21 → 34.19)** | **0.769** | 22.72 | 0.848 |
| QDPG (Amplitude) | 11.09 (0.44 → 22.54) | 0.130 | 8.72 | 0.241 |
| Mean-Variance Optimization | 17.06 (10.37 → 25.36) | 0.593 | 15.22 | 0.459 |
| Deep Q-Learning | 15.17 (4.07 → 21.16) | 0.381 | 13.50 | 0.528 |
| Equal Weights | — | — | 11.52 | 0.584 |
| Classical DDPG | 10.57 (2.85 → 18.31) | 0.437 | 10.57 | 0.438 |

**Stacked-angle QDPG is the strongest model on DPO** (≈29.6% p.a., Sharpe 0.77, all three folds in 21–34%), beating the amplitude QDPG and every classical baseline. The result is consistent with the encoding mechanism: data re-uploading is more expressive and preserves magnitude + temporal order, whereas amplitude encoding discards both.

> **Fair comparison note.** Classical DDPG is now trained on the *same* regime as QDPG (batch_size=32, forecast_window=0, early_stopping=True, identical LRs and hyperparameters) — the only difference is `NeuralNetwork` vs `QuantumNeuralNetwork`. At ≈10.6% DPO profit / Sharpe 0.44, classical DDPG is competitive with amplitude QDPG (≈11.1%) and only the stacked-angle encoding clearly leads, suggesting that advantage is genuine rather than a configuration artifact.

> **Note on variance.** At 3 folds the *SPO* figure is noisy (folds spanned 5–52%); the *DPO* figure is tight and is the one to trust. The win reflects the full stacked-angle *strategy* (a deeper re-uploading circuit), and amplitude's cheap state-prep is a simulation convenience — on real hardware amplitude encoding is the expensive one.


## Requirements

- Python 3.12.6
- Core libraries: `pennylane`, `torch`, `scikit-learn`
- Full dependency list: `requirements.txt`


## Installation

```bash
git clone <your-fork-url>
cd qrl-dpo-public
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## Quick Start

```bash
python MAIN.py
```

To run only specific models, edit `RUN_MODELS` in `MAIN.py`. With `CV_SPLITS = 3`, a full run takes roughly **19 min for stacked angle** and **6 min for amplitude** on a CPU state-vector simulator (`default.qubit`). For faster iteration, lower `CV_SPLITS`; for a speedup with identical results, switch the device in `predictors/quantum_neural_network.py` from `default.qubit` to `lightning.qubit`.

Design notes and a full change report are in `encoding_comparison_v2.html`, `qdpg_pipeline.html`, and `changes_report.html`. A detailed project handoff is in `context.md`.


## Repo Layout

- `MAIN.py` — entrypoint; configure `RUN_MODELS` and `GLOBAL_CONFIG` here
- `config.py` — asset tickers and date range
- `models.py` — high-level model/pipeline wrappers
- `ddpg/` — DDPG and QDPG implementation
- `q_learning/` — Q-Learning / DQN implementation
- `mvo/` — Mean-Variance Optimization
- `predictors/` — classical (MLP) and quantum (VQC) network implementations, encoding strategies, and input transformations
- `utilities/` — data processing, training loop utilities, metrics, and logging
- `data/` — dataset files and download script
- `visualizations/` — result plotting notebooks


## License

MIT License. See `LICENSE` for details.

Original work by [Vincent Gurgul](https://www.linkedin.com/in/vincent-gurgul/) — [github.com/VincentGurgul/qrl-dpo-public](https://github.com/VincentGurgul/qrl-dpo-public).
