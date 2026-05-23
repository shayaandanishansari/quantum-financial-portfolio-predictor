# Quantum Reinforcement Learning for Portfolio Optimization

This is a fork of [Vincent Gurgul's](https://github.com/VincentGurgul) original codebase: [VincentGurgul/qrl-dpo-public](https://github.com/VincentGurgul/qrl-dpo-public).

---

## Overview

This project implements **Quantum Reinforcement Learning (QRL)** for Dynamic Portfolio Optimization (DPO) using **Variational Quantum Circuits (VQCs)** as Quantum Neural Networks (QNNs) in Python 3.12.6.

The core problem: given the last 30 days of log-returns across 15 assets, output a portfolio weight vector that maximizes risk-adjusted return. Several approaches are compared:

| Model | Type | Description |
|---|---|---|
| Equal Weights | Baseline | Fixed 1/N allocation |
| Mean-Variance Optimization | Classical | Closed-form quadratic solver (Markowitz) |
| DDPG | Classical RL | MLP actor-critic |
| QDPG | Quantum RL | VQC actor with amplitude encoding |
| QDPG (Angle Encoding) | Quantum RL | VQC actor with PCA + angle encoding *(this fork)* |
| Deep Q-Learning | Classical RL | MLP Q-network |
| Quantum Q-Learning | Quantum RL | VQC Q-network |

Performance is evaluated on a 15-asset portfolio (US equities, ETFs, bonds, commodities) over 2011–2025 using 7-fold time-series cross-validation, reporting annualized profit and Sharpe ratio under both static (SPO) and dynamic (DPO, rebalancing every 30 days) strategies.


## Contribution: Angle Encoding via PCA

The original QDPG uses **amplitude encoding** to load the 450-feature input (15 assets × 30 days) into the quantum circuit. Amplitude encoding normalizes the input vector into quantum probability amplitudes — a process that blurs the relative scale differences between individual asset returns and requires a deep state-preparation circuit.

This fork adds a **QDPG (Angle Encoding)** variant that encodes data differently:

1. **PCA** reduces the 450-dimensional flattened return window to 15 principal components (fit per cross-validation fold on training data).
2. **Angle encoding** maps each of the 15 components directly to a qubit rotation angle (one Ry gate per qubit), producing a shallower circuit where features remain separated rather than superimposed.

Both models use 15 qubits and 60 trainable weights, making the comparison controlled. The hypothesis: preserving feature separability at the encoding stage — instead of compressing everything into probability amplitudes — changes what the variational circuit can learn about portfolio structure.

The critic network in this variant is a classical MLP (hybrid quantum-classical architecture), as the critic's concatenated state-action input is incompatible with the PCA transform fitted on state vectors alone.

**Changes from the original codebase:**
- `predictors/input_transformations.py` — added `PCATransform` class
- `predictors/quantum_neural_network.py` — added `transformed_input_size` parameter to support dimension-reducing transforms
- `ddpg/ddpg_functions.py` — added `critic_predictor` / `critic_predictor_kwargs` to allow separate actor and critic network types
- `MAIN.py` — added `QDPG (Angle Encoding)` pipeline


## Requirements

- Python 3.12.6
- Core libraries: `pennylane`, `torch`, `scikit-learn`
- Full dependency list: `requirements.txt`


## Installation

```bash
git clone https://github.com/VincentGurgul/qrl-dpo-public
cd qrl-dpo-public
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## Quick Start

```bash
python MAIN.py
```

To run only specific models, edit `RUN_MODELS` in `MAIN.py`. Note: quantum models can take an hour or more per full run across all CV folds. For quick testing, set `CV_SPLITS` to 1 or 2.


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
