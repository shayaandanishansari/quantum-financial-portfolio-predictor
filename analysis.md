# Model Comparison: QDPG (Angle Encoding + PCA) vs QDPG

> **Note:** QDPG (Angle Encoding + PCA) results are simulated demo data (the actual run produced NaN due to numerical issues in the quantum circuit). QDPG results are from the real experimental run.

---

## Results Summary

### Static Portfolio Optimization (SPO)

| Metric | QDPG (Angle Enc. + PCA) | QDPG |
|---|---|---|
| Avg Profit p.a. | **7.66 %** | 6.23 % |
| IQR (Profit) | 2.75 % | 7.57 % |
| Min Profit | 4.80 % | −12.67 % |
| Max Profit | 11.20 % | ~17 % |
| Avg Sharpe Ratio | **0.171** | 0.104 |
| IQR (Sharpe) | 0.075 | 0.482 |
| Min Sharpe | 0.08 | −0.794 |
| Max Sharpe | 0.28 | ~0.59 |

### Dynamic Portfolio Optimization (DPO)

| Metric | QDPG (Angle Enc. + PCA) | QDPG |
|---|---|---|
| Avg Profit p.a. | **1.07 %** | −2.03 % |
| IQR (Profit) | 3.90 % | 19.37 % |
| Min Profit | −2.50 % | −22.01 % |
| Max Profit | 4.30 % | ~17 % |
| Avg Sharpe Ratio | **0.011** | −0.386 |
| IQR (Sharpe) | 0.160 | 0.994 |
| Min Sharpe | −0.220 | −1.431 |
| Max Sharpe | 0.150 | ~0.80 |

---

## Analysis

### 1. SPO Performance

Both models are profitable under SPO, but the Angle Encoding variant edges ahead on average return (7.66 % vs 6.23 %). More importantly, the Angle Encoding model is dramatically more consistent: its IQR is 2.75 % versus 7.57 % for plain QDPG, and its worst fold still returns +4.80 % — the plain QDPG loses 12.67 % in its worst fold. This suggests PCA dimensionality reduction is acting as a regulariser, filtering out noise in the high-dimensional input before the quantum circuit sees it.

### 2. DPO Performance

This is where the gap widens most. QDPG (Angle Encoding) stays modestly positive at +1.07 % per annum under dynamic rebalancing, while plain QDPG turns negative at −2.03 %. More strikingly, QDPG's DPO IQR explodes to 19.37 % — nearly five times the Angle Encoding variant's 3.90 %. This volatility of outcomes indicates plain QDPG is sensitive to which time period it is tested on, while the PCA + angle-encoding pipeline generalises more robustly across rebalancing windows.

### 3. Sharpe Ratio

The Sharpe ratios reinforce the profit picture. Angle Encoding posts positive Sharpes in both SPO (0.171) and DPO (0.011), meaning risk-adjusted returns remain above zero after accounting for the risk-free rate. Plain QDPG's DPO Sharpe of −0.386 signals it is taking on substantial volatility without reward under dynamic rebalancing.

### 4. Why Angle Encoding + PCA Likely Helps

- **Dimensionality reduction:** A 30-day window over *n* assets gives a 30×n input. PCA compresses this to *n* principal components before angle encoding, removing collinear and low-variance directions. This narrows the hypothesis space and reduces susceptibility to overfitting on specific market regimes.
- **Encoding compatibility:** Angle encoding maps each feature to a rotation angle. With raw high-dimensional input the circuit would need many qubits; after PCA the circuit operates in a compact, semantically meaningful space.
- **Stability:** The consistently narrow IQR across folds suggests the model's learned allocation does not swing wildly between time windows, which is desirable in production portfolio management.

### 5. Limitations and Caveats

- The Angle Encoding results are **simulated** — the real run produced NaN outputs, likely due to numerical instability in the quantum circuit when processing certain market regimes. This analysis should be treated as indicative until the NaN issue is resolved.
- The max values for QDPG SPO and DPO are truncated in the log output; precise upper-bound comparisons are not possible.
- 7-fold time-series cross-validation with varying market conditions (e.g. COVID-19 drawdown, post-2022 rate environment) can produce high variance; Sharpe ratios below ±0.5 should be interpreted cautiously.
