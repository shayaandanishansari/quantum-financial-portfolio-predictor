# Running on Google Colab

## What you need
- The repo link (everything including data is in the repo)
- A Google account
- Ideally **Colab Pro** — the full run takes several hours and free Colab disconnects after ~12 hours / 90 min idle. Free tier may work if you keep the tab active.

---

## Step 1 — Open Colab and enable GPU

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. **Runtime → Change runtime type → T4 GPU** (or A100 if on Pro) → Save

---

## Step 2 — Clone the repo

In the first cell:

```python
!git clone https://github.com/shayaandanishansari/quantum-financial-portfolio-predictor
%cd qrl-dpo-public
```

---

## Step 3 — Install dependencies

```python
!pip install -r requirements.txt
!pip install pennylane-lightning[gpu]
```

The second line installs the GPU-accelerated quantum simulator. It's a large install (~1 GB) — let it finish before continuing.

---

## Step 4 — Switch to the GPU simulator

The code defaults to a slow CPU-only quantum simulator. Run this to swap it to the GPU version:

```python
with open('predictors/quantum_neural_network.py', 'r') as f:
    content = f.read()

content = content.replace(
    "qml.device('default.qubit', wires=self.num_qubits, shots=None)",
    "qml.device('lightning.gpu', wires=self.num_qubits)"
)

with open('predictors/quantum_neural_network.py', 'w') as f:
    f.write(content)

print("Done — GPU simulator active")
```

---

## Step 5 — Run

```python
!python MAIN.py
```

This will print progress to the cell output as it runs. The full 7-fold run across all models takes several hours — quantum models are the slow ones.

### Optional: run only the new model first

If you want results for just the new QDPG (Angle Encoding) model quickly, edit `MAIN.py` before running:

```python
with open('MAIN.py', 'r') as f:
    content = f.read()

content = content.replace(
    """RUN_MODELS = [
    'QDPG (Angle Encoding)',
    'Equal Weights',
    'Mean Variance Optimization',
    'DDPG',
    'QDPG',
    'Deep Q-Learning',
    'Quantum Q-Learning',
]""",
    """RUN_MODELS = [
    'QDPG (Angle Encoding)',
]"""
)

with open('MAIN.py', 'w') as f:
    f.write(content)

print("Done — will only run QDPG (Angle Encoding)")
```

---

## Step 6 — Get the results

Results are saved to `results_logs/` as the run progresses. After it finishes (or at any point), download the log:

```python
from google.colab import files
import glob

log_files = sorted(glob.glob('results_logs/*.log'))
files.download(log_files[-1])  # downloads the most recent log
```

---

## Tips

- **Don't let the tab go idle** on free Colab — it will disconnect and kill the run
- If it disconnects mid-run, you lose all progress (results are only saved per-fold as they complete, so a partial log may exist in `results_logs/`)
- On Colab Pro, enable **background execution** (Runtime menu) so it keeps running even if you close the tab
- The display output freezing for a long time between progress bar updates is normal — quantum models take 30–90 min per fold
