# TODO

## Performance — Speed up quantum model training (currently 2+ hrs/model)

Root cause: 15-qubit CPU state-vector simulation (2^15 = 32,768-dim complex vector) × ~2.4M circuit evaluations per full run.

- [ ] **Enable early stopping** — `early_stopping=False` is hardcoded in the QDPG (Angle Encoding) block in `MAIN.py`. Switch to `True`; the patience/min_delta params are already there.
- [ ] **Increase batch size** — `batch_size=1` in `MAIN.py` forces a full gradient update per training window (~1,370/epoch). Try 16 or 32 to cut gradient steps ~32×.
- [ ] **Fix hardcoded replay buffer sample size** — `replay_buffer.sample(1)` in `ddpg/ddpg_functions.py` line 349 is hardcoded to 1 regardless of batch_size, negating the above for that step.
- [ ] **Reduce `num_weights`** — currently 60. Try 15 or 30; fewer weights = fewer rotation gates per circuit = faster forward/backward pass.
- [ ] **Reduce `CV_SPLITS`** for experimentation — drop from 7 to 3 when iterating quickly; restore to 7 for final runs.
