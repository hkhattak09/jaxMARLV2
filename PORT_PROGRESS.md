# CTM JAX Port Progress Tracker

## Status: Testing Phase

### ✅ Completed Steps
*   **Step 1:** Created `SuperLinear` module (NLM core) in `smax_ctm/ctm_jax.py`.
*   **Step 2:** Created `NLM` module (trace processor) in `smax_ctm/ctm_jax.py`.
*   **Step 3:** Created `CTMBackbone` module in `smax_ctm/ctm_jax.py`.
*   **Step 4:** Created `Synapses` module in `smax_ctm/ctm_jax.py`.
*   **Step 5:** Implemented `compute_synchronisation` function in `smax_ctm/ctm_jax.py`.
*   **Step 6:** Created `CTMCell` and `ScannedCTM` (single-step and scanned modules) in `smax_ctm/ctm_jax.py`.
*   **Step 7:** Created `ActorCTM` module in `smax_ctm/train_mappo_ctm.py`.
*   **Step 8:** Modified `train_mappo_ctm.py` (wiring, config, hidden state shapes, minibatch slicing).
*   **Test Script:** Created `smax_ctm/run_tests.py` with correct path injection for Colab testing.

### ⏳ Pending Steps
*   **Step 8 (Testing):** User to run `smax_ctm/run_tests.py` on Google Colab and report results.
*   **Step 9:** Smoke test training on Colab (running `train_mappo_ctm.py` with tiny config, then full config).

## Notes
- JAX 0.7.2 constraints considered (e.g., no `LazyLinear`, explicit GLU, correct tuple scan carry).
- Path resolution handled for Colab execution (`_REPO_ROOT` injection).
