# Simulation LSTM-VHO (VLC/WLAN Hybrid Network)

**Hybrid Multi-Objective Optimization and LSTM-based Vertical Handover for VLC/WLAN Hybrid Networks**

This repository implements and extends the Location-Aware Vertical Handover
(LA-VHO) algorithm for hybrid Visible Light Communication (VLC) / Wireless
Local Area Network (WLAN) indoor networks, originally proposed by Zeshan and
Baykas. On top of a faithful reproduction of that baseline, this project
adds a **hybrid multi-objective evolutionary optimizer** and an **LSTM-based
handover policy** trained to imitate/approximate the optimized decisions,
enabling fast, low-complexity, data-driven vertical handover at inference
time.

---

## 1. Background

The rapid growth of mobile devices and bandwidth-intensive applications has
significantly increased the demand for high-speed, reliable, and seamless
wireless communication in indoor environments. Most wireless data traffic is
now generated indoors, putting increasing pressure on conventional Radio
Frequency (RF) technologies such as WLAN, which — despite offering ubiquitous
coverage and robust connectivity — suffer from spectrum scarcity, growing
interference, and limited capacity as user density increases.

Visible Light Communication (VLC) has emerged as a promising complementary
technology for next-generation indoor wireless communication. By reusing
existing LED lighting infrastructure as optical access points, VLC offers
abundant unlicensed spectrum, high data rates, low electromagnetic
interference, enhanced security, and energy efficiency. However, VLC
performance is strongly constrained by its Line-of-Sight (LoS) requirement:
user mobility, body shadowing, and temporary blockages of the light path can
rapidly degrade link quality, making uninterrupted connectivity difficult to
guarantee on its own.

**Hybrid VLC/WLAN networks** address this by combining the high-capacity,
low-interference transmission of VLC with the wide coverage and robustness
of WLAN. In such heterogeneous networks, the **Vertical Handover (VHO)**
mechanism is responsible for dynamically selecting the most appropriate
access technology according to network conditions and user mobility.

Several VHO strategies exist in the literature — RSS-based approaches,
dwell-time mechanisms, fuzzy logic, Markov Decision Processes, machine
learning techniques, and multi-criteria optimization. Zeshan and Baykas
proposed a **Location-Aware Vertical Handover (LA-VHO)** algorithm for
hybrid VLC/WLAN networks that incorporates indoor localization into the
handover decision. Their method applies a threshold based on the number of
users in the environment and optimizes a weighted cost function via Gradient
Descent to select the most suitable network. It reduces unnecessary
handovers while maintaining acceptable QoE and packet loss — but the
decision process still relies on a limited weighted cost function and a
local optimization method, which struggles to capture the highly dynamic,
multi-objective nature of real heterogeneous wireless environments (the
handover decision simultaneously depends on conflicting indicators such as
SINR, throughput, packet loss, BER, delay, QoE, user density, mobility, and
blockage probability).

## 2. Project Goal

This project reframes the handover decision as a **Multi-Objective
Optimization (MOO)** problem and integrates a **hybrid evolutionary
optimizer** — combining multiple population-based multi-objective
algorithms — with a **Long Short-Term Memory (LSTM)** neural network that
learns to reproduce the optimized decisions from simulated trajectories.
Once trained, the LSTM performs handover decisions in real time with very
low computational overhead compared to running the optimizer online, while
being compared against the classical I-VHO, D-VHO, and LA-VHO baselines.

Reference baseline paper:

> A. Zeshan and T. Baykas, "Location Aware Vertical Handover in a VLC/WLAN
> Hybrid Network," _IEEE Access_, vol. 9, pp. 129810–129819, 2021.

---

## 3. Repository Structure

| File                     | Description                                                                                                                                                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `vho_sim.py`             | Core simulation engine: user mobility (fluid-flow model), VLC geometric coverage, optical shadowing/blocking, and the vertical handover schemes (`I-VHO`, `D-VHO`, `LA-VHO`, `LSTM-VHO`). Supports `log_decisions=True` to export a per-step, per-user decision dataset.       |
| `hybrid_algorithm.py`    | Hybrid multi-objective optimizer combining four population-based algorithms — MOWGA, NSGA-III (simplified), MOPSO, and MOWCA — with periodic elite exchange between populations.                                                                                               |
| `optimize_vho.py`        | Uses the hybrid optimizer to search handover-related parameters (e.g. dwell time, user-count threshold) against multiple objectives (handover count, packet loss), then re-runs the simulation with the selected parameters to generate the labeled handover-decision dataset. |
| `train_lstm_handover.py` | Trains an LSTM binary classifier on the generated dataset to predict the VLC/RF decision from a sliding window of past observations. Includes early stopping, training curves, confusion matrix, ROC and Precision-Recall curves, and classification metrics.                  |
| `lstm_policy.py`         | Lightweight inference wrapper that loads the trained LSTM model, scaler, and metadata, and exposes a vectorized `decide()` method used by `vho_sim.py` during simulation (`LSTM-VHO` scheme).                                                                                  |
| `run_experiments.py`     | Runs the comparative experiments (varying number of users and user velocity, across both room sizes) for all schemes — `I-VHO`, `D-VHO(0.5s)`, `D-VHO(1s)`, `LA-VHO`, and `LSTM-VHO` — and generates the comparison figures.                                                   |
| `output/`                | Generated artifacts: datasets (`handover_decisions_dataset.csv`), trained model (`lstm_vho_model.pt`), scaler and metadata, training/evaluation plots, and the final comparison figures.                                                                                       |

---

## 4. What Was Faithfully Implemented (baseline reproduction)

- **Layout**: building 16m×16m×3m; rooms of 5.6m×5.6m×3m (4 VLC APs) and
  8m×8m×3m (9 VLC APs); VLC coverage radius per AP = 1.5 m (as in the
  reference paper's system table).
- **User mobility**: speed in the 0.2–1 m/s range, uniformly random
  direction in [0, 2π], fluid-flow model with periodic direction changes.
- **Optical blocking probability**: increasing and saturating with the
  number of active users `Nu`, matching the qualitative trend of the
  reference paper's blocking/BER degradation behavior.
- **I-VHO**: immediate handover as soon as the VLC link drops; switches back
  to VLC as soon as the link is restored (baseline "ping-pong" scheme).
- **D-VHO**: waits for a configurable dwell time (0.5 s or 1 s) before
  switching to RF, trading handover frequency for packet loss.
- **LA-VHO**: two-stage decision — (1) if `Nu` is greater than or equal to a
  threshold, WLAN is forced for the whole session (handovers → 0); (2)
  otherwise, a small hysteresis window represents the cost-function/gradient
  descent minimization described in the reference paper.
- **Metrics**: average handovers per iteration (`NVHO`), QoE (penalized by
  handover frequency and by outage while attempting to use a blocked VLC
  link), and packet loss (% of time in outage while the active mode is VLC).

## 5. What This Project Adds (extension beyond the reference paper)

- **Handover decision dataset**: `vho_sim.py` can log, at every simulation
  step and for every simulated user, the position, velocity, geometric VLC
  coverage, shadowing/blocking state, link availability, and the handover
  decision taken (`mode_before` / `action_switch` / `mode_after`).
- **Hybrid multi-objective optimizer** (`hybrid_algorithm.py` +
  `optimize_vho.py`): rather than the single weighted-cost gradient descent
  used in the reference LA-VHO, handover-related parameters are searched
  using a hybrid of multiple population-based multi-objective algorithms
  that periodically exchange elite (Pareto-optimal) solutions, jointly
  minimizing handover count and packet loss.
- **LSTM-based handover policy** (`train_lstm_handover.py` +
  `lstm_policy.py`): a 2-layer LSTM binary classifier is trained on sliding
  windows of the logged simulation state to predict the VLC/RF decision.
  Training uses a per-user train/validation/test split (to avoid leaking a
  user's trajectory across splits), early stopping on validation loss, and
  reports accuracy, precision, recall, F1-score, ROC-AUC, and a confusion
  matrix on the held-out test users.
- **`LSTM-VHO` scheme**: the trained network is plugged back into
  `vho_sim.py` as a fifth handover scheme, maintaining a per-user sliding
  window during simulation and falling back to `I-VHO` behavior during the
  initial warm-up period (before the window is full), so it can be directly
  compared against `I-VHO`, `D-VHO`, and `LA-VHO` in `run_experiments.py`.

---

## 6. Limitations and Transparency

- The reference paper does not publish source code or all of the hidden
  numerical parameters of its model (exact `w_u`/`w_s` weights, delay cost
  `d_c`, the closed-form blocking probability beyond its symbolic equation,
  etc.). The blocking probability and QoE models here are physically
  coherent but heuristically calibrated — **exact percentage values from the
  paper are not reproducible** without those hidden parameters. The
  **qualitative trends** are preserved: `I-VHO` has the highest handover
  count and worst QoE (ping-pong effect) but the lowest packet loss;
  larger-dwell `D-VHO` reduces handovers at the cost of more packet loss;
  `LA-VHO` dominates in handover count and QoE, with handovers dropping to
  zero once `Nu` exceeds the threshold.
- Simulations use a reduced number of iterations/duration per point compared
  to the reference paper's setup, in order to run in seconds rather than
  hours; values already converge statistically at this scale.
- The current hybrid optimizer implements **MOWGA, NSGA-III, MOPSO, and
  MOWCA**. Note that this differs from the four-algorithm combination
  described in the original project proposal, which also considered a
  Multi-Objective Whale Optimization Algorithm (MOWOA); MOWGA is used here
  in its place.
- Dataset generation and the multi-objective optimization run entirely in
  Python/NumPy (via `vho_sim.py` and `hybrid_algorithm.py`), rather than in
  MATLAB.
- The LSTM is trained to imitate the decisions produced by the simulation
  under the optimizer-selected parameters; it is therefore only as good as
  the coverage and diversity of the scenarios (number of users, room,
  velocity range) used to generate the training dataset.

---

## 7. How to Run

```bash
# 1. Generate the labeled handover-decision dataset using the hybrid
#    multi-objective optimizer
python3 optimize_vho.py

# 2. Train the LSTM handover policy on the generated dataset
python3 train_lstm_handover.py

# 3. Run the comparative experiments (I-VHO, D-VHO, LA-VHO, LSTM-VHO)
python3 run_experiments.py
```

This produces, under `output/`:

- `handover_decisions_dataset.csv` — labeled dataset of handover decisions.
- `lstm_vho_model.pt`, `lstm_vho_scaler.pkl`, `lstm_vho_meta.json` — trained
  LSTM policy artifacts.
- `lstm_training_curves.png`, `lstm_confusion_matrix.png`,
  `lstm_roc_curve.png`, `lstm_pr_curve.png` — LSTM training/evaluation
  plots.
- `fig5_nvho.png`, `fig6_qoe.png`, `fig7_packetloss.png` — comparative
  figures across all handover schemes.

---

## 8. Results

<!--
  Add the final comparison figures and a short discussion here once the
  experiments have been run, e.g.:

  ### Average number of handovers vs. number of users
  ![NVHO](output/fig5_nvho.png)

  ### QoE comparison
  ![QoE](output/fig6_qoe.png)

  ### Packet loss comparison
  ![Packet loss](output/fig7_packetloss.png)

  ### LSTM-VHO training and evaluation
  ![Training curves](output/lstm_training_curves.png)
  ![Confusion matrix](output/lstm_confusion_matrix.png)
  ![ROC curve](output/lstm_roc_curve.png)

  Discuss here how LSTM-VHO compares against I-VHO / D-VHO / LA-VHO in
  terms of handover count, QoE, and packet loss, and summarize the LSTM
  test-set classification metrics (accuracy, precision, recall, F1,
  ROC-AUC).
-->

_(Results to be added.)_

---

## 9. References

1. A. Zeshan and T. Baykas, "Location Aware Vertical Handover in a VLC/WLAN
   Hybrid Network," _IEEE Access_, vol. 9, pp. 129810–129819, 2021.
2. K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization
   Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
   Part I: Solving Problems with Box Constraints," _IEEE Transactions on
   Evolutionary Computation_, vol. 18, no. 4, pp. 577–601, 2014. (NSGA-III)
3. C. A. Coello Coello and M. S. Lechuga, "MOPSO: A Proposal for Multiple
   Objective Particle Swarm Optimization," _Proceedings of the 2002
   Congress on Evolutionary Computation (CEC'02)_, vol. 2, pp. 1051–1056, 2002. (MOPSO)
4. H. Eskandar, A. Sadollah, A. Bahreininejad, and M. Hamdi, "Water Cycle
   Algorithm – A Novel Metaheuristic Optimization Method for Solving
   Constrained Engineering Optimization Problems," _Computers & Structures_,
   vol. 110–111, pp. 151–166, 2012. (Water Cycle Algorithm)
5. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," _Neural
   Computation_, vol. 9, no. 8, pp. 1735–1780, 1997. (LSTM)
6. E. Zitzler, K. Deb, and L. Thiele, "Comparison of Multiobjective
   Evolutionary Algorithms: Empirical Results," _Evolutionary Computation_,
   vol. 8, no. 2, pp. 173–195, 2000. (ZDT benchmark suite used in
   `optimization_hybrid.py`)
