# ⚡ Autonomous Power Grid Defense Agent

> **800:1 class imbalance. 19x performance lift. One agent defending a live power grid.**

A full end-to-end research prototype for autonomous power grid risk detection and mitigation, built on the [L2RPN WCCI 2020](https://l2rpn.chalearn.org/) benchmark — the standard competition environment for learned power grid control. The system spans temporal deep learning, ensemble ML, retrieval-augmented generation, LLM-ranked decision making, and physics-verified action execution.

---

## Table of Contents

- [Project Goals](#project-goals)
- [High-Level Architecture](#high-level-architecture)
- [Repository Structure](#repository-structure)
- [Environment & Grid](#environment--grid)
- [Phase 1 — Data Collection & Feature Engineering](#phase-1--data-collection--feature-engineering)
- [Phase 2 — Mamba Temporal Encoder Training](#phase-2--mamba-temporal-encoder-training)
- [Phase 3 — XGBoost Classifier Chain](#phase-3--xgboost-classifier-chain)
- [Phase 4 — Agentic Copilot Pipeline](#phase-4--agentic-copilot-pipeline)
  - [Risk Detection](#risk-detection)
  - [SCADA Payload Builder](#scada-payload-builder)
  - [NERC RAG Retrieval](#nerc-rag-retrieval)
  - [LLM Action Ranking](#llm-action-ranking)
  - [Sandbox Physical Verifier](#sandbox-physical-verifier)
  - [Autonomous Loop](#autonomous-loop)
- [Key Design Decisions & Assumptions](#key-design-decisions--assumptions)
- [Results & Metrics](#results--metrics)
- [Visualizations](#visualizations)
- [Setup & Requirements](#setup--requirements)
- [Known Limitations](#known-limitations)

---

## Project Goals

Power grids operate under strict physical constraints. A single line overload can cascade into a full blackout within seconds — and unlike software failures, grid failures are physically irreversible in real time. The goal of this project is to explore whether a fully autonomous ML agent can:

1. **Detect** thermal overloads, voltage sags, and blackout precursors **before** they cascade, using temporal patterns learned from grid simulations
2. **Reason** over detected anomalies using domain-grounded NERC operational protocols
3. **Propose and rank** corrective actions using an LLM
4. **Verify** those actions in a physics-faithful simulator **before** committing them to the live grid
5. **Execute** verified interventions in a closed-loop fashion, continuing to monitor afterward

The system is built around two core beliefs: ML alone should never be the last line of defense in a safety-critical system, and temporal context is irreplaceable for distinguishing genuine stress trajectories from momentary noise.

Note that while this agent achieves a robust 19x performance lift over statistical baselines, real-world power grids are infinitely complex, nonlinear environments. This repository is not intended to be a fully-certified deployment for a NERC control room. Rather, it serves as an advanced architectural proof-of-concept and research study. It was built to demonstrate that pairing sequence models (Mamba) with calibrated discrete logic (XGBoost) and a deterministic isolation sandbox can safely navigate the extreme class imbalances inherent to safety-critical infrastructure.
---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LIVE GRID2OP ENVIRONMENT                     │
│                    (L2RPN WCCI 2020 — 59 lines, 37 loads, 22 gens)  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ obs (per timestep)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TIER 1: TEMPORAL ENCODER                         │
│                                                                     │
│  obs_buffer (deque, maxlen=128)                                     │
│  ┌──────────────────────────────────────────┐                       │
│  │  128 × 649-dim sequence                  │                       │
│  │  [rho, v_or, v_ex, p_or, q_or,           │                       │
│  │   line_status, overflow, load_p,          │                       │
│  │   gen_p, topo_vect]                       │                       │
│  └──────────────┬───────────────────────────┘                       │
│                 │  per-window z-score norm + clip[-5, 5]            │
│                 ▼                                                   │
│  ┌──────────────────────────────────────────┐                       │
│  │  TransformerEncoder (2 layers, 4 heads)  │  ← global attention   │
│  │  + Mamba2 SSM Backbone (4 layers)        │  ← temporal memory    │
│  │  → final hidden state: emb [1 × 128]     │                       │
│  └──────────────┬───────────────────────────┘                       │
└─────────────────┼───────────────────────────────────────────────────┘
                  │ embedding (128-dim temporal state vector)
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TIER 2: RISK CLASSIFIER                          │
│                                                                     │
│  XGBoost Classifier Chain (119 models, one per physical target)     │
│  Input: [emb (128) ‖ tab (26)] = 154-dim feature vector            │
│                                                                     │
│  Targets: thermal overload per line (59) +                         │
│           voltage sag per line (59) +                               │
│           system blackout (1) = 119 binary labels                  │
│                                                                     │
│  Calibrated via Platt scaling (CalibratedClassifierCV, sigmoid)     │
│  Chain order is randomized for correlation awareness               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ binary predictions per label
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TIER 3: PHYSICS SAFETY NET                       │
│                                                                     │
│  physics_scalar = count(obs.rho >= 0.75)                           │
│  scalar = max(xgb_anomaly_count, physics_scalar)                   │
│                                                                     │
│  ← ML provides early warning; physics provides guaranteed floor     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ scalar >= 0.99 → THREAT DETECTED
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TIER 4: AGENTIC COPILOT                          │
│                                                                     │
│  1. Build enterprise SCADA JSON payload (physics-grounded)          │
│  2. Retrieve NERC protocols via FAISS RAG (TOP-001, EOP-011)        │
│  3. Generate Top-K heuristic action candidates (ML-proposed)        │
│  4. LLM (Gemini) ranks candidates against NERC protocols            │
│  5. Sandbox verifier: obs.simulate(action) → physics penalty delta  │
│  6. Execute if delta > 0 (grid physically improved)                 │
│  7. Log mitigated state into Mamba buffer (no amnesia)              │
│  8. Repeat up to 3 attempts with failure history fed back to LLM    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── THE_Grid.ipynb          # Phase 1-3: Data collection, Mamba training, XGBoost training
├── Agentic_Grid.ipynb      # Phase 4: Agentic Copilot, visualizations, SHAP analysis
├── grid_models/
│   ├── best_model.pt       # Saved TF-Mamba weights (best val AP checkpoint)
│   ├── meta.json           # Grid topology metadata and feature slice indices
│   ├── X_train.npy         # Mamba embeddings + tab features (train)
│   ├── X_test.npy          # Mamba embeddings + tab features (test)
│   ├── y_train.npy         # Binary labels (train)
│   └── y_test.npy          # Binary labels (test)
├── faiss_grid_index/       # FAISS vector store of NERC protocol documents
├── assets/
│   ├── agent_behavior_trace.png
│   ├── grid_heatmap_final.png
│   └── shap_xgb.png
└── README.md
```

---

## Environment & Grid

**Environment:** `l2rpn_wcci_2020` via [Grid2Op](https://github.com/rte-france/Grid2Op) with [LightSim2Grid](https://github.com/BDonnot/lightsim2grid) C++ backend for fast physics simulation.

**Grid topology:**
| Property | Value |
|---|---|
| Transmission lines | 59 |
| Substations | 36 |
| Loads | 37 |
| Generators | 22 (8 solar, 4 wind, 8 thermal, 1 hydro, 1 nuclear) |
| Topology vector dim | 177 |
| Sequence feature dim | **649** |

The 649-dim sequence vector per timestep is composed as:
```
rho (59) + v_or (59) + v_ex (59) + p_or (59) + q_or (59)
+ line_status (59) + timestep_overflow (59)
+ load_p (37) + gen_p (22) + topo_vect (177) = 649
```

**Chronics** are pre-recorded operational scenarios (load/generation time series) that drive the simulation. The environment contains multiple chronics representing different demand profiles across time of day, season, and weather conditions.

---

## Phase 1 — Data Collection & Feature Engineering

**Notebook:** `THE_Grid.ipynb`

### Collection Process

The data collection loop iterates over all available chronics, stepping through each with a do-nothing action (`env.action_space({})`) to capture the natural grid evolution under no control. At every timestep, two parallel feature vectors are extracted:

**Tier 1 — Sequence features** (`extract_sequence_features`): The 649-dim dense array described above. This is designed to preserve the full physical state of the grid at each timestep for temporal modeling.

**Tier 2 — Tabular features** (`extract_tabular_features`): 26 scalar summaries that collapse the grid state into human-readable operational metrics:

| Feature Group | Features |
|---|---|
| Weather proxies | `solar_cf`, `wind_cf` (capacity factors from gen_pmax ratios) |
| Grid stress | `rho_max`, `rho_mean`, `rho_std`, `n_overloaded_85`, `n_critical_95`, `min_voltage`, `mean_voltage`, `n_disconnected` |
| Operational state | `n_lines_cooldown`, `max_cooldown_time`, `n_subs_cooldown`, `n_lines_maintenance`, `time_since_last_attack` |
| Storage | `storage_charge_mean`, `storage_power_sum` |
| Load/gen balance | `load_p_sum`, `load_p_std`, `load_peak_ratio`, `gen_p_sum`, `load_gen_ratio` |
| Temporal | `hour_sin`, `hour_cos` (cyclical encoding), `day_of_week`, `month` |

### Label Assignment

Labels are assigned with a 4-step **prediction horizon** — the model is trained to predict risk 4 timesteps ahead, not at the current moment. This makes the system a genuine early-warning detector rather than a reactive classifier.

Three binary label types per window, totaling **119 targets**:

- **Thermal overload** per line (59): `rho >= 0.85` at the future timestep
- **Voltage sag** per line (59): `v_or < 0.90` OR `v_ex < 0.90` at the future timestep
- **System blackout** (1): episode done flag triggered by grid2op physics

### Class Imbalance

The dataset is extremely imbalanced — a direct reflection of real grid operations where catastrophic events are rare by design:

| Label type | Mean frequency |
|---|---|
| Thermal overload | ~0.44% |
| Voltage sag | ~2.88% |
| System blackout | ~0.01% |

This corresponds to up to **800:1 negative-to-positive ratio** for blackout events. This imbalance is handled during training via `pos_weight` (capped at 10x to prevent gradient explosion) and evaluated using **Average Precision** rather than AUROC, since AUROC is misleadingly optimistic on heavily skewed datasets.

### Dataset Scale

- Target: 1,200,000 timesteps across all chronics
- Window size: 128 timesteps, stride: 16
- Split: 70% train / 15% val / 15% test (chronic-level split to prevent data leakage)

---

## Phase 2 — Mamba Temporal Encoder Training

**Notebook:** `THE_Grid.ipynb`

### Model Architecture: `GridRiskMamba` (TF-Hybrid)

The production model is a hybrid architecture combining a Transformer front-end with a Mamba SSM backbone:

```
Input: [B, 128, 649]
    ↓
LayerNorm → Linear(649→128) → LayerNorm    # input_proj
    ↓
TransformerEncoder(d_model=128, nhead=4,   # 2 layers, global attention
    dim_feedforward=256, batch_first=True)  # captures intra-window correlations
    ↓
Mamba2 × 4 layers (d_model=128,            # SSM backbone, temporal memory
    d_state=32, headdim=64)
    ↓
LayerNorm
    ↓
x[:, -1, :]  →  embedding [B, 128]         # final hidden state = temporal summary
    ↓                          ↓
class_head                 physics_head
(Linear→GELU→Dropout→      (Linear→GELU→
  Linear → [119 logits])     Linear → [n_line + 2 + n_line])
```

**Why Transformer + Mamba?** The Transformer prepend provides global attention across the 128-step window, allowing the model to correlate distant timesteps (e.g., a voltage dip 60 steps ago with a current rho spike). Mamba2 then processes this attention-enriched sequence with its selective state space mechanism, capturing the causal temporal trajectory efficiently without quadratic attention cost.

**Critical design note:** At inference time, **only the embedding output is used** from this model. The classification head (`class_head`) was trained purely to force the embedding space to encode risk-relevant temporal patterns via supervised signal. The actual risk classification at inference is performed by XGBoost operating on the embedding, not by the Mamba head directly.

### Physics-Informed Loss

Beyond standard binary cross-entropy with `pos_weight`, the loss includes two physics-grounded regularization terms:

**Kirchhoff regularization** (`lam_kirchhoff=0.01`): Penalizes the physics head's predicted power flows for violating conservation of power at each node — encouraging the model to learn physically consistent internal representations.

**Thermal regularization** (`lam_thermal=0.01`): Penalizes the physics head when it predicts low rho on lines that the sequence data shows as genuinely overloaded — enforcing consistency between the model's thermal predictions and the observable line loadings.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| d_model | 128 |
| n_layers | 4 |
| d_state | 32 |
| Batch size | 256 |
| Learning rate | 3e-4 (AdamW) |
| Weight decay | 1e-2 |
| Scheduler | CosineAnnealingLR |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Mixed precision | bf16 (via HuggingFace Accelerate) |

### Training Results

Best checkpoint selected by **Val Mean AP** (not AUROC, for the imbalance reasons stated above):

| Metric | Value |
|---|---|
| Best Val Mean AP | ~0.5993 |
| Val AUROC | ~0.993–0.998 |
| Val Accuracy | ~0.9989 |

The high accuracy (99.89%) reflects the dominant negative class — the model predicts safe correctly most of the time by volume. AP of 0.60 is the meaningful number, measuring precision-recall on the rare positive events specifically.

---

## Phase 3 — XGBoost Classifier Chain

**Notebook:** `THE_Grid.ipynb` (Phase 2), `Agentic_Grid.ipynb` (inference)

### Embedding Extraction

After Mamba training, the model is run in eval mode over the full train/val/test sets to extract embeddings. The XGBoost feature matrix is constructed as:

```
X = [mamba_embedding (128-dim) ‖ tabular_features (26-dim)] = 154-dim
```

This concatenation is deliberate: the Mamba embedding encodes the **temporal trajectory** of the grid (what has been happening over the past 128 steps), while the tabular features encode the **point-in-time physics** (what the grid looks like right now). Both signals are complementary and neither alone is sufficient.

### Manual Classifier Chain

Rather than using sklearn's built-in `ClassifierChain`, a manual chain is implemented for finer control over calibration:

```python
for i in range(n_targets):  # 119 targets
    model_i = CalibratedClassifierCV(XGBClassifier(...), method='sigmoid', cv=StratifiedKFold(3))
    model_i.fit(X_current, y[:, i])
    models.append(model_i)
    binary_pred = model_i.predict(X_current).reshape(-1, 1)
    X_current = np.hstack([X_current, binary_pred])  # augment for next model
```

Each model in the chain receives the original features **plus the binary predictions of all previous models** as additional inputs. This allows later models (e.g., blackout prediction) to condition on the predictions of earlier models (e.g., thermal overloads on specific lines), capturing inter-label correlations.

**Platt scaling calibration** (`method='sigmoid'`) is applied to each model via 3-fold stratified cross-validation, ensuring that the output probabilities are reliable estimates of actual risk likelihood rather than raw uncalibrated scores. This is critical for the physics-informed trigger logic downstream.

### XGBoost Configuration

```python
params = {
    'n_estimators': 150,
    'max_depth': 4,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda'  # GPU-accelerated training
}
```

### XGBoost Results

| Model | Macro AP |
|---|---|
| Stateless baseline (raw physics + tab, no temporal memory) | 0.0128 |
| Mamba-XGB ensemble (TF-hybrid embedding + tab) | **0.2399** |
| **Lift** | **~19×** |

The stateless baseline uses the same XGBoost chain architecture but replaces the Mamba embedding with the raw physics snapshot at timestep T — no temporal context whatsoever. The ~19x lift directly quantifies how much value the Mamba temporal encoder adds over pure point-in-time physics features.

---

## Phase 4 — Agentic Copilot Pipeline

**Notebook:** `Agentic_Grid.ipynb`

The `ProductionCopilot` class orchestrates the full agentic loop. It holds the live Grid2Op environment, the loaded Mamba model, the XGBoost chain, the FAISS vector store, and the LLM client as persistent state.

### Risk Detection

`detect_risk(obs, mutate_buffer)` is called at every timestep and performs the full three-tier risk assessment:

**Step 1 — Buffer management:** The observation is converted to a 649-dim sequence vector and appended to a rolling `deque(maxlen=128)`. During the first 128 steps (burn-in), the buffer is padded with copies of the current observation to reach full length. The `organic_steps` counter tracks how many genuine (non-padded) observations are in the buffer.

**Step 2 — Inference-time normalization:** The 128×649 buffer matrix is z-score normalized per feature column and clipped to `[-5, 5]`, exactly matching the `GridRiskDataset.__getitem__` normalization applied during training. Skipping this step causes Mamba to receive raw physical values (voltages in kV, power in MW) completely outside its trained input distribution, collapsing sigmoid outputs to near-zero regardless of grid state.

**Step 3 — Mamba forward pass:** The normalized sequence is passed through `_GridRiskMamba` in eval mode with `torch.no_grad()`. The **embedding** `x[:, -1, :]` (the final hidden state of the last timestep) is extracted. The classification logits are intentionally discarded at inference.

**Step 4 — XGB inference:** The embedding is concatenated with freshly computed tabular features from the current observation. This combined 154-dim vector is passed through the manual classifier chain sequentially, with each model's binary prediction appended before the next model runs. XGB inference is gated behind the burn-in guard: `organic_steps >= window_size` (128 steps). Before burn-in completes, XGB output is forced to zero to prevent garbage predictions on a padded buffer.

**Step 5 — Physics backstop:** `physics_scalar = count(obs.rho >= 0.75)`. The final risk scalar is `max(xgb_anomaly_count, physics_scalar)`. This ensures the agent always wakes up when physical line loading crosses the warning threshold, even if both ML models are uncertain.

### SCADA Payload Builder

`build_enterprise_scada_payload(obs, line_probs, scalar)` constructs a structured JSON snapshot of the grid state for LLM consumption:

```json
{
  "metrics": {"total_mw": 4521.3, "sys_risk": 3.0},
  "offline_lines": [12, 47],
  "critical_lines": [
    {"id": 23, "capacity_rho": 0.94, "substations": [8, 17], "neighboring_lines": [19, 24, 31]}
  ],
  "warning_lines": [...],
  "generators": [
    {"id": 5, "grid_location_substation": 12, "current_mw": 287.4, "max_mw": 350.0, "ramp_mw_per_step": 14.2}
  ],
  "diagnostics": {"n_lines": 59, "n_offline": 2, "n_critical": 1, "n_warning": 4}
}
```

The payload includes **neighboring line topology** for each critical line — substations sharing a connection with the overloaded line — so the LLM can reason about cascade propagation paths, not just the single overloaded line in isolation.

Generator entries are sorted by current output descending and include ramp limits, giving the LLM the physical constraints it needs to propose valid redispatch magnitudes.

### NERC RAG Retrieval

A FAISS vector store is pre-built from three domain documents:
- **NERC TOP-001**: Transmission Operations standard — N-1 contingency obligations and thermal relief timelines
- **NERC EOP-011**: Emergency Operations — blackout prevention protocols
- **L2RPN Challenge documentation**: Environment-specific operational context

At each mitigation attempt, a **dynamic query** is constructed from the current grid state:

```python
crisis_summary = (
    f"{len(critical_lines)} critical line(s) overloaded (rho>0.90): {critical_lines}. "
    f"Max rho: {obs.rho[critical_lines].max():.3f}. "
    f"Voltage stress: min_v_or={obs.v_or.min():.3f}. "
    f"Offline lines: {offline_lines}."
)
```

The top-2 most semantically similar document chunks are retrieved and prepended to the LLM prompt, grounding the action selection in actual regulatory requirements rather than generic power systems knowledge.

### LLM Action Ranking

The LLM (Gemini) receives:
1. The full SCADA JSON payload with generator ramp limits and neighbor topology
2. Retrieved NERC protocol passages
3. A numbered list of Top-K heuristic action candidates

The system prompt instructs the LLM to act as a **conservative safety filter**, not a creative agent — its job is to rank the ML-proposed candidates by NERC compliance and physical safety, not to invent new actions. Output is constrained to a structured JSON:

```json
{"selected_action_index": 2, "rationale": "NERC TOP-001 requires..."}
```

### Heuristic Action Candidates

`build_heuristic_actions(obs, critical_lines)` generates the candidate set that the LLM ranks:

1. **No-op** (always included as the safe default)
2. **Open the most overloaded line** — forces power redistribution through alternate paths
3. **Reconnect all offline lines** — one candidate per offline line (graveyard restoration)
4. **Conservative redispatch** — reduce the largest generator by 50% of its ramp-down limit
5. **Full ramp-down** — reduce the largest generator by 100% of its ramp-down limit

Redispatch magnitudes are **clamped to `gen_max_ramp_down`** before being passed to the LLM and again inside `convert_to_action` before execution. This prevents the LLM from selecting a physically impossible action (e.g., ramping 100MW on a machine with 14MW/step limit).

### Sandbox Physical Verifier

Before any action touches the live grid, `run_sandbox_verifier(obs, baseline_scalar, action)` performs a physics rollout:

```python
sim_obs, _, done, info = obs.simulate(action)   # Grid2Op 1-step physics simulation
```

The physics penalty is computed for both pre- and post-action states:

```python
def get_physics_penalty(o):
    return float(np.max(o.rho)) + float(np.max(np.abs(1.0 - o.v_or)) * 2)
```

This combines **thermal stress** (max line loading) and **voltage deviation** (distance from nominal 1.0 pu) into a single scalar. The voltage term is weighted 2x to reflect its faster cascade potential.

An action is approved only if `delta = baseline_penalty - post_penalty > 0` — meaning the grid is **physically better** after the action than before it. An action that makes the grid worse in any physical dimension is rejected, even if the LLM selected it confidently.

**Why only 1-step simulation?** Multi-step rollout requires loading future chronics which breaks Grid2Op's internal state machine. The 1-step simulator is used as a necessary approximation, with the understanding that it catches immediate failures (line trips, voltage collapse) but cannot foresee delayed cascade effects 10+ steps ahead. This is documented as a known limitation.

### Autonomous Loop

`run_autonomous_loop(max_steps)` is the top-level control loop:

```
for step in range(max_steps):
    1. Step environment with do-nothing action (observe natural grid evolution)
    2. detect_risk(obs) → scalar, line_probs
    3. If step < 129: continue (burn-in)
    4. If post_action_cooldown > 0: decrement and skip (prevent re-triggering)
    5. If step == 135: execute N-3 attack (stress test demo)
    6. If scalar >= 0.99: THREAT DETECTED
        a. build_heuristic_actions
        b. For up to 3 attempts:
            i.  build_enterprise_scada_payload
            ii. retrieve_nerc_protocols (dynamic query)
            iii. ask_llm → selected_action
            iv. run_sandbox_verifier
            v.  If approved: execute, log to buffer, set cooldown=10, break
            vi. Else: append failure to history, retry with updated context
        c. If all 3 attempts fail: halt and hold position
```

The **post-action cooldown** (10 steps) prevents the agent from re-triggering on a grid that is still in a warning state but physically improving. Without this, the agent would flood the grid with sequential actions before the physics of the first action has settled.

The **failure history accumulation** is a key design feature: failed sandbox proposals are appended to the SCADA payload on each retry, so the LLM sees what it already tried and why it was rejected, allowing it to select a different strategy on the next attempt.

---

## Key Design Decisions & Assumptions

**Mamba as encoder, not classifier.** The Mamba classification head is trained but discarded at inference. It exists solely to provide a supervised learning signal that forces the embedding space to encode risk-relevant temporal patterns. This is analogous to using a pre-trained vision model as a feature extractor — the classification head that drove training is not the component that gets deployed.

**Chain order is randomized.** The XGBoost classifier chain uses `order='random', random_state=42`. A fixed sequential order (e.g., always predicting thermal before voltage) would introduce ordering bias where early predictions systematically influence later ones. Randomization distributes this effect across the label space.

**Calibration is mandatory on imbalanced data.** Raw XGBoost probability outputs are poorly calibrated when trained on 800:1 imbalanced data — they systematically underestimate positive class probability. Platt scaling (sigmoid calibration via 3-fold CV) corrects the probability estimates to reflect actual empirical frequencies, which is essential when downstream logic depends on probability thresholds.

**Physics backstop is not a fallback — it is a feature.** The `rho >= 0.75` physics scalar is not a workaround for ML failure. It encodes the operational philosophy that a safety-critical system should have a guaranteed detection floor independent of model confidence. ML provides early warning on subtle patterns; physics provides certainty on overt stress.

**Inference normalization must match training.** The per-window z-score normalization applied in `GridRiskDataset.__getitem__` during training must be identically replicated in `detect_risk` at inference. Omitting this step causes Mamba to receive values 50-100x outside its trained input distribution, making all outputs meaningless regardless of model quality.

**Assumptions:**
- The simulation operates under a do-nothing baseline action (no topology changes between interventions), which means the observed risk reflects natural grid evolution plus any agent interventions.
- Generator redispatch is the primary corrective lever; topology changes (bus switching) are not in the current action space beyond line open/close.
- The NERC documents provide regulatory grounding but the LLM's use of them is semantic similarity-based, not formal rule execution.
- The 1-step physics sandbox catches immediate failures but does not guarantee stability over a longer horizon.

---

## Results & Metrics

### Model Performance

| Model | Macro AP | Notes |
|---|---|---|
| Stateless XGB baseline | 0.0128 | Raw physics + tab, no temporal memory |
| **Mamba-XGB TF-Hybrid** | **0.2399** | Temporal embedding + tab |
| **Lift** | **~19×** | |
| Mamba val AUROC | 0.993–0.998 | High due to dominant negative class |
| Mamba val AP | ~0.5993 | Honest metric under imbalance |

**Why AP over AUROC?** On a dataset with 0.01% blackout frequency, a model predicting zero for everything achieves 99.99% accuracy and 0.9999+ AUROC. AP measures precision and recall specifically on the positive class, where a model that always predicts zero scores 0.0. It is the correct metric when the positive class is the class that matters.

### SHAP Analysis

SHAP attribution on the most active XGB model (Line 0) shows that Mamba embedding dimensions appear consistently in the top 20 features alongside physics tab features (`rho_max`, `gen_p_sum`, `day_of_week`). The presence of `mamba_emb_124` at rank 4 with SHAP value 0.5659 — above `load_p_sum` — confirms the temporal encoder contributes signal beyond what instantaneous physics features capture alone.

---

## Visualizations

### Agent Behavior Trace
Two-panel plot showing risk scalar and physical rho over the full episode, annotated with burn-in boundary, N-3 attack, agent trigger points, and sandbox-approved action execution points.

### Grid Topology Risk Heatmap
NetworkX spring-layout graph of the 59-line, 36-substation grid with edges colored by line loading (green→yellow→red, RdYlGn colormap) and thickness proportional to rho. Offline lines shown as dashed grey. Node colors reflect the maximum rho of connected lines.

### SHAP Feature Importance
Horizontal bar chart of top-20 features by mean |SHAP value|, colored blue for Mamba embedding dimensions and red for physics/operational tab features, with value annotations on each bar.

---

## Setup & Requirements

```bash
# Grid2Op environment
pip install grid2op lightsim2grid

# PyTorch (CUDA 12.8)
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Mamba SSM
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --force-reinstall --no-deps
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# ML stack
pip install xgboost scikit-learn shap accelerate einops

# Agentic stack
pip install langchain-community langchain-huggingface faiss-cpu pypdf
pip install google-genai

# Visualization
pip install matplotlib networkx
```

**Hardware:** Training was performed on an A100 GPU (RunPod). Inference runs on any CUDA-capable GPU.

**Environment variables required:**
```bash
export GEMINI_API_KEY="your_key_here"
export URL_GRID2OP="your_google_drive_url"
export URL_CHRONICS="your_google_drive_url"
```

---

## Known Limitations

- **Sandbox is 1-step only.** Multi-step lookahead requires loading future chronics which is not natively supported in Grid2Op without resetting the environment. The 1-step simulator catches immediate failures but cannot foresee delayed cascade effects.
- **Mamba classification head underperforms on thermal events.** Due to 0.44% thermal label frequency, the Mamba head's per-line thermal sigmoid outputs collapse toward zero during training despite pos_weight. This is why the classification head is discarded at inference in favor of XGB on the embedding.
- **Post-mitigation cooldown is fixed at 10 steps.** Production deployment would require tuning this to the grid's physical settling time, which varies by network topology and the type of intervention applied.
- **LLM quota exhaustion.** Gemini API rate limits will terminate LLM-ranked action selection after sustained agent activity. The agent falls back to the do-nothing safe default when LLM calls fail.
- **Single chronic evaluation.** The autonomous loop demo runs on whichever chronic the environment loads at reset. Results may vary across chronics with different demand profiles.

---

*Built and evaluated on the L2RPN WCCI 2020 benchmark — the standard competition environment for learned power grid control.*
