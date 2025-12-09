# MLB Bullpen Strategy

Offline reinforcement learning and constrained policy evaluation for deciding **when to pull a starting pitcher and which reliever to use**, using Statcast data from recent MLB seasons.

## What This Repo Contains
- `src/data/`: utilities to turn pitch-level Statcast into plate-appearance decision points, build bullpen availability/form features, and compute run-expectancy rewards with the three-batter minimum folded in.
- `src/rl/`: offline RL implementations (dueling DQN, Conservative Q-Learning, tabular Q) plus dataset and training utilities driven by YAML configs.
- `src/ope/`: offline policy evaluation helpers (TD error sweeps, greedy policy stats, Q-value diagnostics) for trained DQN/CQL checkpoints.
- `configs/`: knobs for data layout (`data.yaml`), environment/reward shaping (`env.yaml`), model hyperparameters (`model.yaml`), training/inference defaults.
- `notebooks/`: end-to-end notebooks to build the dataset, train the different agents, and run sanity/EDA checks.
- `models/`: example trained checkpoints (`*.pt`, `*.npz`) for 2022–2023 data.

## Problem Setup
- **State**: inning/half, outs, base state, score diff, pitch count, times-through-order, platoon matchup flags, next hitters (windowed embeddings + positional encodings), and reliever feature blocks. State is flattened in `BullpenOfflineDataset` (`src/rl/dqn.py`, `src/rl/cql.py`).
- **Actions**: `0 = stay with current pitcher`, `1..R = choose reliever r` (R set in `configs/model.yaml` and derived from the dataset’s availability mask).
- **Reward**: negative change in run expectancy from the fielding team’s view, folded over a 3-batter SMDP window with a small pull penalty (see `src/data/reward.py`; horizons in `configs/env.yaml`).
- **Behavior data**: Logged Statcast decisions determine availability masks (who could pitch) and the behavior action labels used for offline RL.

## Quick Start
1) **Environment**
   - Python 3.11+ (repo uses a `.venv` created with 3.11.5).
   - Install deps (minimal): `pip install torch pandas numpy pyyaml pybaseball matplotlib scikit-learn`. Add Jupyter/nbformat for notebooks.

2) **Build the dataset**
   - Run `notebooks/01_build_dataset.ipynb` (and `01_constrained_dataset.ipynb` for bullpen constraints) to pull Statcast via `pybaseball`, construct PAs, attach lineup/availability features, and emit `rl_tensors_*.npz` as defined in `configs/data.yaml`.

3) **Train a policy**
   - Open one of the training notebooks (e.g., `02_train_dqn.ipynb`, `02_train_constrained_dqn.ipynb`, `02_train_cql.ipynb`, `02_train_qlearn.ipynb`) and point them at your dataset file.
   - Or script it directly:
     ```python
     from pathlib import Path
     from src.rl.dqn import load_dqn_training_config, train_dqn

     cfg = load_dqn_training_config(
         model_config_path=Path("configs/model.yaml"),
         data_path=Path("data/processed/rl_tensors_2022_2023.npz"),
     )
     model = train_dqn(cfg)
     Path("models/my_dqn.pt").parent.mkdir(parents=True, exist_ok=True)
     torch.save(model.state_dict(), "models/my_dqn.pt")
     ```

4) **Offline evaluation**
   - Compute TD error, policy-vs-behavior stats, and Q distributions:
     ```python
     from pathlib import Path
     from src.ope.offline_eval import (
         OfflineEvalConfig, load_model_and_dataset,
         evaluate_td_error_full_mse, compute_policy_behavior_stats,
         summarize_policy_behavior_stats, compute_q_distributions,
         summarize_q_distributions,
     )

     cfg = OfflineEvalConfig(
         model_config_path=Path("configs/model.yaml"),
         model_path=Path("models/deep_dqn_bullpen_2022_2023.pt"),
         tensors_path=Path("data/processed/rl_tensors_2022_2023.npz"),
     )
     model, ds, loader = load_model_and_dataset(cfg)
     print("Full-dataset TD MSE:", evaluate_td_error_full_mse(model, loader, cfg.gamma, cfg.device))
     summarize_policy_behavior_stats(compute_policy_behavior_stats(model, loader, cfg.device))
     summarize_q_distributions(compute_q_distributions(model, loader, cfg.device))
     ```
   - For CQL, swap in `src/ope/offline_eval_cql.py` and a CQL checkpoint.

5) **Inference / decision thresholds**
   - `configs/inference.yaml` holds the checkpoint path, masking behavior, and optional stay/pull thresholds for turning Q-values into bullpen calls. The notebooks show how to map pitcher IDs to names for reporting.

## Data & Feature Notes
- Data sources are pure `pybaseball` calls (`data/datacard.md` lists APIs). Lineups and batting order are inferred from the plate-appearance sequence.
- Key feature builders:
  - `src/data/pa_builder.py`: PA grouping, base/out state, handedness flags, pitch count, times-through-order.
  - `src/data/lineup_builder.py` and `src/data/availability.py`: who is active/available in the bullpen.
  - `src/data/bullpen_form.py`: recent workload/form features for relievers.
  - `src/data/reward.py`: run-expectancy aggregation and SMDP reward folding (three-batter minimum).
- All tensors are stored in `.npz` with `state_vec`, `next_hitters_feats`, `pos_enc`, `reliever_feats`, `avail_mask`, `action_idx`, `reward_folded`, `done`, etc.; see `BullpenOfflineDataset` for exact expectations.

## Repository Layout
- `configs/` – YAML configs for data/model/env/training/inference.
- `data/` – local cache folders plus `datacard.md` for sources.
- `models/` – example checkpoints (DQN, constrained DQN/CQL, tabular Q).
- `notebooks/` – dataset build, training, and EDA runs.
- `src/data/` – Statcast ETL, feature engineering, reward folding.
- `src/rl/` – RL agents and training loops.
- `src/ope/` – offline policy evaluation helpers.
- `tests/` – placeholder for future unit tests.

## Reproducing Results
- Use the same `configs/*.yaml` values committed here to recreate the 2022–2023 runs.
- Start with `model_type: dqn` or `model_type: cql` in `configs/model.yaml`; checkpoint names in `models/` match those settings (`deep_dqn_bullpen_2022_2023.pt`, `constrained_cql_model_2022_2023.pt`, etc.).
- For constrained variants, use the `01_constrained_dataset.ipynb` + `02_train_constrained_*` notebooks so availability masks honor bullpen rules.

## Contributing / Next Steps
- Add lightweight unit tests under `tests/` for reward folding and dataset shapes.
- Wire a small CLI around the training/eval helpers to remove notebook dependency.
- Extend inference to simulate a full game with configurable matchup/pull thresholds.

## License
MIT License (see `LICENSE`).
