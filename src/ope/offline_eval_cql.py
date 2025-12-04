from dataclasses import dataclass
from pathlib import Path
from typing import Tuple,Dict, Any

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.rl.cql import (
    load_cql_training_config,
    BullpenOfflineDataset,
    RLDatasetConfig,
    CQLNet,
    masked_q,
    evaluate_td_error,
)

@dataclass
class OfflineEvalConfig:
    """
    Configuration for offline policy evaluation of the CQL model.
    """
    model_config_path: Path
    model_path: Path
    tensors_path: Path
    device: str = 'cpu'
    batch_size: int =1024
    gamma: float =0.99

def load_model_and_dataset(cfg: OfflineEvalConfig) -> Tuple[CQLNet, BullpenOfflineDataset, DataLoader]:
    """
    Load:
      - training config from YAML (for architecture + hyperparams)
      - BullpenOfflineDataset from rl_tensors_*.npz
      - CQL model with saved weights
      - DataLoader over the full dataset (no train/val split)
    """
    train_cfg = load_cql_training_config(
        model_config_path=cfg.model_config_path,
        data_path=cfg.tensors_path,
        device=cfg.device
    )

    ds = BullpenOfflineDataset(
        RLDatasetConfig(
            data_path=cfg.tensors_path,
            device=cfg.device
        )
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    model = CQLNet(
        state_dim=ds.state_dim,
        num_actions=ds.num_actions,
        hidden_size=train_cfg.hidden_size,
        num_layers=train_cfg.num_layers,
        dropout=train_cfg.dropout
    ).to(cfg.device)

    state_dict = torch.load(cfg.model_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, ds, loader

@torch.no_grad()
def evaluate_td_error_full_mse(
    model: CQLNet,
    loader: DataLoader,
    gamma: float,
    device: str,
) -> float:
    """
    Mean Squared TD Error (MSTE) over the full dataset, using the
    existing evaluate_td_error function from cql.py.

    We simply pass the same network as both model and target,
    and use `loader` as the "val_loader".
    """
    # model is used for both current Q and target Q here
    return evaluate_td_error(
        model=model,
        target=model,
        val_loader=loader,
        gamma=gamma,
        device=device,
    )

## Direct Q-based value estimate of greedy policy (FQE)
@torch.no_grad()
def direct_policy_value_estimate(
    model:CQLNet,
    loader:DataLoader,
    device:str
) -> float:
    """
    Direct Method estimate of V(pi_greedy):

      For each state s in the dataset:
        - compute Q(s, a) for all actions
        - mask unavailable actions with `masked_q`
        - a_greedy = argmax_a Q_masked(s, a)
        - V_hat(s) = Q_masked(s, a_greedy)

      Return the average V_hat(s) over all states.

    NOTE:
      - This is a simple, intuitive estimate.
      - It can be optimistic if Q extrapolates beyond the behavior policy.
    """
    model.eval()

    total_v = 0.0
    total_n = 0

    for batch in loader:
        s,a,r,ns,d,m,m_next = batch

        s = s.to(device)
        m = m.to(device)

        q = model(s)
        q_masked = masked_q(q,m)
        a_greedy = q_masked.argmax(dim=1)

        v_hat = q_masked.gather(1, a_greedy.unsqueeze(1)).squeeze(1)

        total_v += v_hat.sum().item()
        total_n += v_hat.numel()

    if total_n == 0:
        return 0.0
    
    return total_v / total_n


@torch.no_grad()
def compute_policy_behavior_stats(
    model: CQLNet,
    loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """
    Compare the greedy CQL policy to the logged (behavior) policy.

    Returns:
      {
        "num_actions": int,
        "num_samples": int,
        "behavior_action_counts": np.ndarray [A],
        "policy_action_counts": np.ndarray [A],
        "valid_action_counts": np.ndarray [A],
        "behavior_pull_rate": float,
        "policy_pull_rate": float,
        "agreement_rate": float,
      }
    """
    model.eval()

    ds = loader.dataset

    greedy_actions = []
    valid_counts = np.zeros(ds.num_actions, dtype=np.int64)

    for batch in loader:
        s,a,r,ns,d,m,m_next = batch

        s=s.to(device)
        m=m.to(device)

        q=model(s)
        q_masked = masked_q(q,m)
        a_greedy = q_masked.argmax(dim=1)

        greedy_actions.append(a_greedy.cpu().numpy())

        valid_counts += m.sum(dim=0).cpu().numpy().astype(np.int64)
    greedy_actions = np.concatenate(greedy_actions, axis=0)
    behavior_actions = ds.actions.cpu().numpy()
    assert behavior_actions.shape == greedy_actions.shape

    num_actions = ds.num_actions
    num_samples = behavior_actions.shape[0]

    behavior_action_counts = np.bincount(
        behavior_actions, minlength=num_actions
    )
    policy_action_counts = np.bincount(
        greedy_actions, minlength=num_actions
    )

    behavior_pull_rate = float((behavior_actions>0).mean())
    policy_pull_rate = float((greedy_actions >0).mean())
    agreement_rate = float((behavior_actions == greedy_actions).mean())

    return {
        "num_actions": num_actions,
        "num_samples": num_samples,
        "behavior_action_counts": behavior_action_counts,
        "policy_action_counts": policy_action_counts,
        "valid_action_counts": valid_counts,
        "behavior_pull_rate": behavior_pull_rate,
        "policy_pull_rate": policy_pull_rate,
        "agreement_rate": agreement_rate,
    }

def summarize_policy_behavior_stats(stats: Dict[str, Any]) -> None:
    """
    Nicely print the dictionary returned by compute_policy_behavior_stats.
    """
    print("=== Policy vs Behavior Stats ===")
    print(f"Num samples:   {stats['num_samples']}")
    print(f"Num actions:   {stats['num_actions']}")
    print()
    print(f"Behavior pull rate: {stats['behavior_pull_rate']*100:.2f}%")
    print(f"Policy pull rate:   {stats['policy_pull_rate']*100:.2f}%")
    print(f"Action agreement:   {stats['agreement_rate']*100:.2f}%")
    print()
    print("Behavior action counts (per action index):")
    print(stats["behavior_action_counts"])
    print("Policy action counts (per action index):")
    print(stats["policy_action_counts"])
    print("Valid action counts (per action index):")
    print(stats["valid_action_counts"])

## Q value distributions
@torch.no_grad()
def compute_q_distributions(
    model: CQLNet,
    loader: DataLoader,
    device: str
) -> Dict[str, Any]:
    """
    Collect distributions of Q-values for analysis:

      - q_all_valid: Q(s,a) for all actions that are valid under mask
      - q_stay:      Q(s, stay-action=0)
      - q_best_pull: best Q(s,a) over pull-actions a>0, when at least one is valid
      - q_stay_minus_best_pull: Q(s,0) - max_{a>0} Q(s,a) for states with valid pulls
    """
    model.eval()

    q_all_valid_list =[]
    q_stay_list = []
    q_best_pull_list =[]
    q_stay_minus_best_pull_list =[]

    for batch in loader:
        s,a,r,ns,d,m,m_next = batch

        s =s.to(device)
        m = m.to(device)
        q=model(s)

        q_valid = q[m]
        q_all_valid_list.append(q_valid.detach().cpu().numpy())

        q_stay = q[:, 0]
        q_stay_list.append(q_stay.detach().cpu().numpy())

        if q.shape[1] > 1:
            pull_mask = m.clone()
            pull_mask[:,0] = False

            has_valid_pull = pull_mask.any(dim=1)
            if has_valid_pull.any():
                q_pull = q.clone()
                q_pull[~pull_mask] = -1e9
                q_best_pull = q_pull.max(dim=1).values

                q_best_pull_valid = q_best_pull[has_valid_pull]
                q_stay_valid = q_stay[has_valid_pull]

                q_best_pull_list.append(
                    q_best_pull_valid.detach().cpu().numpy()
                )
                q_stay_minus_best_pull_list.append(
                    (q_stay_valid - q_best_pull_valid).detach().cpu().numpy()
                )

    def _concat(xs):
        return np.concatenate(xs, axis=0) if xs else np.array([], dtype=np.float32)

    q_all_valid = _concat(q_all_valid_list)
    q_stay = _concat(q_stay_list)
    q_best_pull = _concat(q_best_pull_list)
    q_stay_minus_best_pull = _concat(q_stay_minus_best_pull_list)

    return {
        "q_all_valid": q_all_valid,
        "q_stay": q_stay,
        "q_best_pull": q_best_pull,
        "q_stay_minus_best_pull": q_stay_minus_best_pull,
    }

def summarize_q_distributions(q_stats: Dict[str, Any]) -> None:
    """
    Nicely print high-level stats of Q distributions.
    """
    import numpy as np

    def _summary(name: str, x: np.ndarray):
        if x.size == 0:
            print(f"{name}: (empty)")
            return
        print(
            f"{name}: n={x.size}, mean={x.mean():.3f}, std={x.std():.3f}, "
            f"min={x.min():.3f}, max={x.max():.3f}"
        )

    print("=== Q Distribution Stats ===")
    _summary("q_all_valid", q_stats["q_all_valid"])
    _summary("q_stay", q_stats["q_stay"])
    _summary("q_best_pull", q_stats["q_best_pull"])
    _summary("q_stay_minus_best_pull", q_stats["q_stay_minus_best_pull"])
