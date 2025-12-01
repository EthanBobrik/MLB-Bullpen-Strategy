from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.rl.dqn import (
    load_dqn_training_config,
    BullpenOfflineDataset,
    RLDatasetConfig,
    DQN,
    masked_q,
    evaluate_td_error,
)

@dataclass
class OfflineEvalConfig:
    """
    Configuration for offline policy evaluation of the DQN model.
    """
    model_config_path: Path
    model_path: Path
    tensors_path: Path
    device: str = 'cpu'
    batch_size: int =1024
    gamma: float =0.99

def load_model_and_dataset(cfg: OfflineEvalConfig) -> Tuple[DQN, BullpenOfflineDataset, DataLoader]:
    """
    Load:
      - training config from YAML (for architecture + hyperparams)
      - BullpenOfflineDataset from rl_tensors_*.npz
      - DQN model with saved weights
      - DataLoader over the full dataset (no train/val split)
    """
    train_cfg = load_dqn_training_config(
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

    model = DQN(
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
    model: DQN,
    loader: DataLoader,
    gamma: float,
    device: str,
) -> float:
    """
    Mean Squared TD Error (MSTE) over the full dataset, using the
    existing evaluate_td_error function from dqn.py.

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
    model:DQN,
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


## Behaviour policy agreement (how often DQN matches logged action)
@torch.no_grad()
def compute_action_agreement(
    model:DQN,
    loader:DataLoader,
    device:str
) -> float:
    """
    Compute how often the model's greedy masked action
    matches the logged action in the dataset.

    Returns:
        agreement_rate in [0,1]
    """
    model.eval()

    total= 0
    correct=0

    for batch in loader:
        s,a,r,ns,d,m,m_next = batch

        s = s.to(device)
        a = a.to(device)
        m = m.to(device)

        q = model(s)
        q_masked = masked_q(q, m)
        a_greedy = q_masked.argmax(dim=1)

        a_valid = m.gather(1, a.unsqueeze(1)).squeeze(1)
        mask_valid =a_valid.bool()

        if mask_valid.sum().item() == 0:
            continue

        a_logged_valid = a[mask_valid]
        a_greedy_valid = a_greedy[mask_valid]

        correct += (a_logged_valid == a_greedy_valid).sum().item()
        total += a_logged_valid.numel()

    if total == 0:
        return 0.0
    
    return correct/ total






