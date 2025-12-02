import torch
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from src.rl.cql import BullpenOfflineDataset, MLPQNetwork

# ===============================
# Config
# ===============================
@dataclass
class OfflineEvalConfig:
    device: str = "cpu"
    batch_size: int = 1024
    num_workers: int = 2

# ===============================
# Load model + dataset
# ===============================
def load_model_and_dataset(model_path: str, dataset_path: str, cfg: OfflineEvalConfig):
    checkpoint = torch.load(model_path, map_location=cfg.device)
    model_cfg = checkpoint["config"]

    q_net = MLPQNetwork(
        input_dim=model_cfg.observation_size,
        num_actions=model_cfg.action_size,
        hidden_size=model_cfg.hidden_sizes,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
    ).to(cfg.device)

    q_net.load_state_dict(checkpoint["q_net_state_dict"])
    q_net.eval()

    dataset = BullpenOfflineDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return q_net, loader

# ===============================
# TD Error Full MSE (dataset Bellman residual)
# ===============================
def evaluate_td_error_full_mse(q_net: MLPQNetwork, loader: DataLoader, discount: float = 0.99):
    mse_vals = []

    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].float()
            act = batch["actions"].long().squeeze(-1)
            rew = batch["rewards"].float().squeeze(-1)
            next_obs = batch["next_obs"].float()
            done = batch["dones"].float().squeeze(-1)

            q_pred = q_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            q_next = q_net(next_obs).max(dim=1)[0]

            target = rew + discount * (1 - done) * q_next
            mse = (q_pred - target).pow(2).mean().item()
            mse_vals.append(mse)

    return float(np.mean(mse_vals))

# ===============================
# Direct Policy Value Estimate (average greedy Q)
# ===============================
def direct_policy_value_estimate(q_net: MLPQNetwork, loader: DataLoader):
    greedy_q_vals = []

    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].float()
            q_vals = q_net(obs)
            greedy_q = q_vals.max(dim=1)[0]
            greedy_q_vals.append(greedy_q.mean().item())

    return float(np.mean(greedy_q_vals))

# ===============================
# Action Agreement: argmax(Q) == behavior action
# ===============================
def compute_action_agreement(q_net: MLPQNetwork, loader: DataLoader):
    matches = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].float()
            beh_act = batch["actions"].long().squeeze(-1)

            greedy = q_net(obs).argmax(dim=1)
            matches += (greedy == beh_act).sum().item()
            total += beh_act.numel()

    return matches / total if total > 0 else 0.0

# ===============================
# Summary Wrapper
# ===============================
def run_full_cql_eval(model_path: str, dataset_path: str, cfg: OfflineEvalConfig):
    q_net, loader = load_model_and_dataset(model_path, dataset_path, cfg)

    td_mse = evaluate_td_error_full_mse(q_net, loader)
    dpv = direct_policy_value_estimate(q_net, loader)
    agree = compute_action_agreement(q_net, loader)

    return {
        "td_mse": td_mse,
        "direct_value_estimate": dpv,
        "action_agreement": agree,
    }
