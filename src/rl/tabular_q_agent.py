from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
from collections import defaultdict


# 1. CONFIG STRUCTURE (matches your DQN style)

@dataclass
class TabularQAgentConfig:
    data_path: Path        # path to rl_tensors npz
    device: str            # cpu/cuda (not used heavily here but kept for symmetry)

    # training hyperparameters
    alpha: float           # Q-learning learning rate
    gamma: float           # discount factor
    num_epochs: int        # how many epochs over dataset
    val_fraction: float    # same as DQN val split
    precision: int         # rounding precision for discretized states

    # logging
    log_interval: int

    # sanity check (optional)
    yaml_num_actions: Optional[int] = None


# 2. LOAD CONFIG FROM YAML (same pattern as load_dqn_training_config)

def load_tabular_q_config(
    model_config_path: Path,
    data_path: Path,
    device: Optional[str] = None
) -> TabularQAgentConfig:

    with open(model_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tq_cfg = cfg.get("tabular_q", {}) or {}

    alpha = float(tq_cfg.get("alpha"))
    gamma = float(tq_cfg.get("gamma"))
    num_epochs = int(tq_cfg.get("num_epochs"))
    precision = int(tq_cfg.get("precision"))
    log_interval = int(tq_cfg.get("log_interval"))
    val_fraction = float(tq_cfg.get("val_fraction", 0.1))

    yaml_num_actions = cfg.get("num_actions", None)
    if yaml_num_actions is not None:
        yaml_num_actions = int(yaml_num_actions)

    # device selection identical to DQN
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return TabularQAgentConfig(
        data_path=data_path,
        device=device,
        alpha=alpha,
        gamma=gamma,
        num_epochs=num_epochs,
        val_fraction=val_fraction,
        precision=precision,
        log_interval=log_interval,
        yaml_num_actions=yaml_num_actions,
    )


# 3. OFFLINE DATASET FOR TABULAR Q (same API as DQN dataset)

class TabularOfflineDataset(Dataset):
    """
    Minimal offline dataset for Tabular Q-learning.

    Uses:
      - state_vec
      - next_state_vec
      - reward_folded
      - action_idx
      - done
      - avail_mask [B, R]  -> expanded to full mask [B, 1+R]
    """

    def __init__(self, data_path: Path, device: str):
        super().__init__()

        data = np.load(data_path)

        # Convert to tensors (though tabular Q uses CPU numpy, we mimic interface)
        self.state_vec = torch.tensor(data["state_vec"], dtype=torch.float32).to(device)
        self.next_state_vec = torch.tensor(data["next_state_vec"], dtype=torch.float32).to(device)

        self.actions = torch.tensor(data["action_idx"], dtype=torch.long).to(device)
        self.rewards = torch.tensor(data["reward_folded"], dtype=torch.float32).to(device)
        self.dones = torch.tensor(data["done"], dtype=torch.float32).to(device)

        # Original avail_mask has shape [B, R]
        mask_rel = torch.tensor(data["avail_mask"], dtype=torch.bool).to(device)
        B, R = mask_rel.shape

        # Build full mask: 0=stay, 1..R = relievers
        full_mask = torch.ones((B, 1 + R), dtype=torch.bool, device=device)
        full_mask[:, 1:] = mask_rel
        self.avail_mask = full_mask

        self.num_actions = full_mask.shape[1]
        self.num_samples = B

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.state_vec[idx].cpu().numpy(),
            int(self.actions[idx].item()),
            float(self.rewards[idx].item()),
            self.next_state_vec[idx].cpu().numpy(),
            bool(self.dones[idx].item()),
            self.avail_mask[idx].cpu().numpy(),
        )


# 4. TABULAR Q-LEARNING AGENT

def discretize_state(state: np.ndarray, precision: int) -> Tuple:
    return tuple(np.round(state, precision))


class TabularQAgent:
    def __init__(self, num_actions: int, gamma: float, alpha: float, precision: int):
        """
        Simple tabular Q-learning with discretized states.
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.precision = precision

        # Q-table stored as: dict[state_tuple] -> np.array(num_actions)
        self.Q: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32)
        )

    def _s(self, state: np.ndarray) -> Tuple:
        return discretize_state(state, self.precision)

    def update(self, s, a, r, ns, done):
        """
        Standard tabular Q-learning TD update.
        """
        s = self._s(s)
        ns = self._s(ns)

        q_sa = self.Q[s][a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[ns])

        self.Q[s][a] = q_sa + self.alpha * (target - q_sa)

    def act(self, state: np.ndarray, avail_mask: Optional[np.ndarray] = None) -> int:
        """
        Greedy action with optional availability mask.
        """
        s = self._s(state)
        q = self.Q[s].copy()

        if avail_mask is not None:
            q[~avail_mask] = -1e9  # invalidate illegal actions

        return int(np.argmax(q))

    def value(self, state: np.ndarray) -> float:
        return float(np.max(self.Q[self._s(state)]))

    def save(self, path: Path):
        np.save(path, dict(self.Q))

    def load(self, path: Path):
        raw = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32),
            raw,
        )



# 5. TRAINING LOOP (mirrors train_dqn)

def train_tabular_q_agent(cfg: TabularQAgentConfig):
    """
    Offline Tabular Q-learning:
        - loads dataset
        - splits train/val
        - trains for num_epochs
        - logs validation TD error

    Returns:
        trained TabularQAgent
    """

    ds = TabularOfflineDataset(cfg.data_path, cfg.device)

    # Train/val split (same as DQN)
    train_size = int((1 - cfg.val_fraction) * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

    agent = TabularQAgent(
        num_actions=ds.num_actions,
        gamma=cfg.gamma,
        alpha=cfg.alpha,
        precision=cfg.precision,
    )

    # --- Training ---
    for epoch in range(cfg.num_epochs):
        for batch in train_loader:
            # Unpack mini-batch
            states, actions, rewards, next_states, dones, masks = batch

            # Loop elementwise (simple tabular updates)
            for i in range(len(states)):
                agent.update(
                    s=states[i],
                    a=int(actions[i]),
                    r=float(rewards[i]),
                    ns=next_states[i],
                    done=bool(dones[i]),
                )

        if epoch % cfg.log_interval == 0:
            # Compute validation TD error
            val_loss = 0.0
            n = 0
            for batch in val_loader:
                states, actions, rewards, next_states, dones, masks = batch
                for i in range(len(states)):
                    s = agent._s(states[i])
                    ns = agent._s(next_states[i])

                    q_sa = agent.Q[s][int(actions[i])]
                    target = (
                        rewards[i]
                        if dones[i]
                        else rewards[i] + cfg.gamma * np.max(agent.Q[ns])
                    )
                    val_loss += (q_sa - target) ** 2
                    n += 1
            val_loss /= max(n, 1)

            print(
                f"[Tabular-Q] epoch={epoch}/{cfg.num_epochs}  val_td_error={val_loss:.6f}"
            )

    return agent
