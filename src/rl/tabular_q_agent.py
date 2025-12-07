from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
from collections import defaultdict


# 1. CONFIG STRUCTURE

@dataclass
class TabularQAgentConfig:
    data_path: Path
    device: str

    alpha: float
    gamma: float
    num_epochs: int
    val_fraction: float
    precision: int  # kept for compatibility, not heavily used now

    log_interval: int

    yaml_num_actions: Optional[int] = None


# 2. LOAD CONFIG FROM YAML

def load_tabular_q_config(
    model_config_path: Path,
    data_path: Path,
    device: Optional[str] = None
) -> TabularQAgentConfig:

    with open(model_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tq_cfg = cfg.get("tabular_q", {}) or {}

    def require(key):
        if key not in tq_cfg or tq_cfg[key] is None:
            raise ValueError(f"Missing required Tabular-Q config key: '{key}'")
        return tq_cfg[key]

    alpha = float(require("alpha"))
    gamma = float(require("gamma"))
    num_epochs = int(require("num_epochs"))
    precision = int(require("precision"))
    log_interval = int(require("log_interval"))
    val_fraction = float(tq_cfg.get("val_fraction", 0.1))

    yaml_num_actions = cfg.get("num_actions", None)
    if yaml_num_actions is not None:
        yaml_num_actions = int(yaml_num_actions)

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


# 3. OFFLINE DATASET

class TabularOfflineDataset(Dataset):
    """
    Minimal offline dataset for Tabular Q-learning.
    Uses:
      - state_vec
      - next_state_vec
      - reward_folded
      - action_idx
      - done
      - avail_mask [B, R] expanded to [B, 1+R]
    """

    def __init__(self, data_path: Path, device: str):
        super().__init__()

        data = np.load(data_path)

        self.state_vec = torch.tensor(data["state_vec"], dtype=torch.float32).to(device)
        self.next_state_vec = torch.tensor(data["next_state_vec"], dtype=torch.float32).to(device)

        self.actions = torch.tensor(data["action_idx"], dtype=torch.long).to(device)
        self.rewards = torch.tensor(data["reward_folded"], dtype=torch.float32).to(device)
        self.dones = torch.tensor(data["done"], dtype=torch.float32).to(device)

        mask_rel = torch.tensor(data["avail_mask"], dtype=torch.bool).to(device)
        B, R = mask_rel.shape

        full_mask = torch.ones((B, 1 + R), dtype=torch.bool, device=device)
        full_mask[:, 1:] = mask_rel
        self.avail_mask = full_mask

        self.num_actions = 1 + R
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


# 4. COMPRESSED STATE ENCODER

def encode_state(raw_state: np.ndarray) -> Tuple[int, ...]:
    """
    Compress the 13-D continuous state_vec into a small, discrete key.

    Expected state layout (from your dataset build):
      0: inning
      1: half
      2: outs
      3: base_state
      4: score_diff
      5: pitch_count
      6: tto
      7: is_platoon_advantage
      8: batter_is_left
      9: batter_is_right
      10: batter_is_switch
      11: pitcher_is_left
      12: pitcher_is_right
    """

    s = np.asarray(raw_state, dtype=np.float32)

    inning = int(s[0])           # already small int
    half = int(s[1])             # 0 = top, 1 = bottom
    outs = int(s[2])             # 0,1,2
    base_state = int(s[3])       # 0..7

    # Bucket score diff: clip to [-4, 4]
    score_diff = float(s[4])
    score_bucket = int(np.clip(score_diff, -4, 4))

    # Bucket pitch count: 0-25, 26-50, 51-75, 76+
    pitch_count = float(s[5])
    if pitch_count <= 25:
        pitch_bucket = 0
    elif pitch_count <= 50:
        pitch_bucket = 1
    elif pitch_count <= 75:
        pitch_bucket = 2
    else:
        pitch_bucket = 3

    # Bucket TTO: 0,1,2,3+
    tto = int(s[6])
    tto_bucket = min(tto, 3)

    platoon = int(round(float(s[7])))  # 0 or 1

    # Batter side: argmax over [L,R,S]
    batter_vec = s[8:11]
    batter_side = int(np.argmax(batter_vec))  # 0=L,1=R,2=S

    # Pitcher side: argmax over [L,R]
    pitcher_vec = s[11:13]
    pitcher_side = int(np.argmax(pitcher_vec))  # 0=L,1=R

    # Final discrete key (all ints)
    return (
        inning,
        half,
        outs,
        base_state,
        score_bucket,
        pitch_bucket,
        tto_bucket,
        platoon,
        batter_side,
        pitcher_side,
    )


def discretize_state(state: np.ndarray, precision: int) -> Tuple:
    """
    Kept for backward compatibility; now delegates to encode_state.
    'precision' is no longer critical but preserved in signature.
    """
    return encode_state(state)


# 5. TABULAR Q AGENT

class TabularQAgent:
    def __init__(self, num_actions: int, gamma: float, alpha: float, precision: int):
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.precision = precision  # not heavily used now, but kept

        self.Q: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32)
        )

    def _s(self, state: np.ndarray) -> Tuple:
        return discretize_state(state, self.precision)

    def update(self, s, a, r, ns, done):
        s_key = self._s(s)
        ns_key = self._s(ns)

        q_sa = self.Q[s_key][a]
        target = r if done else r + self.gamma * np.max(self.Q[ns_key])
        self.Q[s_key][a] = q_sa + self.alpha * (target - q_sa)

    def act(self, state: np.ndarray, avail_mask: Optional[np.ndarray] = None) -> int:
        s_key = self._s(state)
        q = self.Q[s_key].copy()

        if avail_mask is not None:
            q[~avail_mask] = -1e9

        return int(np.argmax(q))

    def value(self, state: np.ndarray) -> float:
        s_key = self._s(state)
        return float(np.max(self.Q[s_key]))

    # FAST SAVE / LOAD

    def save(self, path: Path):
        states = np.array([np.array(s, dtype=np.float32) for s in self.Q.keys()])
        q_vals = np.array([v for v in self.Q.values()], dtype=np.float32)
        np.savez_compressed(path, states=states, q_vals=q_vals)

    def load(self, path: Path):
        data = np.load(path, allow_pickle=False)
        states = data["states"]
        q_vals = data["q_vals"]

        self.Q = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

        for s, q in zip(states, q_vals):
            self.Q[tuple(s.tolist())] = q


# 6. TRAINING LOOP WITH EARLY STOPPING

def train_tabular_q_agent(cfg: TabularQAgentConfig):
    ds = TabularOfflineDataset(cfg.data_path, cfg.device)

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

    best_val = float("inf")
    patience = 0
    max_patience = 5

    for epoch in range(cfg.num_epochs):

        # Training
        for batch in train_loader:
            states, actions, rewards, next_states, dones, masks = batch

            for i in range(len(states)):
                agent.update(
                    s=states[i],
                    a=int(actions[i]),
                    r=float(rewards[i]),
                    ns=next_states[i],
                    done=bool(dones[i]),
                )

        # Validation TD error
        val_loss = 0.0
        n = 0

        for batch in val_loader:
            states, actions, rewards, next_states, dones, masks = batch

            for i in range(len(states)):
                s_key = agent._s(states[i])
                ns_key = agent._s(next_states[i])

                q_sa = agent.Q[s_key][int(actions[i])]
                target = (
                    rewards[i]
                    if dones[i]
                    else rewards[i] + cfg.gamma * np.max(agent.Q[ns_key])
                )

                val_loss += (q_sa - target) ** 2
                n += 1

        val_loss /= max(n, 1)

        if epoch % cfg.log_interval == 0:
            print(f"[Tabular-Q] epoch={epoch}/{cfg.num_epochs}  val_td_error={val_loss:.6f}")

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f"[Early Stopping] No improvement for {max_patience} epochs. Stopping.")
            break

    return agent


# 7. PULL RATE CALCULATION

def compute_pull_rate(agent: TabularQAgent, dataset: TabularOfflineDataset) -> float:
    pull_count = 0
    total = 0

    for i in range(len(dataset)):
        s, _, _, _, _, mask = dataset[i]
        a = agent.act(s, mask)

        if a > 0:
            pull_count += 1

        total += 1

    return pull_count / max(total, 1)

