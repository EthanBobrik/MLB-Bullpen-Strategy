from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import yaml

@dataclass
class DQNTrainingConfig:
    # paths/ runtime
    data_path: Path
    device: str

    # training
    batch_size: int
    lr: float
    gamma: float
    max_steps: int
    target_update_interval: int
    log_interval: int
    val_fraction: float

    #architecture
    hidden_size: int
    num_layers: int
    dropout: float

    # sanity checks
    yaml_num_actions: Optional[int] = None

def load_dqn_training_config(
    model_config_path: Path,
    data_path: Path,
    device: Optional[str] = None
) -> DQNTrainingConfig:
    """
    Load DQN hyperparameters from configs/model.yaml and
    return a DQNTrainingConfig instance.

    - model_config_path: typically Path("configs/model.yaml")
    - data_path:         offline RL npz (e.g. rl_tensors_2022_2023.npz)
    - device:            "cuda" / "cpu" (if None, auto-detect)
    """
    with open(model_config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    dqn_cfg = cfg.get('dqn', {}) or {}

    hidden_size = int(dqn_cfg.get('hidden_size'))
    num_layers = int(dqn_cfg.get('num_layers'))
    dropout = float(dqn_cfg.get('dropout'))

    gamma = float(dqn_cfg.get('gamma'))
    lr = float(dqn_cfg.get('lr'))
    batch_size = int(dqn_cfg.get('batch_size'))
    max_steps = int(dqn_cfg.get('max_steps'))
    target_update_interval = int(dqn_cfg.get('target_update_interval'))
    log_interval = int(dqn_cfg.get('log_interval'))
    val_fraction = float(dqn_cfg.get('val_fraction'))

    yaml_num_actions = cfg.get('num_actions',None)
    if yaml_num_actions is not None:
        yaml_num_actions = int(yaml_num_actions)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return DQNTrainingConfig(
        data_path=data_path,
        device=device,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        max_steps=max_steps,
        target_update_interval=target_update_interval,
        log_interval=log_interval,
        val_fraction=val_fraction,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        yaml_num_actions=yaml_num_actions
    )

@dataclass
class RLDatasetConfig:
    data_path : Path
    device: str

class BullpenOfflineDataset(Dataset):
    """
    Dataset for your bullpen RL tensors.

    Required keys in the npz (all present in your outfile):
      state_vec
      next_state_vec
      next_hitters_feats     [B, H, D_h]
      pos_enc                [B, H, 5]
      reliever_feats         [B, R, D_r]
      avail_mask             [B, num_actions]
      action_idx             [B]
      reward_folded          [B]
      done                   [B]
    """
    def __init__(self, cfg: RLDatasetConfig):
        self.cfg = cfg
        data = np.load(cfg.data_path)

        self.state_vec = torch.tensor(data['state_vec'], dtype=torch.float32)
        self.next_state_vec = torch.tensor(data["next_state_vec"], dtype=torch.float32)

        self.next_hitters_feats = torch.tensor(data["next_hitters_feats"], dtype=torch.float32)
        self.pos_enc = torch.tensor(data["pos_enc"], dtype=torch.float32)
        self.reliever_feats = torch.tensor(data["reliever_feats"], dtype=torch.float32)

        self.avail_mask = torch.tensor(data["avail_mask"], dtype=torch.bool)

        self.actions = torch.tensor(data["action_idx"], dtype=torch.long)
        self.rewards = torch.tensor(data["reward_folded"], dtype=torch.float32)
        self.dones = torch.tensor(data["done"], dtype=torch.float32)

        for name in ["state_vec", "next_state_vec", "next_hitters_feats",
            "pos_enc", "reliever_feats", "avail_mask",
            "actions", "rewards", "dones"]:
            setattr(self, name, getattr(self,name).to(cfg.device))

        B = self.state_vec.shape[0]
        self.H = self.next_hitters_feats.shape[1]
        self.D_h = self.next_hitters_feats.shape[2]
        self.R = self.reliever_feats.shape[1]
        self.D_r = self.reliever_feats.shape[2]

        self.state_dim = (
            self.state_vec.shape[1] + self.H * self.D_h + self.H * 5 + self.R * self.D_r
        )
        self.num_actions = self.avail_mask.shape[1]
        self.num_samples = B

    def _build_state(self, idx):
        s0 = self.state_vec[idx]
        s1 = self.next_hitters_feats[idx].reshape(-1)
        s2 = self.pos_enc[idx].reshape(-1)
        s3 = self.reliever_feats[idx].reshape(-1)
        return torch.cat([s0,s1,s2,s3],dim=0)
    
    def _build_next_state(self, idx):
        ns0 = self.next_state_vec[idx]  # SAME SHAPE as state_vec
        ns1 = self.next_hitters_feats[idx].reshape(-1)
        ns2 = self.pos_enc[idx].reshape(-1)
        ns3 = self.reliever_feats[idx].reshape(-1)
        return torch.cat([ns0, ns1, ns2, ns3], dim=0)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        s = self._build_state(idx)
        ns = self._build_next_state(idx)
        return (
            s,
            self.actions[idx],
            self.rewards[idx],
            ns,
            self.dones[idx],
            self.avail_mask[idx],
            self.avail_mask[idx], # next-mask = same mask (SMDP)
        )
    
class DQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size, num_layers, dropout=0.0):
        super().__init__()

        layers =[]
        dim = state_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout >0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_size

        self.backbone = nn.Sequential(*layers)

        #Dueling heads
        self.value_head = nn.Linear(dim, 1) #V(s)
        self.adv_head = nn.Linear(dim, num_actions) # A(s,a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        value = self.value_head(z) # [B,1]
        adv = self.adv_head(z) # [B,A]
        adv_mean = adv.mean(dim=1,keepdim=True) 
        q = value + adv - adv_mean #[B,A]
        return q
    
def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    q:    [B, num_actions]
    mask: [B, num_actions] bool

    Returns Q with invalid actions set to a large negative constant so that
    argmax never picks them.
    """
    q2 = q.clone()
    q2[~mask] = -1e9
    return q2

@torch.no_grad()
def evaluate_td_error(model: DQN, target: DQN, val_loader:DataLoader, gamma:float, device:str) -> float:
    """
    Compute average TD-error on validation set:

        (Q(s,a) - [r + gamma * (1-done)*max_a' Q_target(s',a')])^2

    Using MSE loss (same as training).
    """
    model.eval()
    target.eval()
    criterion = nn.MSELoss(reduction='sum')

    total_loss = 0.0
    total_n = 0

    for batch in val_loader:
        s, a, r, ns, d, m, m_next = batch

        s = s.to(device)
        ns = ns.to(device)
        a = a.to(device)
        r = r.to(device)
        d = d.to(device)
        m_next = m_next.to(device)

        # Q(s,a)
        q = model(s)
        q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)

        # Target  = r+ gamma * (1-done) * max_a' Q_target(s',a')
        q_next = target(ns)
        q_next_masked = masked_q(q_next, m_next)
        q_next_max = q_next_masked.max(1).values

        tgt = r + gamma *(1.0-d) *q_next_max
        loss = criterion(q_sa, tgt)

        total_loss += loss.item()
        total_n += s.size(0)

    model.train()
    return total_loss / max(total_n,1)
    

def train_dqn(cfg: DQNTrainingConfig):
    """
    Offline DQN training:

    - Loads BullpenOfflineDataset from cfg.data_path
    - Splits into train / val
    - Trains a dueling DQN with a target network and MSE loss
    - Logs training loss and validation TD-error
    """
    ds = BullpenOfflineDataset(RLDatasetConfig(cfg.data_path, cfg.device))
    train_size = int((1-cfg.val_fraction) * len(ds))
    val_size = len(ds) - train_size
    train_ds,val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, cfg.batch_size,shuffle=True)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False)

    model = DQN(
        state_dim=ds.state_dim,
        num_actions=ds.num_actions,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    ).to(cfg.device)

    target = DQN(
        state_dim=ds.state_dim,
        num_actions=ds.num_actions,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    ).to(cfg.device)
    target.load_state_dict(model.state_dict())
    target.eval()

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(),lr =cfg.lr)

    step = 0
    while step < cfg.max_steps:
        for batch in train_loader:
            s, a, r, ns, d, m, m_next = batch

            s = s.to(cfg.device)
            ns = ns.to(cfg.device)
            a = a.to(cfg.device)
            r = r.to(cfg.device)
            d = d.to(cfg.device)
            m = m.to(cfg.device)
            m_next = m_next.to(cfg.device)

            q = model(s)
            q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target(ns)
                q_next_mask = masked_q(q_next, m_next)
                q_next_max = q_next_mask.max(1).values
                tgt = r + cfg.gamma * (1-d) * q_next_max
            
            loss = criterion(q_sa, tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % cfg.target_update_interval == 0:
                target.load_state_dict(model.state_dict())
            
            if step % cfg.log_interval == 0:
                print(f"[DQN] step={step} loss={loss.item():.5f}")
                val_td = evaluate_td_error(
                    model, target, val_loader, cfg.gamma, cfg.device
                )
                print(f"      val_td_error={val_td:.5f}")

            step+=1
            if step >= cfg.max_steps:
                break

    return model
