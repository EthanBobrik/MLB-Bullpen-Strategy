"""
src/rl/cql.py
Conservative Q-Learning (discrete action) implementation compatible with the project's style.
Reads config from configs/{model.yaml, training.yaml, data.yaml, env.yaml, inference.yaml}
Intended to follow the formatting / function names used in src/rl/dqn.py
"""

import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Try to reuse dataset from dqn (if available)
try:
    from src.rl.dqn import BullpenOfflineDataset, RLDatasetConfig
except Exception:
    # Fallback simple dataset if the dqn module isn't importable
    class BullpenOfflineDataset(Dataset):
        """
        Fallback dataset that expects NPZ with arrays:
         - states: [N, d_state]
         - actions: [N] int
         - rewards: [N]
         - next_states: [N, d_state]
         - dones: [N] (0/1)
         - avail_mask: [N, num_actions] optional
        """
        def __init__(self, npz_path: str, max_samples: Optional[int] = None):
            data = np.load(npz_path)
            # Standard naming convention fallback
            self.states = data["states"]
            self.actions = data["actions"]
            self.rewards = data["rewards"]
            self.next_states = data["next_states"]
            self.dones = data["dones"].astype(np.float32)
            self.avail_mask = data.get("avail_mask", None)
            n = len(self.states)
            if max_samples:
                idx = np.random.choice(n, min(n, max_samples), replace=False)
                self.states = self.states[idx]
                self.actions = self.actions[idx]
                self.rewards = self.rewards[idx]
                self.next_states = self.next_states[idx]
                self.dones = self.dones[idx]
                if self.avail_mask is not None:
                    self.avail_mask = self.avail_mask[idx]

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            out = {
                "state": self.states[idx].astype(np.float32),
                "action": int(self.actions[idx]),
                "reward": float(self.rewards[idx]),
                "next_state": self.next_states[idx].astype(np.float32),
                "done": float(self.dones[idx])
            }
            if self.avail_mask is not None:
                out["avail_mask"] = self.avail_mask[idx].astype(np.float32)
            return out


@dataclass
class CQLConfig:
    # general
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # model / architecture
    input_dim: Optional[int] = None
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.05
    num_actions: int = 11

    # training
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 512
    max_steps: int = 50000
    log_interval: int = 1000
    target_update_interval: int = 1000
    tau: float = 0.005

    # CQL specific
    cql_alpha: float = 1.0
    cql_min_q_weight: Optional[float] = None
    cql_temp: float = 1.0
    l2_reg: float = 1e-6

    # IO
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "cql_model.pth"
    use_wandb: bool = False


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_size: int = 256,
                 num_layers: int = 3, dropout: float = 0.05):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        # final head to action-values
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, num_actions]


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_config_from_yamls(
    model_yaml="configs/model.yaml",
    training_yaml="configs/training.yaml",
    env_yaml="configs/env.yaml",
    data_yaml="configs/data.yaml",
    inference_yaml="configs/inference.yaml",
) -> CQLConfig:
    # Load and merge
    model_cfg = load_yaml(model_yaml)
    training_cfg = load_yaml(training_yaml)
    env_cfg = load_yaml(env_yaml)
    data_cfg = load_yaml(data_yaml)
    inference_cfg = load_yaml(inference_yaml)

    # Initialize default config then override
    cfg = CQLConfig()
    # input dim: try to infer; otherwise left None
    # num_actions: prefer model.yaml num_actions
    if "num_actions" in model_cfg:
        cfg.num_actions = int(model_cfg["num_actions"])
    # architecture override
    if "cql" in model_cfg:
        c = model_cfg["cql"]
        cfg.hidden_size = int(c.get("hidden_size", cfg.hidden_size))
        cfg.num_layers = int(c.get("num_layers", cfg.num_layers))
        cfg.dropout = float(c.get("dropout", cfg.dropout))
        cfg.lr = float(c.get("lr", cfg.lr))
        cfg.gamma = float(c.get("gamma", cfg.gamma))
        cfg.batch_size = int(c.get("batch_size", cfg.batch_size)) if "batch_size" in c else cfg.batch_size

    # training.yaml overrides for CQL-specific params
    cfg.max_steps = int(training_cfg.get("epochs", cfg.max_steps)) if isinstance(training_cfg.get("epochs", None), int) else cfg.max_steps
    # training.yaml includes cql_alpha
    if "cql_alpha" in training_cfg:
        cfg.cql_alpha = float(training_cfg["cql_alpha"])
    if "checkpoint_dir" in training_cfg:
        cfg.checkpoint_dir = training_cfg["checkpoint_dir"]
    if "use_wandb" in training_cfg:
        cfg.use_wandb = bool(training_cfg["use_wandb"])
    # a few safe copies
    cfg.tau = float(training_cfg.get("tau_soft_update", cfg.tau))
    cfg.target_update_interval = int(training_cfg.get("target_update_interval", cfg.target_update_interval))
    cfg.log_interval = int(training_cfg.get("log_interval", cfg.log_interval))
    # environment hints
    cfg.gamma = float(env_cfg.get("gamma", cfg.gamma))

    return cfg


def cql_loss(q_values: torch.Tensor,
             actions: torch.LongTensor,
             q_target_values: torch.Tensor,
             rewards: torch.Tensor,
             dones: torch.Tensor,
             gamma: float,
             cql_alpha: float,
             cql_temp: float = 1.0,
             l2_reg: float = 0.0,
             avail_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """
    Compute TD loss + CQL conservative regularization.

    Args:
        q_values: [B, A] current Q network outputs for states
        actions: [B] long tensor of taken actions (index)
        q_target_values: [B, A] target Q network outputs for next_states
        rewards: [B]
        dones: [B] (0/1)
        avail_mask: [B, A] (0/1) optional - denote available actions
    Returns:
        loss_scalar, diagnostics
    """
    device = q_values.device
    B, A = q_values.shape

    # gather q for actions
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

    # TD target: reward + gamma * (1 - done) * max_a' Q_target(next, a')
    if avail_mask is not None:
        # mask invalid actions by setting them to -inf before max
        q_target_masked = q_target_values.clone()
        inf_mask = (avail_mask <= 0)
        q_target_masked[inf_mask.bool()] = -1e9
        q_next_max, _ = q_target_masked.max(dim=1)
    else:
        q_next_max, _ = q_target_values.max(dim=1)

    td_target = rewards + gamma * (1.0 - dones) * q_next_max
    td_loss = F.mse_loss(q_taken, td_target)

    # Conservative regularizer: logsumexp(Q/temperature) - Q(s,a_taken)
    if cql_temp != 1.0:
        scaled_q = q_values / cql_temp
    else:
        scaled_q = q_values
    if avail_mask is not None:
        # mask before logsumexp by setting unavailable to -inf
        scaled_q_masked = scaled_q.clone()
        scaled_q_masked[(avail_mask <= 0).bool()] = -1e9
        logsumexp_q = torch.logsumexp(scaled_q_masked, dim=1)  # [B]
    else:
        logsumexp_q = torch.logsumexp(scaled_q, dim=1)  # [B]

    cql_reg = (logsumexp_q - q_taken).mean()
    loss = td_loss + cql_alpha * cql_reg

    # optional l2 reg
    if l2_reg and l2_reg > 0:
        l2_loss = 0.0
        for p in torch.nn.utils.parameters_to_vector(q_values):  # this is wrong for tensors; skip
            pass
        # skip l2 over raw q tensor; user should set weight_decay in optimizer
    diagnostics = {
        "td_loss": float(td_loss.detach().cpu().item()),
        "cql_reg": float(cql_reg.detach().cpu().item()),
        "cql_alpha": float(cql_alpha)
    }
    return loss, diagnostics


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau)
        tp.data.add_(sp.data * tau)


def save_checkpoint(state: Dict, directory: str, name: str):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    return torch.load(path, map_location=device)


def train_cql(
    cfg: CQLConfig,
    dataset: Optional[Dataset] = None,
    dataset_npz: Optional[str] = None,
    max_steps: Optional[int] = None,
):
    """
    Train a discrete-action CQL agent.
    Either provide a Dataset instance or an npz path to load the fallback BullpenOfflineDataset.
    """
    # seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device

    # load dataset
    if dataset is None:
        assert dataset_npz is not None, "Provide dataset or dataset_npz path"
        dataset = BullpenOfflineDataset(dataset_npz)
    # try to infer input dim from first sample
    sample = dataset[0]
    state = sample["state"]
    input_dim = state.shape[-1] if isinstance(state, (np.ndarray,)) else state.shape[-1]
    cfg.input_dim = int(input_dim)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # networks
    q_net = MLPQNetwork(cfg.input_dim, cfg.num_actions, cfg.hidden_size, cfg.num_layers, cfg.dropout).to(device)
    target_q_net = MLPQNetwork(cfg.input_dim, cfg.num_actions, cfg.hidden_size, cfg.num_layers, cfg.dropout).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr, weight_decay=cfg.l2_reg)

    step = 0
    max_steps = max_steps or cfg.max_steps
    start_time = time.time()
    losses = []
    diagnostics_hist = []

    while step < max_steps:
        for batch in dataloader:
            if step >= max_steps:
                break
            states = torch.tensor(batch["state"], dtype=torch.float32, device=device)
            actions = torch.tensor(batch["action"], dtype=torch.long, device=device)
            rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
            next_states = torch.tensor(batch["next_state"], dtype=torch.float32, device=device)
            dones = torch.tensor(batch["done"], dtype=torch.float32, device=device)
            avail_mask = None
            if "avail_mask" in batch:
                avail_mask = torch.tensor(batch["avail_mask"], dtype=torch.float32, device=device)

            q_values = q_net(states)  # [B, A]
            with torch.no_grad():
                q_target_next = target_q_net(next_states)  # [B, A]

            loss, diag = cql_loss(
                q_values=q_values,
                actions=actions,
                q_target_values=q_target_next,
                rewards=rewards,
                dones=dones,
                gamma=cfg.gamma,
                cql_alpha=cfg.cql_alpha,
                cql_temp=cfg.cql_temp,
                l2_reg=cfg.l2_reg,
                avail_mask=avail_mask
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

            # soft target update
            soft_update(target_q_net, q_net, cfg.tau)

            if step % cfg.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"[CQL] step={step}/{max_steps} loss={loss.item():.6f} td={diag['td_loss']:.6f} cql_reg={diag['cql_reg']:.6f} elapsed={elapsed:.1f}s")
            if step % cfg.target_update_interval == 0 and step > 0:
                # hard save copy
                target_q_net.load_state_dict(q_net.state_dict())

            # checkpointing
            if cfg.checkpoint_dir and (step % max(1, cfg.log_interval * 10) == 0):
                state = {
                    "q_state_dict": q_net.state_dict(),
                    "target_q_state_dict": target_q_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": asdict(cfg),
                    "step": step
                }
                save_checkpoint(state, cfg.checkpoint_dir, cfg.checkpoint_name)

            losses.append(float(loss.item()))
            diagnostics_hist.append(diag)
            step += 1

    # final save
    state = {
        "q_state_dict": q_net.state_dict(),
        "target_q_state_dict": target_q_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": asdict(cfg),
        "step": step
    }
    save_checkpoint(state, cfg.checkpoint_dir, cfg.checkpoint_name)
    print("[CQL] Training complete. Saved checkpoint to", os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
    return q_net, target_q_net, losses, diagnostics_hist


def greedy_policy_action(q_net: nn.Module, state: np.ndarray, device: str = "cpu", avail_mask: Optional[np.ndarray] = None) -> int:
    """Return greedy action index for a single state numpy vector."""
    q_net.eval()
    with torch.no_grad():
        s = torch.tensor(state.astype(np.float32), device=device).unsqueeze(0)
        q = q_net(s).squeeze(0).cpu().numpy()  # [A]
        if avail_mask is not None:
            q = np.where(avail_mask > 0, q, -1e9)
        return int(np.argmax(q))


# If run as script
if __name__ == "__main__":
    # basic CLI usage: python -m src.rl.cql (will attempt to read configs)
    cfg = build_config_from_yamls()
    # set dataset path inferred from data.yaml if available
    data_cfg = load_yaml("configs/data.yaml")
    dataset_npz = None
    if "processed_data_dir" in data_cfg and "dataset_file" in data_cfg:
        dataset_npz = os.path.join(data_cfg["processed_data_dir"], data_cfg["dataset_file"])
    if dataset_npz is None or not os.path.exists(dataset_npz):
        raise FileNotFoundError(f"Dataset npz not found at {dataset_npz}. Provide dataset_npz argument.")

    train_cql(cfg, dataset_npz=dataset_npz)
