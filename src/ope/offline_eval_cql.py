"""
src/ope/offline_eval_cql.py
Offline evaluation utilities for the discrete-action CQL model.
Mirrors the formatting / function names of src/ope/offline_eval.py but adapted for loading
CQL checkpoints and evaluating the greedy policy.
"""

import os
import yaml
import numpy as np
import torch
from typing import Dict, Tuple

# model import
from src.rl.cql import MLPQNetwork, build_config_from_yamls, load_yaml

def load_checkpoint_model(checkpoint_path: str, cfg):
    device = cfg.device
    ckpt = torch.load(checkpoint_path, map_location=device)
    q_net = MLPQNetwork(cfg.input_dim, cfg.num_actions, cfg.hidden_size, cfg.num_layers, cfg.dropout).to(device)
    q_net.load_state_dict(ckpt["q_state_dict"])
    q_net.eval()
    return q_net


def evaluate_greedy_on_dataset(q_net, dataset_npz: str, batch_size: int = 2048) -> Dict:
    """
    A simple offline evaluation: compute average greedy Q chosen action value,
    and behavior cloning agreement (fraction of times greedy == logged action).
    """
    data = np.load(dataset_npz)
    states = data["states"]
    actions = data["actions"]
    if "avail_mask" in data:
        avail_mask = data["avail_mask"]
    else:
        avail_mask = None

    device = next(q_net.parameters()).device
    n = len(states)
    batch_size = int(batch_size)
    greedy_vals = []
    greedy_actions = []
    for i in range(0, n, batch_size):
        batch_states = torch.tensor(states[i:i+batch_size].astype(np.float32), device=device)
        with torch.no_grad():
            q = q_net(batch_states)  # [B, A]
            q_np = q.cpu().numpy()
        if avail_mask is not None:
            mask = avail_mask[i:i+batch_size]
            q_np = np.where(mask > 0, q_np, -1e9)
        greedy_batch = np.argmax(q_np, axis=1)
        greedy_actions.append(greedy_batch)
        greedy_vals.append(np.max(q_np, axis=1))

    greedy_actions = np.concatenate(greedy_actions, axis=0)[:n]
    greedy_vals = np.concatenate(greedy_vals, axis=0)[:n]
    behavior_agreement = np.mean(greedy_actions == actions)
    avg_greedy_q = float(np.mean(greedy_vals))

    return {
        "n": n,
        "behavior_agreement": float(behavior_agreement),
        "avg_greedy_q": float(avg_greedy_q),
    }


if __name__ == "__main__":
    # CLI: python src/ope/offline_eval_cql.py
    cfg = build_config_from_yamls()
    inference_cfg = load_yaml("configs/inference.yaml")
    ckpt = inference_cfg.get("load_checkpoint", None)
    if ckpt is None or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    qnet = load_checkpoint_model(ckpt, cfg)
    data_cfg = load_yaml("configs/data.yaml")
    dataset_path = os.path.join(data_cfg["processed_data_dir"], data_cfg["dataset_file"])
    results = evaluate_greedy_on_dataset(qnet, dataset_path)
    print("Offline evaluation results:", results)
