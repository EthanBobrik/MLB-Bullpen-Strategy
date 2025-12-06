
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_rl_tensors(path: str):
    data = np.load(path)

    state_vec = data["state_vec"]
    next_state_vec = data["next_state_vec"]
    action_idx = data["action_idx"]
    reward = data["reward_folded"]
    done = data["done"]
    avail_mask_rel = data["avail_mask"]

    B, R = avail_mask_rel.shape
    action_dim = 1 + R

    avail_mask_full = np.zeros((B, action_dim), dtype=bool)
    avail_mask_full[:, 0] = True
    avail_mask_full[:, 1:] = avail_mask_rel

    return {
        "state": torch.from_numpy(state_vec).float(),
        "next_state": torch.from_numpy(next_state_vec).float(),
        "action": torch.from_numpy(action_idx).long(),
        "reward": torch.from_numpy(reward).float(),
        "done": torch.from_numpy(done.astype(np.float32)),
        "avail_mask": torch.from_numpy(avail_mask_full),
        "action_dim": action_dim,
    }


def train_q_learning(
    npz_path: str,
    num_epochs: int = 20,
    batch_size: int = 1024,
    gamma: float = 0.99,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    print_every: int = 1,
    device: str | None = None,
):
    data = load_rl_tensors(npz_path)

    state = data["state"]
    next_state = data["next_state"]
    action = data["action"]
    reward = data["reward"]
    done = data["done"]
    avail_mask = data["avail_mask"]

    N, state_dim = state.shape
    action_dim = data["action_dim"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = state.to(device)
    next_state = next_state.to(device)
    action = action.to(device)
    reward = reward.to(device)
    done = done.to(device)
    avail_mask = avail_mask.to(device)

    q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    indices = torch.arange(N, device=device)

    for epoch in range(1, num_epochs + 1):
        perm = indices[torch.randperm(N)]
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = perm[start:end]

            s = state[batch_idx]
            s_next = next_state[batch_idx]
            a = action[batch_idx]
            r = reward[batch_idx].unsqueeze(1)
            d = done[batch_idx].unsqueeze(1)
            mask_next = avail_mask[batch_idx]

            q_values = q_net(s)
            q_sa = q_values.gather(1, a.unsqueeze(1))

            with torch.no_grad():
                q_next_all = q_net(s_next)
                large_negative = -1e9
                invalid_mask = ~mask_next
                q_next_all = q_next_all.masked_fill(invalid_mask, large_negative)
                q_next_max, _ = q_next_all.max(dim=1, keepdim=True)
                target = r + gamma * (1.0 - d) * q_next_max

            loss = loss_fn(q_sa, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if epoch % print_every == 0:
            print(f"Epoch {epoch} | Avg TD loss: {epoch_loss / max(num_batches, 1):.6f}")

    return q_net


@torch.no_grad()
def select_action(q_net: QNetwork, state_vec: np.ndarray, avail_mask_row: np.ndarray, epsilon: float = 0.0, device: str | None = None) -> int:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    avail_mask_row = np.asarray(avail_mask_row, dtype=bool)
    valid_actions = np.where(avail_mask_row)[0]
    if len(valid_actions) == 0:
        return 0

    if np.random.rand() < epsilon:
        return int(np.random.choice(valid_actions))

    s = torch.from_numpy(state_vec).float().to(device).unsqueeze(0)
    q_values = q_net(s).cpu().numpy()[0]

    q_values[~avail_mask_row] = -1e9
    return int(q_values.argmax())
