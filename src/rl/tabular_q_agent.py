import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional


def discretize_state(state: np.ndarray, precision: int = 2) -> Tuple:
    """
    Convert a continuous state vector into a hashable tuple for tabular Q-learning.
    """
    return tuple(np.round(state, precision))


class TabularQAgent:
    def __init__(
        self,
        num_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        precision: int = 2,
    ):
        """
        num_actions: 1 + R (0=stay, 1..R=reliever slots)
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.precision = precision

        # Q: dict[state_tuple] -> np.array(num_actions)
        self.Q: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32)
        )

    def _s(self, state: np.ndarray) -> Tuple:
        return discretize_state(state, precision=self.precision)

    def td_update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        s = self._s(state)
        ns = self._s(next_state)

        q_sa = self.Q[s][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[ns])

        self.Q[s][action] = q_sa + self.alpha * (target - q_sa)

    def batch_train(
        self,
        state_vec: np.ndarray,
        action_idx: np.ndarray,
        reward_vec: np.ndarray,
        next_state_vec: np.ndarray,
        done_vec: np.ndarray,
        num_epochs: int = 20,
        shuffle: bool = True,
    ) -> None:
        """
        Offline Q-learning over the static dataset.
        """
        B = len(state_vec)
        for epoch in range(num_epochs):
            if shuffle:
                idx = np.random.permutation(B)
            else:
                idx = np.arange(B)

            for i in idx:
                self.td_update(
                    state=state_vec[i],
                    action=int(action_idx[i]),
                    reward=float(reward_vec[i]),
                    next_state=next_state_vec[i],
                    done=bool(done_vec[i]),
                )

            print(f"[TabularQAgent] Epoch {epoch+1}/{num_epochs} complete.")

    def act(
        self,
        state: np.ndarray,
        avail_mask: Optional[np.ndarray] = None,
        epsilon: float = 0.0,
    ) -> int:
        """
        ε-greedy action (optionally respecting an availability mask).
        avail_mask: shape (num_actions,), bool (True = allowed).
        """
        s = self._s(state)
        q = self.Q[s].copy()

        if avail_mask is not None:
            invalid = ~avail_mask
            q[invalid] = -1e9

        if np.random.rand() < epsilon:
            if avail_mask is not None:
                valid_actions = np.where(avail_mask)[0]
            else:
                valid_actions = np.arange(self.num_actions)
            return int(np.random.choice(valid_actions))

        return int(np.argmax(q))

    def value(self, state: np.ndarray) -> float:
        s = self._s(state)
        return float(np.max(self.Q[s]))

    def save(self, path: str) -> None:
        np.save(path, dict(self.Q))

    def load(self, path: str) -> None:
        raw = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32),
            raw,
        )
