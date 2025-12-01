import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

class MultiModalQNetwork(nn.Module):
    """
    Q-network that handles the multi-modal state representation for bullpen management.
    
    State components:
    - game_state: [B, 13] - inning, outs, base_state, score_diff, fatigue, matchup
    - lineup_feats: [B, H, 18] - next H hitters with batter stats
    - pos_enc: [B, H, H] - positional encodings for lineup
    - reliever_feats: [B, R, 8] - bullpen reliever form metrics
    - avail_mask: [B, R] - availability mask for relievers
    """
    
    def __init__(self, 
                 game_state_dim: int = 13,
                 lineup_feat_dim: int = 18,
                 lineup_window: int = 5,
                 reliever_feat_dim: int = 8,
                 max_bullpen_size: int = 10,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.05):
        super(MultiModalQNetwork, self).__init__()
        
        self.lineup_window = lineup_window
        self.max_bullpen_size = max_bullpen_size
        self.num_actions = 1 + max_bullpen_size  # 0 = stay, 1..R = relievers
        
        # Game state encoder
        self.game_state_encoder = nn.Sequential(
            nn.Linear(game_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Lineup feature encoder 
        self.lineup_encoder = nn.Sequential(
            nn.Linear(lineup_feat_dim + lineup_window, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        self.lineup_aggregator = nn.Sequential(
            nn.Linear((hidden_dim // 2) * lineup_window, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Bullpen encoder
        self.reliever_encoder = nn.Sequential(
            nn.Linear(reliever_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        self.reliever_aggregator = nn.Sequential(
            nn.Linear((hidden_dim // 2) * max_bullpen_size, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Combined feature processing
        combined_dim = hidden_dim * 3  # game_state + lineup + bullpen
        
        # Q-value layers
        layers = []
        current_dim = combined_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, self.num_actions))
        self.q_net = nn.Sequential(*layers)
        
    def forward(self, 
                game_state: torch.Tensor,
                lineup_feats: torch.Tensor,
                pos_enc: torch.Tensor,
                reliever_feats: torch.Tensor,
                avail_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q-network.
        
        Args:
            game_state: [B, 13]
            lineup_feats: [B, H, 18] 
            pos_enc: [B, H, H]
            reliever_feats: [B, R, 8]
            avail_mask: [B, R]
            
        Returns:
            q_values: [B, num_actions] where num_actions = 1 + R
        """
        batch_size = game_state.shape[0]
        
        # Encode game state
        game_encoded = self.game_state_encoder(game_state)  # [B, hidden_dim]
        
        # Encode lineup features
        lineup_with_pos = torch.cat([lineup_feats, pos_enc], dim=-1)  # [B, H, 18 + H]
        lineup_encoded = self.lineup_encoder(
            lineup_with_pos.view(batch_size * self.lineup_window, -1)
        )  # [B*H, hidden_dim//2]
        
        lineup_encoded = lineup_encoded.view(
            batch_size, self.lineup_window, -1
        )  # [B, H, hidden_dim//2]
        
        lineup_aggregated = self.lineup_aggregator(
            lineup_encoded.view(batch_size, -1)
        )  # [B, hidden_dim]
        
        # Encode bullpen features
        reliever_encoded = self.reliever_encoder(
            reliever_feats.view(batch_size * self.max_bullpen_size, -1)
        )  # [B*R, hidden_dim//2]
        
        reliever_encoded = reliever_encoded.view(
            batch_size, self.max_bullpen_size, -1
        )  # [B, R, hidden_dim//2]
        
        reliever_aggregated = self.reliever_aggregator(
            reliever_encoded.view(batch_size, -1)
        )  # [B, hidden_dim]
        
        # Combine all features
        combined = torch.cat([
            game_encoded,           # [B, hidden_dim]
            lineup_aggregated,      # [B, hidden_dim]  
            reliever_aggregated     # [B, hidden_dim]
        ], dim=-1)  # [B, hidden_dim * 3]
        
        # Compute Q-values
        q_values = self.q_net(combined)  # [B, num_actions]
        
        # Apply availability mask - set unavailable actions to very negative value
        action_mask = torch.ones(batch_size, self.num_actions, device=q_values.device)
        action_mask[:, 1:] = avail_mask.float()  # Action 0 (stay) is always available
        
        masked_q_values = q_values - (1 - action_mask) * 1e9
        
        return masked_q_values


class CQLAgent:
    """
    Conservative Q-Learning agent for MLB bullpen management.
    """
    
    def __init__(self,
                 game_state_dim: int = 13,
                 lineup_feat_dim: int = 18,
                 lineup_window: int = 5,
                 reliever_feat_dim: int = 8,
                 max_bullpen_size: int = 10,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 cql_alpha: float = 1.0,
                 target_update_interval: int = 5000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.target_update_interval = target_update_interval
        self.steps = 0
        
        # Networks
        self.q_network = MultiModalQNetwork(
            game_state_dim=game_state_dim,
            lineup_feat_dim=lineup_feat_dim,
            lineup_window=lineup_window,
            reliever_feat_dim=reliever_feat_dim,
            max_bullpen_size=max_bullpen_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        self.target_network = MultiModalQNetwork(
            game_state_dim=game_state_dim,
            lineup_feat_dim=lineup_feat_dim,
            lineup_window=lineup_window,
            reliever_feat_dim=reliever_feat_dim,
            max_bullpen_size=max_bullpen_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(self.q_network.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=1e-5)
        
    def get_action(self, 
                   game_state: np.ndarray,
                   lineup_feats: np.ndarray, 
                   pos_enc: np.ndarray,
                   reliever_feats: np.ndarray,
                   avail_mask: np.ndarray,
                   epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        """
        if np.random.random() < epsilon:
            # Random action, but only among available ones
            available_actions = [0]  # Stay is always available
            available_actions.extend(i+1 for i in range(len(avail_mask)) if avail_mask[i])
            return np.random.choice(available_actions)
        else:
            with torch.no_grad():
                state_tensors = self._prepare_state_tensors(
                    game_state, lineup_feats, pos_enc, reliever_feats, avail_mask
                )
                q_values = self.q_network(*state_tensors)
                return q_values.argmax(dim=1).item()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the Q-network using CQL loss.
        """
        self.steps += 1
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(
                batch['next_game_state'],
                batch['next_lineup_feats'],
                batch['next_pos_enc'], 
                batch['next_reliever_feats'],
                batch['next_avail_mask']
            )
            next_q_values = next_q_values.max(1)[0]
            target_q_values = batch['reward'] + (1 - batch['done'].float()) * self.gamma * next_q_values
        
        # Current Q-values for taken actions
        current_q_values = self.q_network(
            batch['game_state'],
            batch['lineup_feats'],
            batch['pos_enc'],
            batch['reliever_feats'],
            batch['avail_mask']
        )
        action_q_values = current_q_values.gather(1, batch['action'].unsqueeze(1))
        
        # Standard TD loss
        td_loss = nn.MSELoss()(action_q_values, target_q_values.unsqueeze(1))
        
        # CQL regularization loss
        batch_size = batch['game_state'].shape[0]
        logsumexp_q = torch.logsumexp(current_q_values, dim=1).mean()
        data_q = action_q_values.mean()
        
        cql_loss = self.cql_alpha * (logsumexp_q - data_q)
        
        # Total loss
        total_loss = td_loss + cql_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.target_update_interval == 0:
            self.soft_update_target_network()
        
        return {
            'total_loss': total_loss.item(),
            'td_loss': td_loss.item(), 
            'cql_loss': cql_loss.item(),
            'mean_q': current_q_values.mean().item()
        }
    
    def soft_update_target_network(self):
        """Soft update target network parameters."""
        for target_param, param in zip(self.target_network.parameters(), 
                                      self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def _prepare_state_tensors(self, 
                              game_state: np.ndarray,
                              lineup_feats: np.ndarray,
                              pos_enc: np.ndarray,
                              reliever_feats: np.ndarray, 
                              avail_mask: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """Convert numpy arrays to torch tensors."""
        return (
            torch.FloatTensor(game_state).unsqueeze(0).to(self.device),
            torch.FloatTensor(lineup_feats).unsqueeze(0).to(self.device),
            torch.FloatTensor(pos_enc).unsqueeze(0).to(self.device),
            torch.FloatTensor(reliever_feats).unsqueeze(0).to(self.device),
            torch.BoolTensor(avail_mask).unsqueeze(0).to(self.device)
        )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']