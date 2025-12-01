"""
CQL Model Training, Validation, and Evaluation for MLB Bullpen Management.

This script implements the complete pipeline for training and evaluating
Conservative Q-Learning (CQL) on the MLB bullpen dataset.
All configuration is loaded from YAML files in the config directory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


def load_config(config_dir: str = "configs") -> Dict:
    """Load all configuration files from the config directory."""
    config_files = {
        'model': 'model.yaml',
        'training': 'training.yaml', 
        'data': 'data.yaml',
        'env': 'env.yaml',
        'inference': 'inference.yaml'
    }
    
    configs = {}
    for key, filename in config_files.items():
        filepath = Path(config_dir) / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                configs[key] = yaml.safe_load(f)
            print(f"Loaded config: {filepath}")
        else:
            print(f"Warning: Config file {filepath} not found, using defaults")
    
    # Set comprehensive defaults for any missing configs
    configs = _set_default_configs(configs)
    
    # Validate required fields
    _validate_config(configs)
    
    return configs


def _set_default_configs(config: Dict) -> Dict:
    """Set default values for any missing configuration sections."""
    
    # Model defaults
    if 'model' not in config:
        config['model'] = {}
    config['model'].setdefault('hidden_size', 256)
    config['model'].setdefault('num_layers', 3)
    config['model'].setdefault('dropout', 0.05)
    config['model'].setdefault('num_actions', 11)
    config['model'].setdefault('model_type', 'CQL')
    
    # Training defaults
    if 'training' not in config:
        config['training'] = {}
    config['training'].setdefault('epochs', 30)
    config['training'].setdefault('batch_size', 2048)
    config['training'].setdefault('learning_rate', 0.0003)
    config['training'].setdefault('weight_decay', 0.00001)
    config['training'].setdefault('target_update_interval', 5000)
    config['training'].setdefault('tau_soft_update', 0.005)
    config['training'].setdefault('discount', 0.99)
    config['training'].setdefault('method', 'CQL')
    config['training'].setdefault('cql_alpha', 1.0)
    config['training'].setdefault('iql_expectile', 0.7)
    config['training'].setdefault('iql_temperature', 3.0)
    config['training'].setdefault('log_interval', 200)
    config['training'].setdefault('save_interval', 1)
    config['training'].setdefault('checkpoint_dir', 'checkpoints/')
    config['training'].setdefault('use_wandb', False)
    
    # Data defaults
    if 'data' not in config:
        config['data'] = {}
    config['data'].setdefault('raw_data_dir', 'data/raw')
    config['data'].setdefault('processed_data_dir', 'data/processed')
    config['data'].setdefault('dataset_file', 'rl_tensors_2023.npz')
    config['data'].setdefault('lineup_window', 5)
    config['data'].setdefault('max_relievers', 10)
    
    # Environment defaults
    if 'env' not in config:
        config['env'] = {}
    config['env'].setdefault('gamma', 0.99)
    config['env'].setdefault('penalty_pull', 0.005)
    config['env'].setdefault('max_smdp_horizon', 3)
    config['env'].setdefault('stay_action', 0)
    config['env'].setdefault('num_relievers', 10)
    config['env'].setdefault('reward_column', 'reward_folded')
    config['env'].setdefault('terminal_flags', {
        'half_inning_over': True,
        'game_over': True
    })
    
    # Inference defaults
    if 'inference' not in config:
        config['inference'] = {}
    config['inference'].setdefault('mask_unavailable', True)
    config['inference'].setdefault('force_stay_threshold', 0.0)
    config['inference'].setdefault('force_pull_threshold', 0.5)
    config['inference'].setdefault('map_pitcher_ids_to_names', True)
    config['inference'].setdefault('simulate_n_games', 100)
    
    return config


def _validate_config(config: Dict):
    """Validate that required configuration fields are present."""
    required_fields = {
        'data': ['processed_data_dir', 'dataset_file', 'lineup_window', 'max_relievers'],
        'training': ['epochs', 'batch_size', 'learning_rate', 'method'],
        'model': ['hidden_size', 'num_layers', 'model_type']
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field {field} in {section} configuration")


class BullpenDataset:
    """Dataset class for bullpen RL tensors."""
    
    def __init__(self, data_path: str, config: Dict):
        self.config = config
        self.data = np.load(data_path)
        
        # Store data shapes
        self.n_samples = len(self.data['state_vec'])
        self.game_state_dim = self.data['state_vec'].shape[1]
        self.lineup_feat_dim = self.data['next_hitters_feats'].shape[2]
        self.lineup_window = self.data['next_hitters_feats'].shape[1]
        self.reliever_feat_dim = self.data['reliever_feats'].shape[2]
        self.max_bullpen_size = self.data['reliever_feats'].shape[1]
        
        # Validate against config
        self._validate_against_config()
        
        print(f"Dataset loaded from {data_path}")
        print(f"Number of samples: {self.n_samples:,}")
        print(f"Game state dimension: {self.game_state_dim}")
        print(f"Lineup features: {self.lineup_feat_dim} × {self.lineup_window}")
        print(f"Bullpen features: {self.reliever_feat_dim} × {self.max_bullpen_size}")
        
        # Calculate class distribution
        self._calculate_statistics()
    
    def _validate_against_config(self):
        """Validate dataset dimensions against configuration."""
        config_lineup_window = self.config['data']['lineup_window']
        config_max_relievers = self.config['data']['max_relievers']
        
        if self.lineup_window != config_lineup_window:
            print(f"Warning: Dataset lineup window ({self.lineup_window}) doesn't match config ({config_lineup_window})")
        
        if self.max_bullpen_size != config_max_relievers:
            print(f"Warning: Dataset max relievers ({self.max_bullpen_size}) doesn't match config ({config_max_relievers})")
    
    def _validate_shapes(self):
        """Validate that all arrays have consistent shapes."""
        expected_length = self.n_samples
        
        arrays_to_check = [
            ('state_vec', self.data['state_vec']),
            ('next_hitters_feats', self.data['next_hitters_feats']),
            ('pos_enc', self.data['pos_enc']),
            ('reliever_feats', self.data['reliever_feats']),
            ('avail_mask', self.data['avail_mask']),
            ('action_idx', self.data['action_idx']),
            ('reward_folded', self.data['reward_folded']),
            ('done', self.data['done']),
            ('next_state_idx', self.data['next_state_idx'])
        ]
        
        for name, array in arrays_to_check:
            if len(array) != expected_length:
                raise ValueError(f"{name} has length {len(array)}, expected {expected_length}")
    
    def _calculate_statistics(self):
        """Calculate dataset statistics."""
        self.action_distribution = np.bincount(self.data['action_idx'], 
                                               minlength=self.max_bullpen_size + 1)
        self.mean_reward = np.mean(self.data['reward_folded'])
        self.std_reward = np.std(self.data['reward_folded'])
        
        print(f"\nDataset Statistics:")
        print(f"  Mean reward: {self.mean_reward:.4f}")
        print(f"  Reward std: {self.std_reward:.4f}")
        print(f"  Stay actions (0): {self.action_distribution[0]:,} ({self.action_distribution[0]/self.n_samples:.1%})")
        print(f"  Pull actions (>0): {self.action_distribution[1:].sum():,} ({self.action_distribution[1:].sum()/self.n_samples:.1%})")
        
        # Calculate availability statistics
        avg_available = self.data['avail_mask'].mean(axis=1).mean()
        print(f"  Average available relievers: {avg_available:.2f} / {self.max_bullpen_size}")
    
    def get_split_indices(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, np.ndarray]:
        """Split indices into train, validation, and test sets."""
        n_train = int(self.n_samples * train_ratio)
        n_val = int(self.n_samples * val_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(self.n_samples)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        print(f"\nData Split:")
        print(f"  Training: {len(train_indices):,} samples ({len(train_indices)/self.n_samples:.1%})")
        print(f"  Validation: {len(val_indices):,} samples ({len(val_indices)/self.n_samples:.1%})")
        print(f"  Test: {len(test_indices):,} samples ({len(test_indices)/self.n_samples:.1%})")
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    
    def get_batch(self, indices: np.ndarray, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Get batch data for given indices."""
        # Convert indices to list for consistent slicing
        idx_list = indices.tolist()
        
        # Get next state indices
        next_idx = self.data['next_state_idx'][idx_list]
        
        batch = {
            'game_state': torch.FloatTensor(self.data['state_vec'][idx_list]),
            'lineup_feats': torch.FloatTensor(self.data['next_hitters_feats'][idx_list]),
            'pos_enc': torch.FloatTensor(self.data['pos_enc'][idx_list]),
            'reliever_feats': torch.FloatTensor(self.data['reliever_feats'][idx_list]),
            'avail_mask': torch.BoolTensor(self.data['avail_mask'][idx_list]),
            'action': torch.LongTensor(self.data['action_idx'][idx_list]),
            'reward': torch.FloatTensor(self.data['reward_folded'][idx_list]),
            'done': torch.BoolTensor(self.data['done'][idx_list]),
            # Next states
            'next_game_state': torch.FloatTensor(self.data['state_vec'][next_idx]),
            'next_lineup_feats': torch.FloatTensor(self.data['next_hitters_feats'][next_idx]),
            'next_pos_enc': torch.FloatTensor(self.data['pos_enc'][next_idx]),
            'next_reliever_feats': torch.FloatTensor(self.data['reliever_feats'][next_idx]),
            'next_avail_mask': torch.BoolTensor(self.data['avail_mask'][next_idx])
        }
        
        # Move to device
        if device != 'cpu':
            batch = {k: v.to(device) for k, v in batch.items()}
            
        return batch


class MultiModalQNetwork(nn.Module):
    """Q-network for bullpen management with multi-modal state."""
    
    def __init__(self, 
                 game_state_dim: int,
                 lineup_feat_dim: int,
                 lineup_window: int,
                 reliever_feat_dim: int,
                 max_bullpen_size: int,
                 config: Dict):
        super().__init__()
        
        self.config = config['model']
        self.lineup_window = lineup_window
        self.max_bullpen_size = max_bullpen_size
        self.num_actions = 1 + max_bullpen_size  # 0 = stay, 1..R = relievers
        
        hidden_dim = self.config['hidden_size']
        num_layers = self.config['num_layers']
        dropout = self.config.get('dropout', 0.05)
        
        # Game state encoder
        self.game_state_encoder = nn.Sequential(
            nn.Linear(game_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Lineup encoder
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
        
        # Combined network
        combined_dim = hidden_dim * 3
        
        layers = []
        current_dim = combined_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, self.num_actions))
        self.q_net = nn.Sequential(*layers)
        
        print(f"Initialized MultiModalQNetwork with {self.num_actions} actions")
        print(f"  Hidden size: {hidden_dim}")
        print(f"  Layers: {num_layers}")
        print(f"  Dropout: {dropout}")
    
    def forward(self, 
                game_state: torch.Tensor,
                lineup_feats: torch.Tensor,
                pos_enc: torch.Tensor,
                reliever_feats: torch.Tensor,
                avail_mask: torch.Tensor) -> torch.Tensor:
        
        batch_size = game_state.shape[0]
        
        # Game state encoding
        game_encoded = self.game_state_encoder(game_state)
        
        # Lineup encoding
        lineup_with_pos = torch.cat([lineup_feats, pos_enc], dim=-1)
        lineup_flat = lineup_with_pos.view(batch_size * self.lineup_window, -1)
        lineup_encoded = self.lineup_encoder(lineup_flat)
        lineup_encoded = lineup_encoded.view(batch_size, self.lineup_window, -1)
        lineup_aggregated = self.lineup_aggregator(lineup_encoded.view(batch_size, -1))
        
        # Bullpen encoding
        reliever_flat = reliever_feats.view(batch_size * self.max_bullpen_size, -1)
        reliever_encoded = self.reliever_encoder(reliever_flat)
        reliever_encoded = reliever_encoded.view(batch_size, self.max_bullpen_size, -1)
        reliever_aggregated = self.reliever_aggregator(reliever_encoded.view(batch_size, -1))
        
        # Combine features
        combined = torch.cat([game_encoded, lineup_aggregated, reliever_aggregated], dim=-1)
        
        # Q-values
        q_values = self.q_net(combined)
        
        # Mask unavailable actions
        action_mask = torch.ones(batch_size, self.num_actions, device=q_values.device)
        action_mask[:, 1:] = avail_mask.float()
        masked_q_values = q_values - (1 - action_mask) * 1e9
        
        return masked_q_values


class CQLAgent:
    """Conservative Q-Learning agent."""
    
    def __init__(self, 
                 game_state_dim: int,
                 lineup_feat_dim: int,
                 lineup_window: int,
                 reliever_feat_dim: int,
                 max_bullpen_size: int,
                 config: Dict,
                 device: str = "cuda"):
        
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load parameters from config
        training_config = config['training']
        self.gamma = training_config['discount']
        self.tau = training_config['tau_soft_update']
        self.cql_alpha = training_config['cql_alpha']
        self.target_update_interval = training_config['target_update_interval']
        self.method = training_config['method']
        
        self.steps = 0
        
        # Initialize networks
        self.q_network = MultiModalQNetwork(
            game_state_dim=game_state_dim,
            lineup_feat_dim=lineup_feat_dim,
            lineup_window=lineup_window,
            reliever_feat_dim=reliever_feat_dim,
            max_bullpen_size=max_bullpen_size,
            config=config
        ).to(self.device)
        
        self.target_network = MultiModalQNetwork(
            game_state_dim=game_state_dim,
            lineup_feat_dim=lineup_feat_dim,
            lineup_window=lineup_window,
            reliever_feat_dim=reliever_feat_dim,
            max_bullpen_size=max_bullpen_size,
            config=config
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), 
            lr=training_config['learning_rate'], 
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000, 
            eta_min=1e-6
        )
        
        print(f"\nCQL Agent initialized:")
        print(f"  Device: {self.device}")
        print(f"  Method: {self.method}")
        print(f"  Gamma: {self.gamma}")
        print(f"  CQL alpha: {self.cql_alpha}")
        print(f"  Target update interval: {self.target_update_interval}")
        print(f"  Learning rate: {training_config['learning_rate']}")
        print(f"  Number of parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update the Q-network using CQL loss."""
        self.steps += 1
        
        # Target Q-values
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
        
        # Current Q-values
        current_q_values = self.q_network(
            batch['game_state'],
            batch['lineup_feats'],
            batch['pos_enc'],
            batch['reliever_feats'],
            batch['avail_mask']
        )
        
        action_q_values = current_q_values.gather(1, batch['action'].unsqueeze(1))
        
        # TD loss
        td_loss = nn.MSELoss()(action_q_values, target_q_values.unsqueeze(1))
        
        # CQL regularization
        batch_size = batch['game_state'].shape[0]
        logsumexp_q = torch.logsumexp(current_q_values, dim=1).mean()
        data_q = action_q_values.mean()
        cql_loss = self.cql_alpha * (logsumexp_q - data_q)
        
        # Total loss
        total_loss = td_loss + cql_loss
        
        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network
        if self.steps % self.target_update_interval == 0:
            self.soft_update_target_network()
        
        return {
            'total_loss': total_loss.item(),
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'mean_q': current_q_values.mean().item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def soft_update_target_network(self):
        """Soft update target network parameters."""
        for target_param, param in zip(self.target_network.parameters(), 
                                      self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on a batch."""
        with torch.no_grad():
            q_values = self.q_network(
                batch['game_state'],
                batch['lineup_feats'],
                batch['pos_enc'],
                batch['reliever_feats'],
                batch['avail_mask']
            )
            
            predicted_actions = q_values.argmax(dim=1)
            actual_actions = batch['action']
            
            accuracy = (predicted_actions == actual_actions).float().mean().item()
            
            # Stay accuracy
            stay_mask = actual_actions == 0
            stay_accuracy = (predicted_actions[stay_mask] == 0).float().mean().item() if stay_mask.any() else 0.0
            
            # Pull accuracy
            pull_mask = actual_actions > 0
            pull_accuracy = (predicted_actions[pull_mask] == actual_actions[pull_mask]).float().mean().item() if pull_mask.any() else 0.0
            
            # Q-value statistics
            mean_q = q_values.mean().item()
            mean_stay_q = q_values[:, 0].mean().item()
            mean_pull_q = q_values[:, 1:].mean().item()
            
            return {
                'accuracy': accuracy,
                'stay_accuracy': stay_accuracy,
                'pull_accuracy': pull_accuracy,
                'mean_q': mean_q,
                'mean_stay_q': mean_stay_q,
                'mean_pull_q': mean_pull_q,
                'action_distribution': predicted_actions.cpu().numpy()
            }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'steps': self.steps,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps = checkpoint['steps']
        print(f"Checkpoint loaded from {path}")