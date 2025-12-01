import torch
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, List
import time
from torch.utils.data import DataLoader, TensorDataset
import warnings

from cql_model import CQLAgent

class BullpenDataLoader:
    """
    Data loader for the bullpen RL dataset.
    """
    
    def __init__(self, data_path: str, device: str = "cpu"):
        self.device = device
        self.data = np.load(data_path)
        
        # Extract feature dimensions from data
        self.game_state_dim = self.data['state_vec'].shape[1]
        self.lineup_feat_dim = self.data['next_hitters_feats'].shape[2]
        self.lineup_window = self.data['next_hitters_feats'].shape[1]
        self.reliever_feat_dim = self.data['reliever_feats'].shape[2]
        self.max_bullpen_size = self.data['reliever_feats'].shape[1]
        
        print(f"Data shapes:")
        print(f"  State vector: {self.data['state_vec'].shape}")
        print(f"  Lineup features: {self.data['next_hitters_feats'].shape}")
        print(f"  Positional encodings: {self.data['pos_enc'].shape}")
        print(f"  Reliever features: {self.data['reliever_feats'].shape}")
        print(f"  Availability mask: {self.data['avail_mask'].shape}")
        print(f"  Actions: {self.data['action_idx'].shape}")
        print(f"  Rewards: {self.data['reward_folded'].shape}")
        print(f"  Done flags: {self.data['done'].shape}")
    
    def create_data_loader(self, batch_size: int = 2048, shuffle: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader for training.
        """
        # Convert to tensors
        game_state = torch.FloatTensor(self.data['state_vec'])
        lineup_feats = torch.FloatTensor(self.data['next_hitters_feats'])
        pos_enc = torch.FloatTensor(self.data['pos_enc'])
        reliever_feats = torch.FloatTensor(self.data['reliever_feats'])
        avail_mask = torch.BoolTensor(self.data['avail_mask'])
        
        # Next state tensors
        next_state_indices = self.data['next_state_idx']
        next_game_state = torch.FloatTensor(self.data['state_vec'][next_state_indices])
        next_lineup_feats = torch.FloatTensor(self.data['next_hitters_feats'][next_state_indices])
        next_pos_enc = torch.FloatTensor(self.data['pos_enc'][next_state_indices])
        next_reliever_feats = torch.FloatTensor(self.data['reliever_feats'][next_state_indices])
        next_avail_mask = torch.BoolTensor(self.data['avail_mask'][next_state_indices])
        
        # Actions, rewards, done flags
        actions = torch.LongTensor(self.data['action_idx'])
        rewards = torch.FloatTensor(self.data['reward_folded'])
        done = torch.BoolTensor(self.data['done'])
        
        # Create dataset
        dataset = TensorDataset(
            game_state, lineup_feats, pos_enc, reliever_feats, avail_mask,
            next_game_state, next_lineup_feats, next_pos_enc, next_reliever_feats, next_avail_mask,
            actions, rewards, done
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Get specific batch by indices.
        """
        return {
            'game_state': torch.FloatTensor(self.data['state_vec'][indices]).to(self.device),
            'lineup_feats': torch.FloatTensor(self.data['next_hitters_feats'][indices]).to(self.device),
            'pos_enc': torch.FloatTensor(self.data['pos_enc'][indices]).to(self.device),
            'reliever_feats': torch.FloatTensor(self.data['reliever_feats'][indices]).to(self.device),
            'avail_mask': torch.BoolTensor(self.data['avail_mask'][indices]).to(self.device),
            'next_game_state': torch.FloatTensor(self.data['state_vec'][self.data['next_state_idx'][indices]]).to(self.device),
            'next_lineup_feats': torch.FloatTensor(self.data['next_hitters_feats'][self.data['next_state_idx'][indices]]).to(self.device),
            'next_pos_enc': torch.FloatTensor(self.data['pos_enc'][self.data['next_state_idx'][indices]]).to(self.device),
            'next_reliever_feats': torch.FloatTensor(self.data['reliever_feats'][self.data['next_state_idx'][indices]]).to(self.device),
            'next_avail_mask': torch.BoolTensor(self.data['avail_mask'][self.data['next_state_idx'][indices]]).to(self.device),
            'action': torch.LongTensor(self.data['action_idx'][indices]).to(self.device),
            'reward': torch.FloatTensor(self.data['reward_folded'][indices]).to(self.device),
            'done': torch.BoolTensor(self.data['done'][indices]).to(self.device)
        }


def load_config(config_dir: str = "config") -> Dict:
    """Load all configuration files."""
    config_files = {
        'model': 'model.yaml',
        'training': 'training.yaml', 
        'data': 'data.yaml',
        'env': 'env.yaml'
    }
    
    config = {}
    for key, filename in config_files.items():
        filepath = Path(config_dir) / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                config[key] = yaml.safe_load(f)
        else:
            warnings.warn(f"Config file {filepath} not found, using defaults")
    
    return config


def train_cql():
    """Main training function."""
    # Load configuration
    config = load_config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_path = Path(config['data']['processed_data_dir']) / config['data']['dataset_file']
    data_loader = BullpenDataLoader(str(data_path), device=device)
    
    # Create agent
    agent = CQLAgent(
        game_state_dim=data_loader.game_state_dim,
        lineup_feat_dim=data_loader.lineup_feat_dim,
        lineup_window=data_loader.lineup_window,
        reliever_feat_dim=data_loader.reliever_feat_dim,
        max_bullpen_size=data_loader.max_bullpen_size,
        hidden_dim=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        learning_rate=config['training']['learning_rate'],
        gamma=config['env']['gamma'],
        tau=config['training']['tau_soft_update'],
        cql_alpha=config['training']['cql_alpha'],
        target_update_interval=config['training']['target_update_interval'],
        device=device
    )
    
    # Create data loader
    train_loader = data_loader.create_data_loader(
        batch_size=config['training']['batch_size']
    )
    
    # Training loop
    epochs = config['training']['epochs']
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Convert batch data to dictionary format
            batch = {
                'game_state': batch_data[0].to(device),
                'lineup_feats': batch_data[1].to(device),
                'pos_enc': batch_data[2].to(device),
                'reliever_feats': batch_data[3].to(device),
                'avail_mask': batch_data[4].to(device),
                'next_game_state': batch_data[5].to(device),
                'next_lineup_feats': batch_data[6].to(device),
                'next_pos_enc': batch_data[7].to(device),
                'next_reliever_feats': batch_data[8].to(device),
                'next_avail_mask': batch_data[9].to(device),
                'action': batch_data[10].to(device),
                'reward': batch_data[11].to(device),
                'done': batch_data[12].to(device)
            }
            
            # Update agent
            loss_info = agent.update(batch)
            epoch_losses.append(loss_info)
            
            # Logging
            if batch_idx % config['training']['log_interval'] == 0:
                avg_loss = np.mean([l['total_loss'] for l in epoch_losses[-100:]])
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | "
                      f"Loss: {avg_loss:.4f} | Steps: {agent.steps}")
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        avg_total_loss = np.mean([l['total_loss'] for l in epoch_losses])
        avg_td_loss = np.mean([l['td_loss'] for l in epoch_losses])
        avg_cql_loss = np.mean([l['cql_loss'] for l in epoch_losses])
        avg_q = np.mean([l['mean_q'] for l in epoch_losses])
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"  Total Loss: {avg_total_loss:.4f}")
        print(f"  TD Loss: {avg_td_loss:.4f}")
        print(f"  CQL Loss: {avg_cql_loss:.4f}")
        print(f"  Mean Q: {avg_q:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = checkpoint_dir / f"cql_epoch_{epoch+1}.pth"
            agent.save_checkpoint(str(checkpoint_path))
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / "cql_final.pth"
    agent.save_checkpoint(str(final_path))
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    train_cql()