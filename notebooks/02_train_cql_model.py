# train_evaluate.py
"""
Training, Validation, and Evaluation Script for CQL Model.

This file handles the training loop, validation, and evaluation.
It imports the model from cql_model.py.
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Import the model from the separate file
from cql_model import BullpenDataset, CQLAgent, load_config


def create_data_loader(dataset: BullpenDataset, indices: np.ndarray, 
                      batch_size: int = 2048, shuffle: bool = True):
    """Create a data loader for given indices."""
    class BullpenSubset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self.length = len(indices)
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            
            if isinstance(idx, int):
                idx = [idx]
            
            actual_indices = self.indices[idx]
            batch = dataset.get_batch(actual_indices, device='cpu')
            
            # Return as tuple for DataLoader
            return (
                batch['game_state'], batch['lineup_feats'], batch['pos_enc'],
                batch['reliever_feats'], batch['avail_mask'],
                batch['next_game_state'], batch['next_lineup_feats'],
                batch['next_pos_enc'], batch['next_reliever_feats'],
                batch['next_avail_mask'], batch['action'], batch['reward'],
                batch['done']
            )
    
    subset = BullpenSubset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


def train_epoch(epoch: int, train_loader, agent: CQLAgent, log_interval: int = 200):
    """Train for one epoch."""
    agent.q_network.train()
    
    epoch_losses = []
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Move batch to device
        batch = {
            'game_state': batch_data[0].to(agent.device),
            'lineup_feats': batch_data[1].to(agent.device),
            'pos_enc': batch_data[2].to(agent.device),
            'reliever_feats': batch_data[3].to(agent.device),
            'avail_mask': batch_data[4].to(agent.device),
            'next_game_state': batch_data[5].to(agent.device),
            'next_lineup_feats': batch_data[6].to(agent.device),
            'next_pos_enc': batch_data[7].to(agent.device),
            'next_reliever_feats': batch_data[8].to(agent.device),
            'next_avail_mask': batch_data[9].to(agent.device),
            'action': batch_data[10].to(agent.device),
            'reward': batch_data[11].to(agent.device),
            'done': batch_data[12].to(agent.device)
        }
        
        # Update agent
        loss_info = agent.update(batch)
        epoch_losses.append(loss_info)
        
        # Logging
        if batch_idx % log_interval == 0:
            avg_loss = np.mean([l['total_loss'] for l in epoch_losses[-100:]])
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | Steps: {agent.steps}")
    
    # Aggregate epoch metrics
    if epoch_losses:
        metrics = {
            'total_loss': np.mean([l['total_loss'] for l in epoch_losses]),
            'td_loss': np.mean([l['td_loss'] for l in epoch_losses]),
            'cql_loss': np.mean([l['cql_loss'] for l in epoch_losses]),
            'mean_q': np.mean([l['mean_q'] for l in epoch_losses]),
            'learning_rate': epoch_losses[-1]['learning_rate']
        }
    else:
        metrics = {}
    
    return metrics


def validate(agent: CQLAgent, val_loader) -> Dict[str, float]:
    """Validate the model."""
    agent.q_network.eval()
    
    all_metrics = []
    action_distributions = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            batch = {
                'game_state': batch_data[0].to(agent.device),
                'lineup_feats': batch_data[1].to(agent.device),
                'pos_enc': batch_data[2].to(agent.device),
                'reliever_feats': batch_data[3].to(agent.device),
                'avail_mask': batch_data[4].to(agent.device),
                'next_game_state': batch_data[5].to(agent.device),
                'next_lineup_feats': batch_data[6].to(agent.device),
                'next_pos_enc': batch_data[7].to(agent.device),
                'next_reliever_feats': batch_data[8].to(agent.device),
                'next_avail_mask': batch_data[9].to(agent.device),
                'action': batch_data[10].to(agent.device),
                'reward': batch_data[11].to(agent.device),
                'done': batch_data[12].to(agent.device)
            }
            
            metrics = agent.evaluate_batch(batch)
            all_metrics.append(metrics)
            action_distributions.append(metrics['action_distribution'])
    
    # Aggregate metrics
    aggregated = {
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'stay_accuracy': np.mean([m['stay_accuracy'] for m in all_metrics]),
        'pull_accuracy': np.mean([m['pull_accuracy'] for m in all_metrics]),
        'mean_q': np.mean([m['mean_q'] for m in all_metrics]),
        'mean_stay_q': np.mean([m['mean_stay_q'] for m in all_metrics]),
        'mean_pull_q': np.mean([m['mean_pull_q'] for m in all_metrics])
    }
    
    # Combine action distributions
    if action_distributions:
        all_actions = np.concatenate(action_distributions)
        aggregated['action_distribution'] = np.bincount(all_actions, 
                                                       minlength=agent.q_network.num_actions)
    
    return aggregated


def analyze_decision_patterns(agent: CQLAgent, dataset: BullpenDataset, 
                            indices: np.ndarray, n_samples: int = 5000) -> pd.DataFrame:
    """Analyze decision patterns for given indices."""
    if len(indices) > n_samples:
        sample_indices = np.random.choice(indices, n_samples, replace=False)
    else:
        sample_indices = indices
    
    batch = dataset.get_batch(sample_indices, device=agent.device)
    
    with torch.no_grad():
        q_values = agent.q_network(
            batch['game_state'],
            batch['lineup_feats'],
            batch['pos_enc'],
            batch['reliever_feats'],
            batch['avail_mask']
        )
        
        predicted_actions = q_values.argmax(dim=1).cpu().numpy()
        actual_actions = batch['action'].cpu().numpy()
        
        # Get Q-values for stay and best reliever
        stay_q = q_values[:, 0].cpu().numpy()
        best_reliever_q = q_values[:, 1:].max(dim=1)[0].cpu().numpy()
        q_diff = best_reliever_q - stay_q
        
        # Get game state features
        game_state = batch['game_state'].cpu().numpy()
        rewards = batch['reward'].cpu().numpy()
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'predicted_action': predicted_actions,
            'actual_action': actual_actions,
            'stay_q': stay_q,
            'best_reliever_q': best_reliever_q,
            'q_difference': q_diff,
            'correct_prediction': predicted_actions == actual_actions,
            'predicted_stay': predicted_actions == 0,
            'actual_stay': actual_actions == 0,
            'reward': rewards
        })
        
        # Add game state features
        state_cols = ['inning', 'half', 'outs', 'base_state', 'score_diff', 
                      'pitch_count', 'tto', 'is_platoon_advantage']
        for i, col in enumerate(state_cols[:game_state.shape[1]]):
            analysis_df[col] = game_state[:, i]
        
        return analysis_df


def plot_training_history(training_history: Dict, output_dir: Path, config: Dict):
    """Plot training history and save figures."""
    # Configure matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    rcParams['figure.figsize'] = (12, 8)
    rcParams['font.size'] = 12
    rcParams['axes.titlesize'] = 16
    rcParams['axes.labelsize'] = 14
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'CQL Training History (Alpha={config["training"]["cql_alpha"]})', 
                fontsize=20, fontweight='bold')
    
    # Plot 1: Total Loss
    axes[0, 0].plot(training_history['train_loss'], 'b-', linewidth=2, label='Training')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: TD Loss
    axes[0, 1].plot(training_history['train_td_loss'], 'g-', linewidth=2, label='TD Loss')
    axes[0, 1].set_title('TD Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: CQL Loss
    axes[0, 2].plot(training_history['train_cql_loss'], 'r-', linewidth=2, label='CQL Loss')
    axes[0, 2].set_title('CQL Regularization Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Q-values
    axes[1, 0].plot(training_history['train_mean_q'], 'c-', linewidth=2, label='Training')
    axes[1, 0].plot(training_history['val_mean_q'], 'm-', linewidth=2, label='Validation')
    axes[1, 0].set_title('Mean Q-Values')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Q-value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Validation Accuracy
    axes[1, 1].plot(training_history['val_accuracy'], 'b-', linewidth=2, label='Overall')
    axes[1, 1].plot(training_history['val_stay_accuracy'], 'g-', linewidth=2, label='Stay')
    axes[1, 1].plot(training_history['val_pull_accuracy'], 'r-', linewidth=2, label='Pull')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Learning Rate
    axes[1, 2].plot(training_history['learning_rate'], 'purple', linewidth=2)
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to {output_dir / 'training_history.png'}")


def plot_decision_analysis(test_analysis: pd.DataFrame, output_dir: Path):
    """Plot decision analysis and save figures."""
    # Configure matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    rcParams['figure.figsize'] = (12, 8)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CQL Decision Analysis', fontsize=20, fontweight='bold')
    
    # Plot 1: Q-difference distribution
    axes[0, 0].hist(test_analysis['q_difference'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    axes[0, 0].set_title('Q-Difference Distribution (Reliever - Stay)')
    axes[0, 0].set_xlabel('Q-difference')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q-difference by prediction correctness
    correct_q_diff = test_analysis.loc[test_analysis['correct_prediction'], 'q_difference']
    incorrect_q_diff = test_analysis.loc[~test_analysis['correct_prediction'], 'q_difference']
    
    axes[0, 1].boxplot([correct_q_diff, incorrect_q_diff], 
                       labels=['Correct', 'Incorrect'], 
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='darkblue'),
                       medianprops=dict(color='red'))
    axes[0, 1].set_title('Q-Difference by Prediction Correctness')
    axes[0, 1].set_ylabel('Q-difference')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Accuracy by inning
    inning_stats = test_analysis.groupby(pd.cut(test_analysis['inning'], 
                                                bins=[0, 3, 6, 9, 15], 
                                                labels=['Early', 'Middle', 'Late', 'Extra']))\
                                 ['correct_prediction'].mean()
    inning_counts = test_analysis.groupby(pd.cut(test_analysis['inning'], 
                                                 bins=[0, 3, 6, 9, 15]))\
                                  .size()
    
    x = np.arange(len(inning_stats))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x, inning_stats.values, width, color='steelblue', alpha=0.7, label='Accuracy')
    axes[1, 0].set_xlabel('Inning')
    axes[1, 0].set_ylabel('Accuracy', color='steelblue')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(inning_stats.index)
    axes[1, 0].tick_params(axis='y', labelcolor='steelblue')
    
    axes2 = axes[1, 0].twinx()
    bars2 = axes2.bar(x + width, inning_counts.values, width, color='orange', alpha=0.7, label='Count')
    axes2.set_ylabel('Sample Count', color='orange')
    axes2.tick_params(axis='y', labelcolor='orange')
    
    axes[1, 0].legend()
    axes[1, 0].set_title('Accuracy by Inning')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Decision matrix
    decision_matrix = pd.crosstab(
        test_analysis['actual_stay'].map({True: 'Actual Stay', False: 'Actual Pull'}),
        test_analysis['predicted_stay'].map({True: 'Pred Stay', False: 'Pred Pull'}),
        normalize='index'
    )
    
    im = axes[1, 1].imshow(decision_matrix.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['Pred Stay', 'Pred Pull'])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Actual Stay', 'Actual Pull'])
    axes[1, 1].set_title('Decision Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[1, 1].text(j, i, f'{decision_matrix.values[i, j]:.2%}',
                                  ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decision_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Decision analysis plots saved to {output_dir / 'decision_analysis.png'}")


def main():
    """Main training and evaluation function."""
    parser = argparse.ArgumentParser(description='Train and evaluate CQL for MLB bullpen management')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for evaluation')
    parser.add_argument('--config-dir', type=str, default='config', help='Configuration directory')
    parser.add_argument('--data-path', type=str, default='data/processed/rl_tensors_2022_2023.npz', 
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = BullpenDataset(args.data_path, config)
    
    # Create data splits
    splits = dataset.get_split_indices(train_ratio=0.7, val_ratio=0.15)
    
    if args.train:
        # Initialize agent
        agent = CQLAgent(
            game_state_dim=dataset.game_state_dim,
            lineup_feat_dim=dataset.lineup_feat_dim,
            lineup_window=dataset.lineup_window,
            reliever_feat_dim=dataset.reliever_feat_dim,
            max_bullpen_size=dataset.max_bullpen_size,
            config=config
        )
        
        # Create data loaders
        train_loader = create_data_loader(
            dataset, splits['train'], 
            batch_size=config['training']['batch_size'], 
            shuffle=True
        )
        
        val_loader = create_data_loader(
            dataset, splits['val'], 
            batch_size=config['training']['batch_size'], 
            shuffle=False
        )
        
        # Initialize training tracking
        training_history = {
            'train_loss': [],
            'train_td_loss': [],
            'train_cql_loss': [],
            'train_mean_q': [],
            'val_accuracy': [],
            'val_stay_accuracy': [],
            'val_pull_accuracy': [],
            'val_mean_q': [],
            'learning_rate': []
        }
        
        # Training loop
        epochs = config['training']['epochs']
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = train_epoch(epoch, train_loader, agent, config['training']['log_interval'])
            
            # Validate
            val_metrics = validate(agent, val_loader)
            
            # Record metrics
            training_history['train_loss'].append(train_metrics['total_loss'])
            training_history['train_td_loss'].append(train_metrics['td_loss'])
            training_history['train_cql_loss'].append(train_metrics['cql_loss'])
            training_history['train_mean_q'].append(train_metrics['mean_q'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_stay_accuracy'].append(val_metrics['stay_accuracy'])
            training_history['val_pull_accuracy'].append(val_metrics['pull_accuracy'])
            training_history['val_mean_q'].append(val_metrics['mean_q'])
            training_history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Training Loss: {train_metrics['total_loss']:.4f} "
                  f"(TD: {train_metrics['td_loss']:.4f}, CQL: {train_metrics['cql_loss']:.4f})")
            print(f"  Training Mean Q: {train_metrics['mean_q']:.4f}")
            print(f"  Validation Accuracy: {val_metrics['accuracy']:.3f}")
            print(f"    - Stay: {val_metrics['stay_accuracy']:.3f}")
            print(f"    - Pull: {val_metrics['pull_accuracy']:.3f}")
            print(f"  Validation Mean Q: {val_metrics['mean_q']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % config['training']['save_interval'] == 0:
                checkpoint_path = output_dir / f"cql_epoch_{epoch+1}.pth"
                agent.save_checkpoint(str(checkpoint_path))
        
        # Save final model
        final_path = output_dir / "cql_final.pth"
        agent.save_checkpoint(str(final_path))
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in training_history.items()}, f, indent=2)
        
        # Plot training history
        plot_training_history(training_history, output_dir, config)
        
        print(f"\nTraining completed! Results saved to {output_dir}")
        
        # Set checkpoint for evaluation
        args.checkpoint = str(final_path)
        args.evaluate = True
    
    if args.evaluate and args.checkpoint:
        # Initialize agent for evaluation
        agent = CQLAgent(
            game_state_dim=dataset.game_state_dim,
            lineup_feat_dim=dataset.lineup_feat_dim,
            lineup_window=dataset.lineup_window,
            reliever_feat_dim=dataset.reliever_feat_dim,
            max_bullpen_size=dataset.max_bullpen_size,
            config=config
        )
        
        # Load checkpoint
        agent.load_checkpoint(args.checkpoint)
        
        # Create test data loader
        test_loader = create_data_loader(
            dataset, splits['test'], 
            batch_size=config['training']['batch_size'], 
            shuffle=False
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = validate(agent, test_loader)
        
        # Print evaluation results
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(f"Overall Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"Stay Decision Accuracy: {test_metrics['stay_accuracy']:.3f}")
        print(f"Pull Decision Accuracy: {test_metrics['pull_accuracy']:.3f}")
        print(f"Mean Q-value: {test_metrics['mean_q']:.4f}")
        print(f"Mean Stay Q-value: {test_metrics['mean_stay_q']:.4f}")
        print(f"Mean Pull Q-value: {test_metrics['mean_pull_q']:.4f}")
        print(f"Number of test samples: {len(splits['test']):,}")
        
        # Analyze decision patterns
        print("\nAnalyzing decision patterns on test set...")
        test_analysis = analyze_decision_patterns(agent, dataset, splits['test'], n_samples=10000)
        
        # Save analysis results
        analysis_path = output_dir / "decision_analysis.csv"
        test_analysis.to_csv(analysis_path, index=False)
        
        # Plot decision analysis
        plot_decision_analysis(test_analysis, output_dir)
        
        # Print summary
        print(f"\nAnalysis completed! Results saved to {output_dir}")
        print(f"  - Decision analysis: {analysis_path}")
        print(f"  - Decision plots: {output_dir / 'decision_analysis.png'}")


if __name__ == "__main__":
    main()