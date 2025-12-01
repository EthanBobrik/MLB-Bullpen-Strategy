import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from cql_model import CQLAgent
from train_cql import BullpenDataLoader, load_config

class CQLEvaluator:
    """
    Evaluate CQL agent on bullpen decision making.
    """
    
    def __init__(self, config: Dict, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Load data
        data_path = Path(config['data']['processed_data_dir']) / config['data']['dataset_file']
        self.data_loader = BullpenDataLoader(str(data_path), device=device)
        
        # Create agent
        self.agent = CQLAgent(
            game_state_dim=self.data_loader.game_state_dim,
            lineup_feat_dim=self.data_loader.lineup_feat_dim,
            lineup_window=self.data_loader.lineup_window,
            reliever_feat_dim=self.data_loader.reliever_feat_dim,
            max_bullpen_size=self.data_loader.max_bullpen_size,
            hidden_dim=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            device=device
        )
    
    def load_model(self, checkpoint_path: str):
        """Load trained model."""
        self.agent.load_checkpoint(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
    
    def evaluate_policy(self, num_samples: int = 10000) -> Dict:
        """
        Evaluate the policy on a subset of data.
        """
        # Sample random indices
        total_samples = len(self.data_loader.data['state_vec'])
        eval_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        
        batch = self.data_loader.get_batch(eval_indices)
        
        with torch.no_grad():
            # Get Q-values for all actions
            q_values = self.agent.q_network(
                batch['game_state'],
                batch['lineup_feats'],
                batch['pos_enc'],
                batch['reliever_feats'],
                batch['avail_mask']
            )
            
            # Get predicted actions
            predicted_actions = q_values.argmax(dim=1).cpu().numpy()
            actual_actions = batch['action'].cpu().numpy()
            
            # Compute metrics
            accuracy = (predicted_actions == actual_actions).mean()
            stay_accuracy = (predicted_actions[actual_actions == 0] == 0).mean() if (actual_actions == 0).sum() > 0 else 0
            pull_accuracy = (predicted_actions[actual_actions > 0] == actual_actions[actual_actions > 0]).mean() if (actual_actions > 0).sum() > 0 else 0
            
            # Action distribution
            pred_action_dist = np.bincount(predicted_actions, minlength=self.agent.num_actions)
            actual_action_dist = np.bincount(actual_actions, minlength=self.agent.num_actions)
            
            # Q-value statistics
            mean_q = q_values.mean().item()
            max_q = q_values.max().item()
            min_q = q_values.min().item()
        
        return {
            'accuracy': accuracy,
            'stay_accuracy': stay_accuracy,
            'pull_accuracy': pull_accuracy,
            'predicted_action_dist': pred_action_dist,
            'actual_action_dist': actual_action_dist,
            'mean_q_value': mean_q,
            'max_q_value': max_q,
            'min_q_value': min_q,
            'num_samples': len(eval_indices)
        }
    
    def analyze_decisions(self, game_pk: int = None) -> pd.DataFrame:
        """
        Analyze decisions for a specific game or all games.
        """
        if game_pk:
            # Filter for specific game
            game_indices = np.where(self.data_loader.data['game_pk'] == game_pk)[0]
            if len(game_indices) == 0:
                print(f"No data found for game {game_pk}")
                return pd.DataFrame()
            batch = self.data_loader.get_batch(game_indices)
        else:
            # Use all data
            batch = self.data_loader.get_batch(np.arange(len(self.data_loader.data['state_vec'])))
        
        with torch.no_grad():
            q_values = self.agent.q_network(
                batch['game_state'],
                batch['lineup_feats'],
                batch['pos_enc'],
                batch['reliever_feats'],
                batch['avail_mask']
            )
            
            predicted_actions = q_values.argmax(dim=1).cpu().numpy()
            stay_q_values = q_values[:, 0].cpu().numpy()
            best_reliever_q = q_values[:, 1:].max(dim=1)[0].cpu().numpy()
            q_diff = best_reliever_q - stay_q_values
        
        # Create results DataFrame
        results = pd.DataFrame({
            'predicted_action': predicted_actions,
            'actual_action': batch['action'].cpu().numpy(),
            'stay_q_value': stay_q_values,
            'best_reliever_q': best_reliever_q,
            'q_difference': q_diff,
            'reward': batch['reward'].cpu().numpy(),
            'correct_prediction': predicted_actions == batch['action'].cpu().numpy()
        })
        
        return results
    
    def print_evaluation_summary(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "="*50)
        print("CQL EVALUATION SUMMARY")
        print("="*50)
        print(f"Overall Accuracy: {results['accuracy']:.3f}")
        print(f"Stay Decision Accuracy: {results['stay_accuracy']:.3f}")
        print(f"Pull Decision Accuracy: {results['pull_accuracy']:.3f}")
        print(f"Mean Q-value: {results['mean_q_value']:.4f}")
        print(f"Q-value Range: [{results['min_q_value']:.4f}, {results['max_q_value']:.4f}]")
        print(f"Number of Samples: {results['num_samples']}")
        
        print("\nAction Distribution:")
        print("Action | Predicted | Actual")
        print("-" * 25)
        for action in range(self.agent.num_actions):
            pred_pct = results['predicted_action_dist'][action] / results['num_samples']
            actual_pct = results['actual_action_dist'][action] / results['num_samples']
            action_name = "STAY" if action == 0 else f"REL{action}"
            print(f"{action_name:6} | {pred_pct:8.3f} | {actual_pct:6.3f}")


def main():
    """Main evaluation function."""
    config = load_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = CQLEvaluator(config, device=device)
    
    # Load trained model
    checkpoint_path = config['inference']['load_checkpoint']
    evaluator.load_model(checkpoint_path)
    
    # Evaluate policy
    print("Evaluating CQL policy...")
    results = evaluator.evaluate_policy(num_samples=20000)
    evaluator.print_evaluation_summary(results)
    
    # Analyze decisions
    print("\nAnalyzing decision patterns...")
    decision_analysis = evaluator.analyze_decisions()
    
    if not decision_analysis.empty:
        print(f"\nDecision Analysis Summary:")
        print(f"Total decisions: {len(decision_analysis)}")
        print(f"Average Q-difference: {decision_analysis['q_difference'].mean():.4f}")
        print(f"Stay decisions: {(decision_analysis['predicted_action'] == 0).sum()}")
        print(f"Pull decisions: {(decision_analysis['predicted_action'] > 0).sum()}")
        
        # Save analysis to CSV
        output_path = Path("evaluation_results") / "decision_analysis.csv"
        output_path.parent.mkdir(exist_ok=True)
        decision_analysis.to_csv(output_path, index=False)
        print(f"Detailed analysis saved to {output_path}")


if __name__ == "__main__":
    main()