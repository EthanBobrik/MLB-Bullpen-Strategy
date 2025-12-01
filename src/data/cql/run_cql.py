"""
Main script to run CQL training for MLB bullpen management.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from train_cql import train_cql
from evaluate_cql import main as evaluate_main

def main():
    parser = argparse.ArgumentParser(description='CQL Training for MLB Bullpen Management')
    parser.add_argument('--train', action='store_true', help='Train the CQL model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    parser.add_argument('--config-dir', type=str, default='config', help='Configuration directory')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for evaluation')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting CQL training...")
        train_cql()
    
    if args.evaluate:
        if not args.checkpoint:
            print("Please provide checkpoint path with --checkpoint for evaluation")
            return
        print("Starting CQL evaluation...")
        evaluate_main()

if __name__ == "__main__":
    main()