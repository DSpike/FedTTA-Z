#!/usr/bin/env python3
"""
Configuration Management Script
Easy way to modify system configuration without editing code
"""

from config import get_config, update_config, reset_config
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Configure the Blockchain Federated Learning System')
    
    # Add configuration options
    parser.add_argument('--rounds', type=int, help='Number of training rounds')
    parser.add_argument('--clients', type=int, help='Number of clients')
    parser.add_argument('--attack', type=str, help='Zero-day attack type (DoS, Exploits, Analysis, etc.)')
    parser.add_argument('--use-tcn', action='store_true', help='Enable TCN model')
    parser.add_argument('--no-tcn', action='store_true', help='Disable TCN model')
    parser.add_argument('--sequence-length', type=int, help='TCN sequence length')
    parser.add_argument('--sequence-stride', type=int, help='TCN sequence stride')
    parser.add_argument('--epochs', type=int, help='Local training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    
    args = parser.parse_args()
    
    if args.reset:
        reset_config()
        print("✅ Configuration reset to defaults")
        return
    
    if args.show:
        config = get_config()
        print("🔧 Current Configuration:")
        for key, value in config.to_dict().items():
            print(f"   {key}: {value}")
        return
    
    # Update configuration based on arguments
    updates = {}
    
    if args.rounds is not None:
        updates['num_rounds'] = args.rounds
        print(f"✅ Set rounds to {args.rounds}")
    
    if args.clients is not None:
        updates['num_clients'] = args.clients
        print(f"✅ Set clients to {args.clients}")
    
    if args.attack is not None:
        updates['zero_day_attack'] = args.attack
        print(f"✅ Set attack type to {args.attack}")
    
    if args.use_tcn:
        updates['use_tcn'] = True
        print("✅ Enabled TCN model")
    
    if args.no_tcn:
        updates['use_tcn'] = False
        print("✅ Disabled TCN model")
    
    if args.sequence_length is not None:
        updates['sequence_length'] = args.sequence_length
        print(f"✅ Set sequence length to {args.sequence_length}")
    
    if args.sequence_stride is not None:
        updates['sequence_stride'] = args.sequence_stride
        print(f"✅ Set sequence stride to {args.sequence_stride}")
    
    if args.epochs is not None:
        updates['local_epochs'] = args.epochs
        print(f"✅ Set local epochs to {args.epochs}")
    
    if args.learning_rate is not None:
        updates['learning_rate'] = args.learning_rate
        print(f"✅ Set learning rate to {args.learning_rate}")
    
    if updates:
        update_config(**updates)
        print("\n🎉 Configuration updated successfully!")
    else:
        print("❌ No configuration changes specified")
        print("Use --help to see available options")


if __name__ == "__main__":
    main()


