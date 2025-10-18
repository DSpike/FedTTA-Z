#!/usr/bin/env python3
"""
TTT Configuration Management Script
This script demonstrates how to easily configure TTT parameters without modifying the main code.
"""

from config import get_config, update_config

def configure_ttt_for_experiment(experiment_name: str):
    """
    Configure TTT parameters for different experiments
    
    Args:
        experiment_name: Name of the experiment ('fast', 'balanced', 'thorough', 'custom')
    """
    
    config = get_config()
    
    if experiment_name == "fast":
        # Quick TTT adaptation for testing
        update_config(
            ttt_base_steps=25,
            ttt_max_steps=50,
            ttt_lr=0.002,
            ttt_patience=5,
            ttt_timeout=15
        )
        print("🚀 Configured for FAST TTT adaptation:")
        print(f"   Base steps: {config.ttt_base_steps}")
        print(f"   Max steps: {config.ttt_max_steps}")
        print(f"   Learning rate: {config.ttt_lr}")
        print(f"   Patience: {config.ttt_patience}")
        print(f"   Timeout: {config.ttt_timeout}s")
        
    elif experiment_name == "balanced":
        # Balanced TTT adaptation (default)
        update_config(
            ttt_base_steps=50,
            ttt_max_steps=100,
            ttt_lr=0.001,
            ttt_patience=15,
            ttt_timeout=30
        )
        print("⚖️ Configured for BALANCED TTT adaptation:")
        print(f"   Base steps: {config.ttt_base_steps}")
        print(f"   Max steps: {config.ttt_max_steps}")
        print(f"   Learning rate: {config.ttt_lr}")
        print(f"   Patience: {config.ttt_patience}")
        print(f"   Timeout: {config.ttt_timeout}s")
        
    elif experiment_name == "thorough":
        # Thorough TTT adaptation for maximum performance
        update_config(
            ttt_base_steps=100,
            ttt_max_steps=200,
            ttt_lr=0.0005,
            ttt_patience=25,
            ttt_timeout=60
        )
        print("🔬 Configured for THOROUGH TTT adaptation:")
        print(f"   Base steps: {config.ttt_base_steps}")
        print(f"   Max steps: {config.ttt_max_steps}")
        print(f"   Learning rate: {config.ttt_lr}")
        print(f"   Patience: {config.ttt_patience}")
        print(f"   Timeout: {config.ttt_timeout}s")
        
    elif experiment_name == "custom":
        # Custom configuration
        print("🔧 Custom TTT Configuration:")
        print("Enter your desired TTT parameters:")
        
        try:
            base_steps = int(input("Base steps (default 50): ") or "50")
            max_steps = int(input("Max steps (default 200): ") or "200")
            lr = float(input("Learning rate (default 0.001): ") or "0.001")
            patience = int(input("Patience (default 15): ") or "15")
            timeout = int(input("Timeout in seconds (default 30): ") or "30")
            
            update_config(
                ttt_base_steps=base_steps,
                ttt_max_steps=max_steps,
                ttt_lr=lr,
                ttt_patience=patience,
                ttt_timeout=timeout
            )
            
            print("✅ Custom configuration applied!")
            print(f"   Base steps: {config.ttt_base_steps}")
            print(f"   Max steps: {config.ttt_max_steps}")
            print(f"   Learning rate: {config.ttt_lr}")
            print(f"   Patience: {config.ttt_patience}")
            print(f"   Timeout: {config.ttt_timeout}s")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            print("Using default configuration.")
            
    else:
        print(f"❌ Unknown experiment: {experiment_name}")
        print("Available experiments: 'fast', 'balanced', 'thorough', 'custom'")
        return False
        
    return True

def show_current_config():
    """Display current TTT configuration"""
    config = get_config()
    print("📊 Current TTT Configuration:")
    print(f"   Base steps: {config.ttt_base_steps}")
    print(f"   Max steps: {config.ttt_max_steps}")
    print(f"   Learning rate: {config.ttt_lr}")
    print(f"   Weight decay: {config.ttt_weight_decay}")
    print(f"   Patience: {config.ttt_patience}")
    print(f"   Timeout: {config.ttt_timeout}s")
    print(f"   Improvement threshold: {config.ttt_improvement_threshold}")

def main():
    """Main function to demonstrate TTT configuration"""
    print("🔧 TTT Configuration Management")
    print("=" * 50)
    
    # Show current configuration
    show_current_config()
    print()
    
    # Example configurations
    print("Available experiment configurations:")
    print("1. fast     - Quick adaptation (25-50 steps)")
    print("2. balanced - Balanced adaptation (50-100 steps)")  
    print("3. thorough - Thorough adaptation (100-200 steps)")
    print("4. custom   - Custom configuration")
    print()
    
    # Get user choice
    choice = input("Select experiment type (1-4) or 'show' to display current config: ").strip().lower()
    
    if choice == "1" or choice == "fast":
        configure_ttt_for_experiment("fast")
    elif choice == "2" or choice == "balanced":
        configure_ttt_for_experiment("balanced")
    elif choice == "3" or choice == "thorough":
        configure_ttt_for_experiment("thorough")
    elif choice == "4" or choice == "custom":
        configure_ttt_for_experiment("custom")
    elif choice == "show":
        show_current_config()
    else:
        print("❌ Invalid choice. Please select 1-4 or 'show'.")
        return
    
    print("\n✅ Configuration updated! You can now run the main system.")
    print("   The TTT parameters will be automatically used from the configuration.")

if __name__ == "__main__":
    main()
