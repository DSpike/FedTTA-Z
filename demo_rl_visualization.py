#!/usr/bin/env python3
"""
Demo script for RL Agent Activity Visualization
Shows how the RL agent exploration and exploitation activity is visualized
following IEEE paper standards
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.performance_visualization import PerformanceVisualizer, extract_rl_agent_data
from models.transductive_fewshot_model import ThresholdAgent
import torch

def create_demo_rl_data():
    """Create realistic demo RL agent data for visualization"""
    
    # Create a threshold agent
    agent = ThresholdAgent(state_dim=2, hidden_dim=32, learning_rate=0.001)
    
    # Simulate training episodes
    num_episodes = 100
    
    for episode in range(num_episodes):
        # Generate realistic state
        mean_confidence = 0.5 + 0.3 * np.sin(episode * 0.1) + np.random.normal(0, 0.1)
        adaptation_success_rate = 0.3 + 0.4 * (episode / num_episodes) + np.random.normal(0, 0.05)
        
        # Clamp values to valid ranges
        mean_confidence = max(0.1, min(0.9, mean_confidence))
        adaptation_success_rate = max(0.1, min(0.9, adaptation_success_rate))
        
        # Create state tensor
        state = torch.tensor([mean_confidence, adaptation_success_rate], dtype=torch.float32)
        
        # Get threshold (this will track action selection)
        threshold = agent.get_threshold(state)
        
        # Simulate adaptation results
        adaptation_success = np.random.random() < adaptation_success_rate
        accuracy_improvement = np.random.normal(0.1, 0.05) if adaptation_success else np.random.normal(-0.05, 0.03)
        
        # Simulate additional metrics
        false_positives = np.random.randint(0, 5)
        false_negatives = np.random.randint(0, 5)
        true_positives = np.random.randint(10, 20)
        true_negatives = np.random.randint(10, 20)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        samples_selected = np.random.randint(5, 15)
        total_samples = 20
        
        # Update agent
        agent.update(
            state, threshold, adaptation_success_rate, accuracy_improvement,
            false_positives, false_negatives, true_positives, true_negatives,
            precision, recall, f1_score, samples_selected, total_samples
        )
    
    return agent

def main():
    """Main demo function"""
    print("🤖 RL Agent Activity Visualization Demo")
    print("=" * 50)
    
    # Create demo RL agent with realistic data
    print("Creating demo RL agent with simulated training data...")
    agent = create_demo_rl_data()
    
    # Extract RL data for visualization
    print("Extracting RL agent data...")
    rl_data = extract_rl_agent_data(agent)
    
    print(f"Extracted data:")
    print(f"  - Threshold history: {len(rl_data.get('threshold_history', []))} steps")
    print(f"  - Epsilon history: {len(rl_data.get('epsilon_history', []))} steps")
    print(f"  - Reward history: {len(rl_data.get('reward_history', []))} steps")
    print(f"  - Action history: {len(rl_data.get('action_history', []))} steps")
    print(f"  - State history: {len(rl_data.get('state_history', []))} steps")
    print(f"  - Adaptation success: {len(rl_data.get('adaptation_success_history', []))} steps")
    
    # Create visualizer
    print("\nCreating performance visualizer...")
    visualizer = PerformanceVisualizer(output_dir="rl_demo_plots", attack_name="Demo")
    
    # Generate RL agent activity visualization
    print("Generating RL agent activity visualization...")
    activity_plot = visualizer.plot_rl_agent_activity(rl_data, save=True)
    if activity_plot:
        print(f"✅ RL agent activity plot saved: {activity_plot}")
    
    # Generate exploration vs exploitation analysis
    print("Generating exploration vs exploitation analysis...")
    exploration_plot = visualizer.plot_rl_exploration_exploitation_analysis(rl_data, save=True)
    if exploration_plot:
        print(f"✅ Exploration vs exploitation analysis saved: {exploration_plot}")
    
    # Create comprehensive system data for full report
    print("Creating comprehensive system data...")
    system_data = {
        'training_history': {
            'epoch_losses': [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08],
            'epoch_accuracies': [0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92]
        },
        'evaluation_results': {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.91,
            'f1_score': 0.90,
            'zero_day_detection_rate': 0.15
        },
        'client_results': [
            {'client_id': 'client_1', 'accuracy': 0.88, 'f1_score': 0.85, 'precision': 0.87, 'recall': 0.83},
            {'client_id': 'client_2', 'accuracy': 0.91, 'f1_score': 0.89, 'precision': 0.90, 'recall': 0.88},
            {'client_id': 'client_3', 'accuracy': 0.89, 'f1_score': 0.87, 'precision': 0.88, 'recall': 0.86}
        ],
        'blockchain_data': {
            'transactions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'ipfs_cids': ['Qm1', 'Qm2', 'Qm3', 'Qm4', 'Qm5'],
            'gas_used': [21000, 25000, 23000, 24000, 22000, 26000, 28000, 27000, 29000, 30000]
        },
        'rl_agent_data': rl_data
    }
    
    # Generate comprehensive report with RL data
    print("Generating comprehensive report with RL agent analysis...")
    comprehensive_plot = visualizer.create_comprehensive_report(system_data, save=True)
    if comprehensive_plot:
        print(f"✅ Comprehensive report saved: {comprehensive_plot}")
    
    print("\n🎉 RL Agent Visualization Demo Completed!")
    print("=" * 50)
    print("Generated visualizations:")
    print("1. RL Agent Activity Analysis - Shows exploration vs exploitation, epsilon decay, threshold evolution, etc.")
    print("2. Exploration vs Exploitation Analysis - Detailed analysis of RL learning behavior")
    print("3. Comprehensive Performance Report - Complete system overview including RL agent performance")
    print("\nThese visualizations follow IEEE paper standards and show:")
    print("• Exploration vs exploitation patterns over time")
    print("• Epsilon decay and learning progress")
    print("• Dynamic threshold evolution")
    print("• Reward learning curves with moving averages")
    print("• State space exploration patterns")
    print("• Policy visualization and learning efficiency")
    print("• Statistical analysis of RL agent performance")

if __name__ == "__main__":
    main()
