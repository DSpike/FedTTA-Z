#!/usr/bin/env python3
"""
Extract and analyze scatter plot statistics from TTT adaptation data
"""

import json
import numpy as np
import pickle
import os
from pathlib import Path

def analyze_scatter_plot_stats():
    """Extract scatter plot statistics from the last run"""
    
    # Try to load from performance metrics JSON
    metrics_file = "performance_plots/performance_metrics_.json"
    
    # Try to load from system state
    state_file = "enhanced_blockchain_federated_system_state.json"
    
    ttt_data = None
    
    # Method 1: Try loading from state file
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                if 'ttt_adaptation_data' in state:
                    ttt_data = state['ttt_adaptation_data']
                    print("✅ Loaded TTT data from state file")
        except Exception as e:
            print(f"⚠️ Could not load from state file: {e}")
    
    # Method 2: Try to extract from main.py's stored data
    # We need to check if the system stored the data
    if ttt_data is None:
        print("⚠️ TTT adaptation data not found in state file")
        print("   This means we need to extract it from the plot generation or run the system again")
        print("   Let me check if we can calculate from the stored model...")
        return
    
    # Extract attack_vs_normal_data
    attack_vs_normal_data = ttt_data.get('attack_vs_normal_data', [])
    
    if not attack_vs_normal_data:
        print("❌ No attack_vs_normal_data found in TTT adaptation data")
        return
    
    print("\n" + "="*80)
    print("📊 TTT ATTACK VS NORMAL SEPARATION ANALYSIS")
    print("="*80)
    
    # Get beginning and end data
    beginning_data = attack_vs_normal_data[0]
    end_data = attack_vs_normal_data[-1]
    
    print(f"\n📈 Data Points Collected: {len(attack_vs_normal_data)} steps")
    print(f"   Beginning Step: {beginning_data.get('step', 'N/A')}")
    print(f"   End Step: {end_data.get('step', 'N/A')}")
    
    # Analyze beginning (Step 1)
    if 'attack_probs' in beginning_data and 'binary_labels' in beginning_data:
        attack_probs_begin = np.array(beginning_data['attack_probs'])
        binary_labels_begin = np.array(beginning_data['binary_labels'])
        
        normal_mask_begin = binary_labels_begin == 0
        attack_mask_begin = binary_labels_begin == 1
        
        normal_probs_begin = attack_probs_begin[normal_mask_begin] if normal_mask_begin.sum() > 0 else np.array([])
        attack_probs_only_begin = attack_probs_begin[attack_mask_begin] if attack_mask_begin.sum() > 0 else np.array([])
        
        if len(normal_probs_begin) > 0 and len(attack_probs_only_begin) > 0:
            separation_begin = attack_probs_only_begin.mean() - normal_probs_begin.mean()
            
            print(f"\n🔵 BEGINNING OF TTT ADAPTATION (Step {beginning_data.get('step', 'N/A')}):")
            print(f"   Normal Samples: {normal_mask_begin.sum()}")
            print(f"   Attack Samples: {attack_mask_begin.sum()}")
            print(f"   Normal Mean Attack Probability: {normal_probs_begin.mean():.4f}")
            print(f"   Attack Mean Attack Probability: {attack_probs_only_begin.mean():.4f}")
            print(f"   Separation (Attack Mean - Normal Mean): {separation_begin:.4f}")
            print(f"   Normal Std: {normal_probs_begin.std():.4f}")
            print(f"   Attack Std: {attack_probs_only_begin.std():.4f}")
    
    # Analyze end (Final Step)
    if 'attack_probs' in end_data and 'binary_labels' in end_data:
        attack_probs_end = np.array(end_data['attack_probs'])
        binary_labels_end = np.array(end_data['binary_labels'])
        
        normal_mask_end = binary_labels_end == 0
        attack_mask_end = binary_labels_end == 1
        
        normal_probs_end = attack_probs_end[normal_mask_end] if normal_mask_end.sum() > 0 else np.array([])
        attack_probs_only_end = attack_probs_end[attack_mask_end] if attack_mask_end.sum() > 0 else np.array([])
        
        if len(normal_probs_end) > 0 and len(attack_probs_only_end) > 0:
            separation_end = attack_probs_only_end.mean() - normal_probs_end.mean()
            improvement = separation_end - separation_begin if 'separation_begin' in locals() else 0.0
            
            print(f"\n🟢 END OF TTT ADAPTATION (Step {end_data.get('step', 'N/A')}):")
            print(f"   Normal Samples: {normal_mask_end.sum()}")
            print(f"   Attack Samples: {attack_mask_end.sum()}")
            print(f"   Normal Mean Attack Probability: {normal_probs_end.mean():.4f}")
            print(f"   Attack Mean Attack Probability: {attack_probs_only_end.mean():.4f}")
            print(f"   Separation (Attack Mean - Normal Mean): {separation_end:.4f}")
            print(f"   Normal Std: {normal_probs_end.std():.4f}")
            print(f"   Attack Std: {attack_probs_only_end.std():.4f}")
            
            if 'separation_begin' in locals():
                print(f"\n📊 IMPROVEMENT DURING TTT ADAPTATION:")
                print(f"   Separation Improvement: {improvement:+.4f}")
                print(f"   Improvement Percentage: {(improvement/separation_begin*100):+.2f}%")
            
            # Interpretation
            print(f"\n🎯 INTERPRETATION:")
            if separation_end > 0.5:
                print(f"   ✅ EXCELLENT: Large separation ({separation_end:.3f}) - TTT successfully distinguishes attacks from normal")
            elif separation_end > 0.3:
                print(f"   ✅ GOOD: Moderate separation ({separation_end:.3f}) - TTT provides reasonable separation")
            elif separation_end > 0.1:
                print(f"   ⚠️  MODERATE: Small separation ({separation_end:.3f}) - TTT provides some separation but could be better")
            else:
                print(f"   ❌ POOR: Very small separation ({separation_end:.3f}) - TTT struggles to distinguish attacks from normal")
            
            if improvement > 0.05:
                print(f"   ✅ TTT ADAPTATION IMPROVED separation by {improvement:.3f} - Adaptation was successful!")
            elif improvement > 0:
                print(f"   ⚠️  TTT ADAPTATION slightly improved separation by {improvement:.3f} - Marginal improvement")
            elif improvement == 0:
                print(f"   ⚠️  TTT ADAPTATION did not change separation - No improvement")
            else:
                print(f"   ❌ TTT ADAPTATION decreased separation by {abs(improvement):.3f} - Adaptation may have hurt performance")
            
            # Check if attacks are clearly separated (high attack prob, low normal prob)
            if attack_probs_only_end.mean() > 0.7 and normal_probs_end.mean() < 0.3:
                print(f"   ✅ CLEAR SEPARATION: Attack samples have high probability ({attack_probs_only_end.mean():.3f})")
                print(f"      while normal samples have low probability ({normal_probs_end.mean():.3f})")
            elif attack_probs_only_end.mean() > normal_probs_end.mean():
                print(f"   ⚠️  PARTIAL SEPARATION: Attack samples ({attack_probs_only_end.mean():.3f}) > Normal samples ({normal_probs_end.mean():.3f})")
                print(f"      but the gap could be larger for better detection")
            else:
                print(f"   ❌ POOR SEPARATION: Attack samples ({attack_probs_only_end.mean():.3f}) <= Normal samples ({normal_probs_end.mean():.3f})")
                print(f"      TTT failed to properly separate attacks from normal samples")

if __name__ == "__main__":
    analyze_scatter_plot_stats()



