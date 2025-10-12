#!/usr/bin/env python3
"""
Display Current Performance Plots
This script displays the most recent performance visualizations from the current DDoS_UDP run.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json
from datetime import datetime

def display_current_plots():
    """Display the current performance plots from the latest run"""
    
    # Get the performance_plots directory
    plots_dir = Path("performance_plots")
    
    if not plots_dir.exists():
        print("❌ Performance plots directory not found!")
        return
    
    print("🎨 Displaying Current Performance Plots (DDoS_UDP Run)")
    print("=" * 60)
    
    # Find the most recent files (from current run)
    current_plots = []
    
    # Look for files with "latest" in the name (these are from the current run)
    for file in plots_dir.glob("*latest*"):
        if file.is_file():
            # Get file modification time
            mod_time = datetime.fromtimestamp(file.stat().st_mtime)
            current_plots.append((file, mod_time))
    
    # Sort by modification time (most recent first)
    current_plots.sort(key=lambda x: x[1], reverse=True)
    
    # Display the most recent plots (from the current run)
    print(f"📅 Found {len(current_plots)} plot files")
    print(f"🕐 Most recent run: {current_plots[0][1].strftime('%Y-%m-%d %H:%M:%S') if current_plots else 'No files found'}")
    
    # Key plots to display from current run
    key_plots = [
        "roc_curves_latest.png",
        "performance_comparison_annotated_latest.png", 
        "confusion_matrices__base_model_latest.png",
        "confusion_matrices__ttt_enhanced_model_latest.png",
        "blockchain_metrics_latest.png",
        "gas_usage_analysis_latest.png",
        "client_performance_latest.png",
        "ttt_adaptation_latest.png"
    ]
    
    displayed_count = 0
    
    # Display each key plot
    for plot_file in key_plots:
        plot_path = plots_dir / plot_file
        if plot_path.exists():
            print(f"\n📊 Displaying: {plot_file}")
            try:
                # Load and display the image
                img = mpimg.imread(plot_path)
                plt.figure(figsize=(14, 10))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Performance Visualization: {plot_file.replace('_latest.png', '').replace('_', ' ').title()}", 
                         fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.show()
                print(f"✅ {plot_file} displayed successfully")
                displayed_count += 1
            except Exception as e:
                print(f"❌ Failed to display {plot_file}: {str(e)}")
        else:
            print(f"⚠️ {plot_file} not found")
    
    # Display performance metrics JSON
    metrics_file = plots_dir / "performance_metrics_latest.json"
    if metrics_file.exists():
        print(f"\n📈 Current Performance Metrics:")
        print("=" * 40)
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Display key metrics
            if 'base_model' in metrics:
                print(f"🔵 Base Model:")
                print(f"   Accuracy: {metrics['base_model'].get('accuracy', 'N/A'):.4f}")
                print(f"   F1-Score: {metrics['base_model'].get('f1_score', 'N/A'):.4f}")
                print(f"   ROC-AUC: {metrics['base_model'].get('roc_auc', 'N/A'):.4f}")
            
            if 'ttt_model' in metrics:
                print(f"🟢 TTT Model:")
                print(f"   Accuracy: {metrics['ttt_model'].get('accuracy', 'N/A'):.4f}")
                print(f"   F1-Score: {metrics['ttt_model'].get('f1_score', 'N/A'):.4f}")
                print(f"   ROC-AUC: {metrics['ttt_model'].get('roc_auc', 'N/A'):.4f}")
            
            if 'final_global_model' in metrics:
                print(f"🟡 Final Global Model:")
                print(f"   Accuracy: {metrics['final_global_model'].get('accuracy', 'N/A'):.4f}")
                print(f"   F1-Score: {metrics['final_global_model'].get('f1_score', 'N/A'):.4f}")
                print(f"   ROC-AUC: {metrics['final_global_model'].get('roc_auc', 'N/A'):.4f}")
                print(f"   Optimal Threshold: {metrics['final_global_model'].get('optimal_threshold', 'N/A'):.4f}")
                
        except Exception as e:
            print(f"❌ Failed to read metrics: {str(e)}")
    
    print(f"\n🎉 Displayed {displayed_count} plots from current DDoS_UDP run!")
    print(f"📁 Plot files are saved in: {plots_dir.absolute()}")
    
    # Show summary of what was displayed
    print(f"\n📋 Summary of Current Run:")
    print(f"   • Attack Type: DDoS_UDP")
    print(f"   • Plots Generated: {displayed_count}")
    print(f"   • Run Time: {current_plots[0][1].strftime('%Y-%m-%d %H:%M:%S') if current_plots else 'Unknown'}")

if __name__ == "__main__":
    display_current_plots()
