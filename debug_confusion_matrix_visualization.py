#!/usr/bin/env python3
"""
Debug script to test confusion matrix visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def debug_confusion_matrix():
    print("🔍 DEBUGGING CONFUSION MATRIX VISUALIZATION")
    print("=" * 50)
    
    # Load the performance metrics
    with open('performance_plots/performance_metrics_latest.json', 'r') as f:
        data = json.load(f)
    
    # Get evaluation results
    evaluation_results = data.get('evaluation_results', {})
    print(f"📊 Evaluation results keys: {list(evaluation_results.keys())}")
    
    # Test base model confusion matrix
    if 'base_model' in evaluation_results:
        base_model = evaluation_results['base_model']
        print(f"\n📊 Base Model keys: {list(base_model.keys())}")
        
        confusion_data = base_model.get('confusion_matrix', {})
        print(f"📊 Base confusion matrix data: {confusion_data}")
        print(f"📊 Base confusion matrix type: {type(confusion_data)}")
        
        # Convert to matrix format
        if isinstance(confusion_data, dict):
            tn = confusion_data.get('tn', 0)
            fp = confusion_data.get('fp', 0)
            fn = confusion_data.get('fn', 0)
            tp = confusion_data.get('tp', 0)
            cm = np.array([[tn, fp], [fn, tp]])
            print(f"📊 Converted base matrix:\n{cm}")
            
            # Test metrics extraction
            accuracy = base_model.get('accuracy', 0)
            precision = base_model.get('precision', 0)
            recall = base_model.get('recall', 0)
            f1_score = base_model.get('f1_score', 0)
            mcc = base_model.get('mccc', 0)
            
            print(f"📊 Base metrics: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1_score:.3f}, MCC={mcc:.3f}")
            
            # Test text generation
            metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}\nMCC: {mcc:.3f}'
            print(f"📊 Generated text:\n{metrics_text}")
    
    # Test TTT model confusion matrix
    if 'ttt_model' in evaluation_results:
        ttt_model = evaluation_results['ttt_model']
        print(f"\n📊 TTT Model keys: {list(ttt_model.keys())}")
        
        confusion_data = ttt_model.get('confusion_matrix', {})
        print(f"📊 TTT confusion matrix data: {confusion_data}")
        print(f"📊 TTT confusion matrix type: {type(confusion_data)}")
        
        # Convert to matrix format
        if isinstance(confusion_data, dict):
            tn = confusion_data.get('tn', 0)
            fp = confusion_data.get('fp', 0)
            fn = confusion_data.get('fn', 0)
            tp = confusion_data.get('tp', 0)
            cm = np.array([[tn, fp], [fn, tp]])
            print(f"📊 Converted TTT matrix:\n{cm}")
            
            # Test metrics extraction
            accuracy = ttt_model.get('accuracy', 0)
            precision = ttt_model.get('precision', 0)
            recall = ttt_model.get('recall', 0)
            f1_score = ttt_model.get('f1_score', 0)
            mcc = ttt_model.get('mccc', 0)
            
            print(f"📊 TTT metrics: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1_score:.3f}, MCC={mcc:.3f}")
            
            # Test text generation
            metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}\nMCC: {mcc:.3f}'
            print(f"📊 Generated text:\n{metrics_text}")

def create_test_confusion_matrix():
    print("\n🎨 CREATING TEST CONFUSION MATRIX")
    print("=" * 50)
    
    # Create a simple test confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Test data
    base_cm = np.array([[428, 572], [394, 606]])
    ttt_cm = np.array([[195, 34], [189, 62]])
    
    models = ['Base Model', 'TTT Model']
    confusion_matrices = [base_cm, ttt_cm]
    metrics = [
        {'accuracy': 0.517, 'precision': 0.515, 'recall': 0.606, 'f1_score': 0.513, 'mcc': 0.035},
        {'accuracy': 0.535, 'precision': 0.646, 'recall': 0.247, 'f1_score': 0.357, 'mcc': 0.123}
    ]
    
    for idx, (title, cm, metric) in enumerate(zip(models, confusion_matrices, metrics)):
        ax = axes[idx]
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        classes = ['Normal', 'Attack']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')
        
        # Add performance metrics as text
        metrics_text = f'Accuracy: {metric["accuracy"]:.3f}\nPrecision: {metric["precision"]:.3f}\nRecall: {metric["recall"]:.3f}\nF1-Score: {metric["f1_score"]:.3f}\nMCC: {metric["mcc"]:.3f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(f'{title} Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Set overall title
    fig.suptitle('Test Confusion Matrix Comparison', fontsize=16, fontweight='bold')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight', 
                edgecolor='none', pad_inches=0.1)
    plt.close()
    
    print("✅ Test confusion matrix saved as 'test_confusion_matrix.png'")

if __name__ == "__main__":
    debug_confusion_matrix()
    create_test_confusion_matrix()



