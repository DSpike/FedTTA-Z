#!/usr/bin/env python3
"""
Test script to verify text visibility in confusion matrix plots
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend

def test_text_visibility():
    print("🔍 TESTING TEXT VISIBILITY IN CONFUSION MATRIX")
    print("=" * 50)
    
    # Create a simple confusion matrix with visible text
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Test confusion matrix
    cm = np.array([[428, 572], [394, 606]])
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.colorbar(im, ax=ax)
    
    # Set labels
    classes = ['Normal', 'Attack']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add text annotations on the matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    # Add performance metrics as text box - TEST DIFFERENT POSITIONS
    metrics_text = 'Accuracy: 0.517\nPrecision: 0.514\nRecall: 0.606\nF1-Score: 0.513\nMCC: 0.035'
    
    # Test different positions and styles
    positions = [
        (0.02, 0.98, 'top-left'),
        (0.5, 0.98, 'top-center'),
        (0.98, 0.98, 'top-right'),
        (0.02, 0.02, 'bottom-left'),
        (0.5, 0.5, 'center')
    ]
    
    for i, (x, y, pos_name) in enumerate(positions):
        ax.text(x, y, f'{pos_name}\n{metrics_text}', 
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top' if y > 0.5 else 'bottom',
               horizontalalignment='left' if x < 0.5 else 'right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_title('Text Visibility Test - Multiple Positions', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('text_visibility_test.png', dpi=300, bbox_inches='tight', 
                edgecolor='none', pad_inches=0.1)
    plt.close()
    
    print("✅ Text visibility test saved as 'text_visibility_test.png'")
    print("📊 Check the image to see if text boxes are visible in different positions")

def test_simple_confusion_matrix():
    print("\n🎨 CREATING SIMPLE CONFUSION MATRIX WITH CLEAR TEXT")
    print("=" * 50)
    
    # Create a simple confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Test confusion matrix
    cm = np.array([[428, 572], [394, 606]])
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    
    # Set labels
    classes = ['Normal', 'Attack']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add text annotations on the matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
    
    # Add performance metrics as text box - SIMPLE VERSION
    metrics_text = 'Accuracy: 0.517\nPrecision: 0.514\nRecall: 0.606\nF1-Score: 0.513\nMCC: 0.035'
    
    # Use a very visible text box
    ax.text(0.02, 0.98, metrics_text, 
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.9, edgecolor='black', linewidth=2))
    
    ax.set_title('Simple Confusion Matrix with Red Text Box', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('simple_confusion_matrix.png', dpi=300, bbox_inches='tight', 
                edgecolor='none', pad_inches=0.1)
    plt.close()
    
    print("✅ Simple confusion matrix saved as 'simple_confusion_matrix.png'")
    print("📊 This should have a very visible red text box with metrics")

if __name__ == "__main__":
    test_text_visibility()
    test_simple_confusion_matrix()



