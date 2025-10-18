#!/usr/bin/env python3
"""
Diagnostic script to check the actual confusion matrix plots
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def analyze_confusion_matrix_image(file_path):
    """Analyze a confusion matrix image to check for text content"""
    print(f"🔍 ANALYZING: {file_path}")
    print("-" * 40)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Load the image
    try:
        img = mpimg.imread(file_path)
        print(f"✅ Image loaded: shape {img.shape}, dtype {img.dtype}")
        print(f"✅ Value range: {img.min():.3f} to {img.max():.3f}")
        
        # Check if image has any non-white pixels (indicating content)
        if len(img.shape) == 3:  # RGB or RGBA
            # Convert to grayscale for analysis
            gray = np.mean(img, axis=2)
        else:
            gray = img
        
        # Check for non-white pixels
        non_white_pixels = np.sum(gray < 0.95)  # Assuming white is close to 1.0
        total_pixels = gray.size
        content_ratio = non_white_pixels / total_pixels
        
        print(f"📊 Non-white pixels: {non_white_pixels}/{total_pixels} ({content_ratio:.2%})")
        
        if content_ratio < 0.01:
            print("⚠️  WARNING: Very few non-white pixels - image might be mostly empty")
        else:
            print("✅ Image appears to have content")
            
        # Check for text-like patterns (high contrast areas)
        edges = np.abs(np.diff(gray, axis=1))
        high_contrast_pixels = np.sum(edges > 0.1)
        print(f"📊 High contrast pixels: {high_contrast_pixels} (potential text)")
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")

def create_high_contrast_test():
    """Create a high-contrast test confusion matrix"""
    print("\n🎨 CREATING HIGH-CONTRAST TEST CONFUSION MATRIX")
    print("=" * 50)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # High contrast confusion matrix
    cm = np.array([[428, 572], [394, 606]])
    
    # Use high contrast colormap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds, vmin=0, vmax=cm.max())
    plt.colorbar(im, ax=ax)
    
    # Set labels with high contrast
    classes = ['Normal', 'Attack']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=14, fontweight='bold')
    ax.set_yticklabels(classes, fontsize=14, fontweight='bold')
    
    # Add text annotations with high contrast
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20, fontweight='bold')
    
    # Add performance metrics with very high contrast
    metrics_text = 'Accuracy: 0.517\nPrecision: 0.514\nRecall: 0.606\nF1-Score: 0.513\nMCC: 0.035'
    
    # Use black text on white background with thick border
    ax.text(0.02, 0.98, metrics_text, 
           transform=ax.transAxes, fontsize=14, fontweight='bold',
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=1.0, 
                    edgecolor='black', linewidth=3))
    
    ax.set_title('High Contrast Confusion Matrix Test', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Make sure all text is visible
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Save with high DPI
    plt.tight_layout()
    plt.savefig('high_contrast_confusion_matrix.png', dpi=300, bbox_inches='tight', 
                edgecolor='black', pad_inches=0.2)
    plt.close()
    
    print("✅ High contrast confusion matrix saved as 'high_contrast_confusion_matrix.png'")

if __name__ == "__main__":
    # Analyze existing confusion matrix files
    files_to_analyze = [
        'performance_plots/confusion_matrices__base_model_latest.png',
        'performance_plots/confusion_matrices__ttt_enhanced_model_latest.png',
        'performance_plots/confusion_matrices_latest.png',
        'test_confusion_matrix.png',
        'simple_confusion_matrix.png',
        'text_visibility_test.png'
    ]
    
    for file_path in files_to_analyze:
        analyze_confusion_matrix_image(file_path)
        print()
    
    # Create high contrast test
    create_high_contrast_test()



