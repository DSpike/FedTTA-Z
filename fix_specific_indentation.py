#!/usr/bin/env python3
"""
Targeted Indentation Fixer for main.py
Fixes specific known indentation issues.
"""

import re

def fix_specific_indentation():
    """
    Fix specific indentation issues in main.py
    """
    print("🔧 Fixing specific indentation issues...")
    
    # Read the file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Lines after try: that should be indented
    # Pattern: try: followed by unindented create_sequences calls
    pattern1 = r'(\s*try:\s*\n\s*#.*\n\s*if.*\n\s*torch\.cuda\.empty_cache\(\)\n\s*\)\n\s*)\n(\s*X_.*_seq.*=.*create_sequences\()'
    replacement1 = r'\1\n                    \2'
    content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
    
    # Fix 2: Lines after else: that should be indented
    pattern2 = r'(\s*else:\s*\n\s*#.*\n)\n(\s*[^#\s].*)'
    replacement2 = r'\1\n                \2'
    content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)
    
    # Fix 3: Lines after if statements that should be indented
    pattern3 = r'(\s*if.*:\s*\n)\n(\s*[^#\s].*)'
    replacement3 = r'\1\n                        \2'
    content = re.sub(pattern3, replacement3, content, flags=re.MULTILINE)
    
    # Manual fixes for specific known issues
    fixes = [
        # Fix training sequence creation
        ('                X_train_seq, y_train_seq = self.preprocessor.create_sequences(', 
         '                    X_train_seq, y_train_seq = self.preprocessor.create_sequences('),
        
        # Fix validation sequence creation  
        ('                X_val_seq, y_val_seq = self.preprocessor.create_sequences(',
         '                    X_val_seq, y_val_seq = self.preprocessor.create_sequences('),
        
        # Fix test sequence creation
        ('                X_test_seq, y_test_seq = self.preprocessor.create_sequences(',
         '                    X_test_seq, y_test_seq = self.preprocessor.create_sequences('),
        
        # Fix else blocks
        ('            else:\n                # Fallback to random sampling if not enough samples\n            support_indices',
         '            else:\n                # Fallback to random sampling if not enough samples\n                support_indices'),
        
        # Fix if blocks
        ('                    if probs_np.shape[1] > 1:\n                    final_predictions',
         '                    if probs_np.shape[1] > 1:\n                        final_predictions'),
        
        # Fix threshold calculation
        ('                        else:\n                            # Use Youden\'s J statistic for balanced data\n                        optimal_idx',
         '                        else:\n                            # Use Youden\'s J statistic for balanced data\n                            optimal_idx'),
    ]
    
    for old, new in fixes:
        content = content.replace(old, new)
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Specific indentation fixes applied!")

if __name__ == "__main__":
    fix_specific_indentation()



