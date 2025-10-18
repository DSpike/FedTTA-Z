#!/usr/bin/env python3
"""
Fix indentation errors in main.py
"""

import re

def fix_indentation():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix specific indentation patterns
    fixes = [
        # Fix try blocks without proper indentation
        (r'(\s+)try:\s*\n(\s+)X_test_seq, y_test_seq = self\.preprocessor\.create_sequences\(', 
         r'\1try:\n\1    X_test_seq, y_test_seq = self.preprocessor.create_sequences('),
        
        # Fix other similar patterns
        (r'(\s+)try:\s*\n(\s+)([^#\s].*)', 
         r'\1try:\n\1    \3'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Indentation fixes applied")

if __name__ == "__main__":
    fix_indentation()