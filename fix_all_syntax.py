#!/usr/bin/env python3
"""
Comprehensive syntax fix script for main.py
"""

import re

def fix_syntax():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix common syntax issues
    fixes = [
        # Fix missing closing braces
        (r'return \{\s*\n\s*def ', 'return {}\n    \n    def '),
        (r'return \{\s*\n\s*try:', 'return {}\n        \n        try:'),
        (r'return \{\s*\n\s*if ', 'return {}\n        \n        if '),
        
        # Fix indentation issues
        (r'^        X_test_tensor = torch\.FloatTensor\(X_test\)\.to\(device\)', '            X_test_tensor = torch.FloatTensor(X_test).to(device)'),
        (r'^        y_test_tensor = torch\.LongTensor\(y_test\)\.to\(device\)', '            y_test_tensor = torch.LongTensor(y_test).to(device)'),
        (r'^        support_x = X_test_tensor\[:support_size\]', '            support_x = X_test_tensor[:support_size]'),
        (r'^        support_y = y_test_tensor\[:support_size\]', '            support_y = y_test_tensor[:support_size]'),
        (r'^        query_x = X_test_tensor\[support_size:support_size\+query_size\]', '            query_x = X_test_tensor[support_size:support_size+query_size]'),
        (r'^        query_y = y_test_tensor\[support_size:support_size\+query_size\]', '            query_y = y_test_tensor[support_size:support_size+query_size]'),
        
        # Fix other common indentation issues
        (r'^    def ', '    def '),
        (r'^        def ', '        def '),
        (r'^            def ', '            def '),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Syntax fixed!")

if __name__ == "__main__":
    fix_syntax()









