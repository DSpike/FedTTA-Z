#!/usr/bin/env python3
"""
Comprehensive syntax and indentation fix for main.py
"""

import re

def fix_syntax_comprehensive():
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix specific syntax issues
        if 'else:' in line and i > 0:
            prev_line = lines[i-1].strip()
            # Check if the previous line ends with a colon or is a try/except block
            if not prev_line.endswith(':') and not prev_line.startswith('try') and not prev_line.startswith('except'):
                # This else doesn't have a matching if/try, remove it
                i += 1
                continue
        
        # Fix indentation issues
        if line.strip().startswith('optimal_threshold, roc_auc, fpr, tpr, thresholds = find_optimal_threshold('):
            # Ensure proper indentation for threshold optimization
            indent = '                        '
            fixed_lines.append(indent + line.strip() + '\n')
        elif line.strip().startswith('logger.info(f"📊 Static threshold selected:'):
            # Ensure proper indentation for logging
            indent = '                        '
            fixed_lines.append(indent + line.strip() + '\n')
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Syntax and indentation fixed successfully!")

if __name__ == "__main__":
    fix_syntax_comprehensive()



