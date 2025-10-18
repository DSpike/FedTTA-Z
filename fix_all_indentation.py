#!/usr/bin/env python3
"""
Comprehensive indentation fix for main.py
"""

def fix_all_indentation():
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix specific problematic patterns
        if 'max_index = len(y_test) - 1' in line and line.startswith('    '):
            # This line should not be indented
            fixed_lines.append('max_index = len(y_test) - 1\n')
        elif 'if max_index >= 0:' in line and line.startswith('    '):
            # This line should not be indented
            fixed_lines.append('if max_index >= 0:\n')
        elif 'y_test = y_test[:max_index + 1]' in line and line.startswith('    '):
            # This line should not be indented
            fixed_lines.append('y_test = y_test[:max_index + 1]\n')
        elif 'logger.info(f"Adjusted y_test length to {len(y_test)}")' in line and line.startswith('    '):
            # This line should not be indented
            fixed_lines.append('logger.info(f"Adjusted y_test length to {len(y_test)}")\n')
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("All indentation fixes applied")

if __name__ == "__main__":
    fix_all_indentation()