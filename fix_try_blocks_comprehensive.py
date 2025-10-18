#!/usr/bin/env python3
"""
Comprehensive fix for all try block indentation issues in main.py
"""

def fix_try_blocks():
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a try statement
        if line.strip() == 'try:' and i + 1 < len(lines):
            fixed_lines.append(line)
            i += 1
            
            # Fix all lines until we hit except/finally
            while i < len(lines):
                next_line = lines[i]
                
                # Stop if we hit except or finally
                if next_line.strip().startswith(('except', 'finally')):
                    fixed_lines.append(next_line)
                    i += 1
                    break
                
                # Fix indentation for non-empty lines
                if next_line.strip():
                    if not next_line.startswith('    '):  # Not properly indented
                        fixed_lines.append('    ' + next_line.lstrip())
                    else:
                        fixed_lines.append(next_line)
                else:
                    fixed_lines.append(next_line)
                
                i += 1
        else:
            fixed_lines.append(line)
            i += 1
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("All try block indentation issues fixed")

if __name__ == "__main__":
    fix_try_blocks()



