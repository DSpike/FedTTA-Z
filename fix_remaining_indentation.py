#!/usr/bin/env python3
"""
Fix all remaining indentation errors in main.py
"""

import re

def fix_remaining_indentation():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines for easier processing
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix specific patterns that cause indentation errors
        if re.match(r'^\s+else:\s*$', line) and i + 1 < len(lines):
            # Check if next line is not properly indented
            next_line = lines[i + 1]
            if next_line.strip() and not next_line.startswith('    '):
                # Fix the indentation of the next line
                fixed_lines.append(line)
                i += 1
                if i < len(lines):
                    # Indent the next line properly
                    if lines[i].strip():
                        fixed_lines.append('    ' + lines[i].lstrip())
                    else:
                        fixed_lines.append(lines[i])
            else:
                fixed_lines.append(line)
        elif re.match(r'^\s+if\s+.*:\s*$', line) and i + 1 < len(lines):
            # Check if next line is not properly indented
            next_line = lines[i + 1]
            if next_line.strip() and not next_line.startswith('    '):
                # Fix the indentation of the next line
                fixed_lines.append(line)
                i += 1
                if i < len(lines):
                    # Indent the next line properly
                    if lines[i].strip():
                        fixed_lines.append('    ' + lines[i].lstrip())
                    else:
                        fixed_lines.append(lines[i])
            else:
                fixed_lines.append(line)
        elif re.match(r'^\s+try:\s*$', line) and i + 1 < len(lines):
            # Check if next line is not properly indented
            next_line = lines[i + 1]
            if next_line.strip() and not next_line.startswith('    '):
                # Fix the indentation of the next line
                fixed_lines.append(line)
                i += 1
                if i < len(lines):
                    # Indent the next line properly
                    if lines[i].strip():
                        fixed_lines.append('    ' + lines[i].lstrip())
                    else:
                        fixed_lines.append(lines[i])
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Join lines back
    fixed_content = '\n'.join(fixed_lines)
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Remaining indentation fixes applied")

if __name__ == "__main__":
    fix_remaining_indentation()



