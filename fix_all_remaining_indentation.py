#!/usr/bin/env python3
"""
Fix ALL remaining indentation errors in main.py
"""

def fix_all_remaining_indentation():
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix specific problematic patterns
        if line.strip() == 'else:' and i + 1 < len(lines):
            # Check if next line is not properly indented
            next_line = lines[i + 1]
            if next_line.strip() and not next_line.startswith('    '):
                # Fix the indentation
                fixed_lines.append(line)
                i += 1
                # Fix all subsequent lines until we hit another control structure
                while i < len(lines) and not lines[i].strip().startswith(('if ', 'else:', 'elif ', 'try:', 'except', 'finally', 'for ', 'while ', 'def ', 'class ')):
                    if lines[i].strip():  # If line is not empty
                        if not lines[i].startswith('    '):  # If not properly indented
                            fixed_lines.append('    ' + lines[i].lstrip())
                        else:
                            fixed_lines.append(lines[i])
                    else:
                        fixed_lines.append(lines[i])
                    i += 1
                # Don't increment i here as we want to process the control structure
                continue
            else:
                fixed_lines.append(line)
        elif line.strip() == 'if ' and i + 1 < len(lines):
            # Check if next line is not properly indented
            next_line = lines[i + 1]
            if next_line.strip() and not next_line.startswith('    '):
                # Fix the indentation
                fixed_lines.append(line)
                i += 1
                # Fix all subsequent lines until we hit another control structure
                while i < len(lines) and not lines[i].strip().startswith(('if ', 'else:', 'elif ', 'try:', 'except', 'finally', 'for ', 'while ', 'def ', 'class ')):
                    if lines[i].strip():  # If line is not empty
                        if not lines[i].startswith('    '):  # If not properly indented
                            fixed_lines.append('    ' + lines[i].lstrip())
                        else:
                            fixed_lines.append(lines[i])
                    else:
                        fixed_lines.append(lines[i])
                    i += 1
                # Don't increment i here as we want to process the control structure
                continue
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("All remaining indentation fixes applied")

if __name__ == "__main__":
    fix_all_remaining_indentation()



