#!/usr/bin/env python3
"""
Fix try-except indentation issues in main.py
"""

def fix_try_except_indentation():
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
            
            # Fix the next few lines that should be indented
            while i < len(lines) and not lines[i].strip().startswith('except') and not lines[i].strip().startswith('finally'):
                next_line = lines[i]
                if next_line.strip():  # If line is not empty
                    if not next_line.startswith('    '):  # If not properly indented
                        fixed_lines.append('    ' + next_line.lstrip())
                    else:
                        fixed_lines.append(next_line)
                else:
                    fixed_lines.append(next_line)
                i += 1
            
            # Add the except/finally block
            if i < len(lines):
                fixed_lines.append(lines[i])
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Try-except indentation fixes applied")

if __name__ == "__main__":
    fix_try_except_indentation()



