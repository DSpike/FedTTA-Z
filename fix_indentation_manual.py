#!/usr/bin/env python3
"""
Manual indentation fix for main.py - handles specific patterns
"""

def fix_indentation_manual():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix specific problematic patterns
        if line.strip() == 'try:' and i + 1 < len(lines):
            fixed_lines.append(line)
            i += 1
            
            # Fix the next few lines that should be indented
            indent_level = len(line) - len(line.lstrip())
            base_indent = ' ' * (indent_level + 4)  # Add 4 spaces for try block content
            
            while i < len(lines):
                current_line = lines[i]
                
                # Stop if we hit except, finally, or another control structure
                if current_line.strip().startswith(('except', 'finally', 'if ', 'else:', 'elif ', 'for ', 'while ', 'def ', 'class ')):
                    break
                
                # Fix indentation
                if current_line.strip():  # Non-empty line
                    if not current_line.startswith('    '):  # Not properly indented
                        fixed_lines.append(base_indent + current_line.lstrip())
                    else:
                        fixed_lines.append(current_line)
                else:
                    fixed_lines.append(current_line)
                
                i += 1
        else:
            fixed_lines.append(line)
            i += 1
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print("Manual indentation fix applied")

if __name__ == "__main__":
    fix_indentation_manual()



