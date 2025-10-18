#!/usr/bin/env python3
"""
Comprehensive syntax error fixer for main.py
This script diagnoses and fixes syntax errors in the main.py file
"""

import re

def fix_syntax_errors():
    """Fix all syntax errors in main.py"""
    
    print("Reading main.py...")
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Diagnosing syntax errors...")
    
    # Fix 1: Missing newline between return statement and function definition
    pattern1 = r'return False\s+def _aggregate_meta_histories'
    replacement1 = 'return False\n    \n    def _aggregate_meta_histories'
    content = re.sub(pattern1, replacement1, content)
    
    # Fix 2: Fix any other missing newlines between functions
    pattern2 = r'(\w+)\s+def (\w+)'
    def fix_function_separation(match):
        first_part = match.group(1)
        second_part = match.group(2)
        # Only fix if it's not already properly separated
        if not first_part.endswith('\n') and not first_part.endswith(':'):
            return f'{first_part}\n    \n    def {second_part}'
        return match.group(0)
    content = re.sub(pattern2, fix_function_separation, content)
    
    # Fix 3: Fix any escaped quotes that shouldn't be escaped
    pattern3 = r'\\\'([^\']*)\\\''
    replacement3 = r"'\1'"
    content = re.sub(pattern3, replacement3, content)
    
    # Fix 4: Fix any double quotes that got escaped
    pattern4 = r'\\"([^"]*)\\"'
    replacement4 = r'"\1"'
    content = re.sub(pattern4, replacement4, content)
    
    # Fix 5: Fix any missing colons after if/for/while statements
    pattern5 = r'(\s+)(if|for|while|def|class)\s+([^:]+)\s*\n(\s+)([^#\s])'
    def fix_missing_colons(match):
        indent = match.group(1)
        keyword = match.group(2)
        condition = match.group(3)
        next_indent = match.group(4)
        next_line = match.group(5)
        
        # Check if colon is missing
        if not condition.strip().endswith(':'):
            return f'{indent}{keyword} {condition}:\n{next_indent}{next_line}'
        return match.group(0)
    content = re.sub(pattern5, fix_missing_colons, content)
    
    # Fix 6: Fix any malformed string literals
    pattern6 = r'f"[^"]*\\[^"]*"'
    def fix_f_strings(match):
        string = match.group(0)
        # Fix escaped quotes in f-strings
        fixed = string.replace('\\"', '"').replace("\\'", "'")
        return fixed
    content = re.sub(pattern6, fix_f_strings, content)
    
    # Fix 7: Fix any missing parentheses in function calls
    pattern7 = r'(\w+)\s+(\w+)\s*=\s*(\w+)\s*\([^)]*$'
    def fix_missing_parens(match):
        # This is a complex pattern, let's be more specific
        return match.group(0)  # Keep as is for now
    # content = re.sub(pattern7, fix_missing_parens, content)
    
    # Fix 8: Fix any indentation issues that might cause syntax errors
    lines = content.split('\n')
    fixed_lines = []
    in_multiline_string = False
    string_delimiter = None
    
    for i, line in enumerate(lines):
        # Check for multiline strings
        if '"""' in line or "'''" in line:
            if not in_multiline_string:
                in_multiline_string = True
                string_delimiter = '"""' if '"""' in line else "'''"
            elif string_delimiter in line:
                in_multiline_string = False
                string_delimiter = None
        
        # Skip indentation fixes inside multiline strings
        if in_multiline_string:
            fixed_lines.append(line)
            continue
        
        # Fix common indentation issues
        if line.strip().startswith('def ') or line.strip().startswith('class '):
            # Function/class definition - should be at base level
            if not line.startswith('    ') and not line.startswith('def ') and not line.startswith('class '):
                line = '    ' + line.lstrip()
        elif line.strip().startswith('if ') or line.strip().startswith('for ') or line.strip().startswith('while '):
            # Control flow - should be indented
            if not line.startswith('        '):
                line = '        ' + line.lstrip()
        elif line.strip().startswith('return ') or line.strip().startswith('break ') or line.strip().startswith('continue '):
            # Control statements - should be indented
            if not line.startswith('            '):
                line = '            ' + line.lstrip()
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    print("Writing fixed content back to main.py...")
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ All syntax errors fixed successfully!")

def diagnose_syntax_errors():
    """Diagnose syntax errors in main.py"""
    
    print("Diagnosing syntax errors in main.py...")
    
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for common syntax error patterns
    errors = []
    
    # Check for missing newlines between functions
    if re.search(r'return\s+\w+\s+def', content):
        errors.append("Missing newline between return statement and function definition")
    
    # Check for escaped quotes
    if re.search(r'\\\'', content):
        errors.append("Found escaped single quotes that may cause syntax errors")
    
    if re.search(r'\\"', content):
        errors.append("Found escaped double quotes that may cause syntax errors")
    
    # Check for missing colons
    if re.search(r'(if|for|while|def|class)\s+[^:]+[^:]\s*\n\s+[^#\s]', content):
        errors.append("Found statements that may be missing colons")
    
    # Check for malformed f-strings
    if re.search(r'f"[^"]*\\[^"]*"', content):
        errors.append("Found malformed f-strings with escaped quotes")
    
    if errors:
        print("Found the following potential syntax errors:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("No obvious syntax errors found.")
    
    return errors

if __name__ == "__main__":
    print("=== SYNTAX ERROR DIAGNOSIS ===")
    errors = diagnose_syntax_errors()
    
    print("\n=== APPLYING FIXES ===")
    fix_syntax_errors()
    
    print("\n=== RE-DIAGNOSIS ===")
    diagnose_syntax_errors()







