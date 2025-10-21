#!/usr/bin/env python3
"""
Specific syntax error fixer for main.py
This script fixes the specific syntax error we encountered
"""

def fix_specific_syntax_error():
    """Fix the specific syntax error in main.py"""
    
    print("Reading main.py...")
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Fixing specific syntax error...")
    
    # Fix the specific issue: "return False    def _aggregate_meta_histories"
    # This should be "return False\n    \n    def _aggregate_meta_histories"
    content = content.replace('return False    def _aggregate_meta_histories', 'return False\n    \n    def _aggregate_meta_histories')
    
    # Fix any other similar issues
    content = content.replace('return False    def ', 'return False\n    \n    def ')
    content = content.replace('return True    def ', 'return True\n    \n    def ')
    content = content.replace('return None    def ', 'return None\n    \n    def ')
    content = content.replace('return {}    def ', 'return {}\n    \n    def ')
    content = content.replace('return []    def ', 'return []\n    \n    def ')
    
    # Fix any other missing newlines between functions
    import re
    # Pattern to find function definitions that are not properly separated
    pattern = r'(\w+)\s+def\s+(\w+)'
    def fix_function_separation(match):
        first_part = match.group(1)
        second_part = match.group(2)
        # Only fix if it's not already properly separated
        if not first_part.endswith('\n') and not first_part.endswith(':'):
            return f'{first_part}\n    \n    def {second_part}'
        return match.group(0)
    content = re.sub(pattern, fix_function_separation, content)
    
    print("Writing fixed content back to main.py...")
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Specific syntax error fixed successfully!")

if __name__ == "__main__":
    fix_specific_syntax_error()









