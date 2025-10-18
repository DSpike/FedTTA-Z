#!/usr/bin/env python3
"""
Fix multiple colons in main.py
"""

import re

def fix_multiple_colons():
    """Fix all multiple colons in main.py"""
    
    print("🔧 Fixing multiple colons...")
    
    # Read the file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix multiple colons
    content = re.sub(r'::+', ':', content)
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed multiple colons!")

if __name__ == "__main__":
    fix_multiple_colons()







