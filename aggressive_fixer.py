#!/usr/bin/env python3
"""
Aggressive Code Fixer for severely corrupted main.py
This script will completely rebuild the file structure
"""

import re
import os
from datetime import datetime

def aggressive_fix():
    """Aggressively fix the corrupted main.py file"""
    
    print("🚀 Starting Aggressive Code Fixer...")
    
    # Create a backup
    backup_name = f"main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    os.system(f"copy main.py {backup_name}")
    print(f"📁 Created backup: {backup_name}")
    
    # Read the corrupted file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔧 Applying aggressive fixes...")
    
    # Fix 1: Add missing colons everywhere
    patterns = [
        (r'(def\s+\w+\([^)]*\))\s*$', r'\1:'),
        (r'(class\s+\w+.*\)?)\s*$', r'\1:'),
        (r'(if\s+.*\)?)\s*$', r'\1:'),
        (r'(for\s+.*\)?)\s*$', r'\1:'),
        (r'(while\s+.*\)?)\s*$', r'\1:'),
        (r'(elif\s+.*\)?)\s*$', r'\1:'),
        (r'(else)\s*$', r'\1:'),
        (r'(try)\s*$', r'\1:'),
        (r'(except\s+.*\)?)\s*$', r'\1:'),
        (r'(finally)\s*$', r'\1:'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Fix 2: Remove malformed lines
    content = re.sub(r'^\s*:\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*:\s*\n', '\n', content, flags=re.MULTILINE)
    
    # Fix 3: Fix malformed dictionary syntax
    content = re.sub(r'(\w+)\s*=\s*\{:', r'\1 = {', content)
    
    # Fix 4: Fix malformed function calls
    content = re.sub(r'(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*:', r'\1 = \2(\3)', content)
    
    # Fix 5: Fix escaped quotes
    content = content.replace('\\\'', "'")
    content = content.replace('\\"', '"')
    
    # Fix 6: Fix indentation systematically
    lines = content.split('\n')
    fixed_lines = []
    indent_level = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue
        
        # Determine proper indentation based on content
        if stripped.startswith('def ') or stripped.startswith('class '):
            # Function/class definition - base level
            fixed_line = line.lstrip()
            indent_level = 0
        elif stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while ') or stripped.startswith('try:') or stripped.startswith('except ') or stripped.startswith('else:') or stripped.startswith('elif '):
            # Control flow statements
            if indent_level == 0:
                fixed_line = '    ' + line.lstrip()
            else:
                fixed_line = '        ' + line.lstrip()
        elif stripped.startswith('return ') or stripped.startswith('break ') or stripped.startswith('continue ') or stripped.startswith('pass'):
            # Control statements
            if indent_level == 0:
                fixed_line = '        ' + line.lstrip()
            else:
                fixed_line = '            ' + line.lstrip()
        elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            # Comments and docstrings - maintain relative indentation
            fixed_line = line
        else:
            # Other statements - try to fix indentation
            if stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while '):
                fixed_line = '        ' + line.lstrip()
            elif stripped.startswith('return ') or stripped.startswith('break ') or stripped.startswith('continue '):
                fixed_line = '            ' + line.lstrip()
            elif stripped.startswith('def ') or stripped.startswith('class '):
                fixed_line = line.lstrip()
            else:
                fixed_line = line
        
        fixed_lines.append(fixed_line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix 7: Fix specific known issues
    # Fix the _aggregate_meta_histories method
    pattern = r'(\s+def _aggregate_meta_histories\(self, client_meta_histories: List\[Dict\]\) -> Dict:.*?)(\n\s+def|\n\n\s+def|\Z)'
    
    replacement = '''    def _aggregate_meta_histories(self, client_meta_histories: List[Dict]) -> Dict:
        """
        Aggregate meta-learning histories from all clients.
        
        Args:
            client_meta_histories: List of meta-learning histories from clients
            
        Returns:
            aggregated_history: Aggregated meta-learning history
        """
        if not client_meta_histories:
            return {'epoch_losses': [], 'epoch_accuracies': []}
        
        # Average losses and accuracies across clients
        num_epochs = len(client_meta_histories[0]['epoch_losses'])
        aggregated_losses = []
        aggregated_accuracies = []
        
        for epoch in range(num_epochs):
            # Average loss across clients for this epoch
            epoch_losses = [history['epoch_losses'][epoch] for history in client_meta_histories]
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            aggregated_losses.append(avg_loss)
            
            # Average accuracy across clients for this epoch
            epoch_accuracies = [history['epoch_accuracies'][epoch] for history in client_meta_histories]
            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            aggregated_accuracies.append(avg_accuracy)
        
        return {
            'epoch_losses': aggregated_losses,
            'epoch_accuracies': aggregated_accuracies
        }
'''
    
    content = re.sub(pattern, replacement + r'\2', content, flags=re.DOTALL)
    
    # Fix 8: Fix missing newlines between functions
    content = re.sub(r'(\w+)\s+def\s+(\w+)', r'\1\n    \n    def \2', content)
    content = re.sub(r'(\w+)\s+class\s+(\w+)', r'\1\n    \n    class \2', content)
    
    # Fix 9: Fix any remaining syntax issues
    content = re.sub(r'(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*:', r'\1 = \2(\3)', content)
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Aggressive fixes applied!")
    
    # Test the fixed file
    print("🔍 Testing fixed file...")
    try:
        import ast
        ast.parse(content)
        print("✅ Syntax verification passed!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax verification failed: {e}")
        print(f"   Error at line {e.lineno}: {e.msg}")
        return False

if __name__ == "__main__":
    success = aggressive_fix()
    if success:
        print("🎉 File fixed successfully!")
    else:
        print("⚠️  File may still have issues. Manual review recommended.")







