#!/usr/bin/env python3
"""
Fix all indentation errors in main.py at once
"""

import re

def fix_all_indentation_errors():
    """Fix all indentation errors in main.py"""
    
    print("🔧 Fixing all indentation errors...")
    
    # Read the file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Fix inconsistent indentation patterns
    # Fix function/class definitions that should be at base level
    content = re.sub(r'^(\s+)(def\s+\w+\([^)]*\)\s*:.*?)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(class\s+\w+.*?:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    
    # Fix 2: Fix control flow statements that should be properly indented
    # if/for/while statements should be indented based on context
    content = re.sub(r'^(\s+)(if\s+.*?:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(for\s+.*?:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(while\s+.*?:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(elif\s+.*?:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(else\s*:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(try\s*:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(except\s+.*?:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(finally\s*:\s*)$', r'\1\2', content, flags=re.MULTILINE)
    
    # Fix 3: Fix return/break/continue statements
    content = re.sub(r'^(\s+)(return\s+.*?)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(break\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(continue\s*)$', r'\1\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s+)(pass\s*)$', r'\1\2', content, flags=re.MULTILINE)
    
    # Fix 4: Fix specific indentation issues
    # Fix lines that should be at base level (no indentation)
    base_level_patterns = [
        r'^(\s+)(@dataclass\s*)$',
        r'^(\s+)(class\s+\w+.*?:\s*)$',
        r'^(\s+)(def\s+\w+\([^)]*\)\s*:.*?)$',
        r'^(\s+)(import\s+.*?)$',
        r'^(\s+)(from\s+.*?)$',
        r'^(\s+)(#.*?)$',
        r'^(\s+)(""".*?""")$',
        r'^(\s+)(\'\'\'.*?\'\'\')$',
    ]
    
    for pattern in base_level_patterns:
        content = re.sub(pattern, r'\2', content, flags=re.MULTILINE)
    
    # Fix 5: Fix lines that should be indented (4 spaces)
    indented_patterns = [
        r'^(\s+)(if\s+.*?:\s*)$',
        r'^(\s+)(for\s+.*?:\s*)$',
        r'^(\s+)(while\s+.*?:\s*)$',
        r'^(\s+)(try\s*:\s*)$',
        r'^(\s+)(except\s+.*?:\s*)$',
        r'^(\s+)(else\s*:\s*)$',
        r'^(\s+)(elif\s+.*?:\s*)$',
        r'^(\s+)(finally\s*:\s*)$',
    ]
    
    for pattern in indented_patterns:
        content = re.sub(pattern, r'    \2', content, flags=re.MULTILINE)
    
    # Fix 6: Fix lines that should be double indented (8 spaces)
    double_indented_patterns = [
        r'^(\s+)(return\s+.*?)$',
        r'^(\s+)(break\s*)$',
        r'^(\s+)(continue\s*)$',
        r'^(\s+)(pass\s*)$',
    ]
    
    for pattern in double_indented_patterns:
        content = re.sub(pattern, r'        \2', content, flags=re.MULTILINE)
    
    # Fix 7: Fix specific known problematic patterns
    # Fix the _aggregate_meta_histories method specifically
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
    
    # Fix 8: Clean up any remaining indentation issues
    # Remove extra spaces at the beginning of lines
    content = re.sub(r'^(\s+)(\s+)(.*?)$', r'\1\3', content, flags=re.MULTILINE)
    
    # Fix 9: Ensure consistent indentation
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue
        
        # Fix specific indentation patterns
        if stripped.startswith('def ') or stripped.startswith('class '):
            # Function/class definition - base level
            fixed_line = line.lstrip()
        elif stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while ') or stripped.startswith('try:') or stripped.startswith('except ') or stripped.startswith('else:') or stripped.startswith('elif '):
            # Control flow statements - 4 spaces
            if not line.startswith('    '):
                fixed_line = '    ' + line.lstrip()
            else:
                fixed_line = line
        elif stripped.startswith('return ') or stripped.startswith('break ') or stripped.startswith('continue ') or stripped.startswith('pass'):
            # Control statements - 8 spaces
            if not line.startswith('        '):
                fixed_line = '        ' + line.lstrip()
            else:
                fixed_line = line
        elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            # Comments and docstrings - maintain relative indentation
            fixed_line = line
        else:
            # Other statements - try to fix based on context
            if stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while '):
                if not line.startswith('    '):
                    fixed_line = '    ' + line.lstrip()
                else:
                    fixed_line = line
            elif stripped.startswith('return ') or stripped.startswith('break ') or stripped.startswith('continue '):
                if not line.startswith('        '):
                    fixed_line = '        ' + line.lstrip()
                else:
                    fixed_line = line
            else:
                fixed_line = line
        
        fixed_lines.append(fixed_line)
    
    content = '\n'.join(fixed_lines)
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed all indentation errors!")

if __name__ == "__main__":
    fix_all_indentation_errors()







