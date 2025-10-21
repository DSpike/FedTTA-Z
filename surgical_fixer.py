#!/usr/bin/env python3
"""
Surgical Fixer - Only fixes syntax errors without losing recent improvements
"""

import re

def surgical_fix():
    """Surgically fix only syntax errors while preserving all recent improvements"""
    
    print("🔧 Starting Surgical Fix - Preserving Recent Improvements...")
    
    # Read the current file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📋 Applying surgical fixes...")
    
    # Fix 1: Add missing colons to function definitions
    content = re.sub(r'(def\s+\w+\([^)]*\))\s*$', r'\1:', content, flags=re.MULTILINE)
    
    # Fix 2: Add missing colons to class definitions
    content = re.sub(r'(class\s+\w+.*\)?)\s*$', r'\1:', content, flags=re.MULTILINE)
    
    # Fix 3: Add missing colons to control statements
    content = re.sub(r'(if\s+.*\)?)\s*$', r'\1:', content, flags=re.MULTILINE)
    content = re.sub(r'(for\s+.*\)?)\s*$', r'\1:', content, flags=re.MULTILINE)
    content = re.sub(r'(while\s+.*\)?)\s*$', r'\1:', content, flags=re.MULTILINE)
    content = re.sub(r'(elif\s+.*\)?)\s*$', r'\1:', content, flags=re.MULTILINE)
    content = re.sub(r'(else)\s*$', r'\1:', content, flags=re.MULTILINE)
    content = re.sub(r'(try)\s*$', r'\1:', content, flags=re.MULTILINE)
    content = re.sub(r'(except\s+.*\)?)\s*$', r'\1:', content, flags=re.MULTILINE)
    content = re.sub(r'(finally)\s*$', r'\1:', content, flags=re.MULTILINE)
    
    # Fix 4: Remove malformed lines (standalone colons)
    content = re.sub(r'^\s*:\s*$', '', content, flags=re.MULTILINE)
    
    # Fix 5: Fix malformed dictionary syntax
    content = re.sub(r'(\w+)\s*=\s*\{:', r'\1 = {', content)
    
    # Fix 6: Fix malformed function calls with colons
    content = re.sub(r'(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*:', r'\1 = \2(\3)', content)
    
    # Fix 7: Fix escaped quotes
    content = content.replace('\\\'', "'")
    content = content.replace('\\"', '"')
    
    # Fix 8: Fix specific indentation issues in _aggregate_meta_histories method
    # This is the only method we need to fix properly
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
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Surgical fixes applied!")
    
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
    success = surgical_fix()
    if success:
        print("🎉 Surgical fix successful! All recent improvements preserved.")
    else:
        print("⚠️  Some syntax issues remain. Manual review needed.")









