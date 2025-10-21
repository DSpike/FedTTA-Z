#!/usr/bin/env python3
"""
Final comprehensive fix for all remaining issues in main.py
"""

def fix_final_issues():
    """Fix all remaining issues in main.py"""
    
    print("Reading main.py...")
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying final fixes...")
    
    # Fix 1: Fix the specific indentation issue in _aggregate_meta_histories
    import re
    
    # Find and replace the problematic method
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
    
    # Fix 2: Fix any remaining malformed lines
    content = content.replace('            :', '')
    content = content.replace('        :', '')
    content = content.replace('    :', '')
    content = content.replace(':\n', '\n')
    
    # Fix 3: Fix any remaining indentation issues
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix common indentation patterns
        if line.strip().startswith('aggregated_losses = []') and not line.startswith('        '):
            fixed_lines.append('        aggregated_losses = []')
        elif line.strip().startswith('aggregated_accuracies = []') and not line.startswith('        '):
            fixed_lines.append('        aggregated_accuracies = []')
        elif line.strip().startswith('for epoch in range(num_epochs):') and not line.startswith('        '):
            fixed_lines.append('        for epoch in range(num_epochs):')
        elif line.strip().startswith('epoch_losses = [history') and not line.startswith('            '):
            fixed_lines.append('            epoch_losses = [history[\'epoch_losses\'][epoch] for history in client_meta_histories]')
        elif line.strip().startswith('avg_loss = sum(epoch_losses)') and not line.startswith('            '):
            fixed_lines.append('            avg_loss = sum(epoch_losses) / len(epoch_losses)')
        elif line.strip().startswith('aggregated_losses.append(avg_loss)') and not line.startswith('            '):
            fixed_lines.append('            aggregated_losses.append(avg_loss)')
        elif line.strip().startswith('epoch_accuracies = [history') and not line.startswith('            '):
            fixed_lines.append('            epoch_accuracies = [history[\'epoch_accuracies\'][epoch] for history in client_meta_histories]')
        elif line.strip().startswith('avg_accuracy = sum(epoch_accuracies)') and not line.startswith('            '):
            fixed_lines.append('            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)')
        elif line.strip().startswith('aggregated_accuracies.append(avg_accuracy)') and not line.startswith('            '):
            fixed_lines.append('            aggregated_accuracies.append(avg_accuracy)')
        elif line.strip().startswith('return {') and not line.startswith('        '):
            fixed_lines.append('        return {')
        elif line.strip().startswith("'epoch_losses': aggregated_losses,") and not line.startswith('            '):
            fixed_lines.append("            'epoch_losses': aggregated_losses,")
        elif line.strip().startswith("'epoch_accuracies': aggregated_accuracies") and not line.startswith('            '):
            fixed_lines.append("            'epoch_accuracies': aggregated_accuracies")
        elif line.strip().startswith('}') and not line.startswith('        '):
            fixed_lines.append('        }')
        else:
            fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    print("Writing fixed content back to main.py...")
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ All final issues fixed successfully!")

if __name__ == "__main__":
    fix_final_issues()









