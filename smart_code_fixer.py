#!/usr/bin/env python3
"""
Smart Code Fixer for main.py
This script intelligently diagnoses and fixes all syntax and indentation issues
"""

import re
import ast
import sys
from typing import List, Tuple, Dict

class SmartCodeFixer:
    def __init__(self, filename: str = "main.py"):
        self.filename = filename
        self.content = ""
        self.lines = []
        self.issues_found = []
        self.fixes_applied = []
    
    def load_file(self):
        """Load the file content"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.content = f.read()
            self.lines = self.content.split('\n')
            print(f"✅ Loaded {self.filename} ({len(self.lines)} lines)")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            sys.exit(1)
    
    def diagnose_issues(self):
        """Diagnose all issues in the code"""
        print("🔍 Diagnosing issues...")
        
        # Check for syntax errors
        try:
            ast.parse(self.content)
            print("✅ No syntax errors found")
        except SyntaxError as e:
            print(f"❌ Syntax error found: {e}")
            self.issues_found.append(f"Syntax error at line {e.lineno}: {e.msg}")
        
        # Check for common issues
        self._check_missing_colons()
        self._check_malformed_lines()
        self._check_indentation_issues()
        self._check_missing_newlines()
        self._check_escaped_quotes()
        
        print(f"📊 Found {len(self.issues_found)} issues")
        for i, issue in enumerate(self.issues_found, 1):
            print(f"  {i}. {issue}")
    
    def _check_missing_colons(self):
        """Check for missing colons after control statements"""
        patterns = [
            (r'def\s+\w+\([^)]*\)\s*$', 'Function definition missing colon'),
            (r'class\s+\w+.*\)?\s*$', 'Class definition missing colon'),
            (r'if\s+.*\)?\s*$', 'If statement missing colon'),
            (r'for\s+.*\)?\s*$', 'For loop missing colon'),
            (r'while\s+.*\)?\s*$', 'While loop missing colon'),
            (r'elif\s+.*\)?\s*$', 'Elif statement missing colon'),
            (r'else\s*$', 'Else statement missing colon'),
            (r'try\s*$', 'Try statement missing colon'),
            (r'except\s+.*\)?\s*$', 'Except statement missing colon'),
            (r'finally\s*$', 'Finally statement missing colon'),
        ]
        
        for i, line in enumerate(self.lines, 1):
            for pattern, message in patterns:
                if re.search(pattern, line.strip()) and not line.strip().endswith(':'):
                    self.issues_found.append(f"Line {i}: {message}")
    
    def _check_malformed_lines(self):
        """Check for malformed lines"""
        for i, line in enumerate(self.lines, 1):
            stripped = line.strip()
            
            # Check for standalone colons
            if stripped == ':':
                self.issues_found.append(f"Line {i}: Standalone colon")
            
            # Check for malformed dictionary syntax
            if re.search(r'\w+\s*=\s*\{:', line):
                self.issues_found.append(f"Line {i}: Malformed dictionary syntax")
            
            # Check for malformed function calls
            if re.search(r'\w+\s*=\s*\w+\s*\([^)]*\)\s*:', line):
                self.issues_found.append(f"Line {i}: Malformed function call with colon")
    
    def _check_indentation_issues(self):
        """Check for indentation issues"""
        expected_indent = 0
        in_multiline_string = False
        string_delimiter = None
        
        for i, line in enumerate(self.lines, 1):
            if not line.strip():
                continue
            
            # Check for multiline strings
            if '"""' in line or "'''" in line:
                if not in_multiline_string:
                    in_multiline_string = True
                    string_delimiter = '"""' if '"""' in line else "'''"
                elif string_delimiter in line:
                    in_multiline_string = False
                    string_delimiter = None
            
            if in_multiline_string:
                continue
            
            stripped = line.strip()
            
            # Check for inconsistent indentation
            if stripped.startswith('def ') or stripped.startswith('class '):
                if not line.startswith('    ') and not line.startswith('def ') and not line.startswith('class '):
                    self.issues_found.append(f"Line {i}: Function/class definition indentation issue")
            elif stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while '):
                if not line.startswith('        '):
                    self.issues_found.append(f"Line {i}: Control flow indentation issue")
            elif stripped.startswith('return ') or stripped.startswith('break ') or stripped.startswith('continue '):
                if not line.startswith('            '):
                    self.issues_found.append(f"Line {i}: Control statement indentation issue")
    
    def _check_missing_newlines(self):
        """Check for missing newlines between functions"""
        for i in range(len(self.lines) - 1):
            current_line = self.lines[i].strip()
            next_line = self.lines[i + 1].strip()
            
            # Check for functions/classes without proper separation
            if (current_line.startswith('return ') or current_line.startswith('break ') or 
                current_line.startswith('continue ') or current_line.endswith('}')):
                if next_line.startswith('def ') or next_line.startswith('class '):
                    if not self.lines[i + 1].startswith('\n') and not self.lines[i].endswith('\n'):
                        self.issues_found.append(f"Line {i + 1}: Missing newline before function/class definition")
    
    def _check_escaped_quotes(self):
        """Check for escaped quotes that shouldn't be escaped"""
        for i, line in enumerate(self.lines, 1):
            if '\\\'' in line or '\\"' in line:
                self.issues_found.append(f"Line {i}: Escaped quotes that may cause syntax errors")
    
    def fix_all_issues(self):
        """Fix all identified issues"""
        print("🔧 Applying fixes...")
        
        # Fix 1: Add missing colons
        self._fix_missing_colons()
        
        # Fix 2: Fix malformed lines
        self._fix_malformed_lines()
        
        # Fix 3: Fix indentation issues
        self._fix_indentation_issues()
        
        # Fix 4: Fix missing newlines
        self._fix_missing_newlines()
        
        # Fix 5: Fix escaped quotes
        self._fix_escaped_quotes()
        
        # Fix 6: Fix specific known issues
        self._fix_specific_issues()
        
        print(f"✅ Applied {len(self.fixes_applied)} fixes")
    
    def _fix_missing_colons(self):
        """Fix missing colons"""
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
            if re.search(pattern, self.content):
                self.content = re.sub(pattern, replacement, self.content)
                self.fixes_applied.append("Fixed missing colons")
    
    def _fix_malformed_lines(self):
        """Fix malformed lines"""
        # Fix standalone colons
        self.content = re.sub(r'^\s*:\s*$', '', self.content, flags=re.MULTILINE)
        
        # Fix malformed dictionary syntax
        self.content = re.sub(r'(\w+)\s*=\s*\{:', r'\1 = {', self.content)
        
        # Fix malformed function calls with colons
        self.content = re.sub(r'(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*:', r'\1 = \2(\3)', self.content)
        
        self.fixes_applied.append("Fixed malformed lines")
    
    def _fix_indentation_issues(self):
        """Fix indentation issues"""
        lines = self.content.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Fix function/class definitions
            if stripped.startswith('def ') or stripped.startswith('class '):
                if not line.startswith('    ') and not line.startswith('def ') and not line.startswith('class '):
                    line = '    ' + line.lstrip()
            
            # Fix control flow statements
            elif stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while '):
                if not line.startswith('        '):
                    line = '        ' + line.lstrip()
            
            # Fix control statements
            elif stripped.startswith('return ') or stripped.startswith('break ') or stripped.startswith('continue '):
                if not line.startswith('            '):
                    line = '            ' + line.lstrip()
            
            fixed_lines.append(line)
        
        self.content = '\n'.join(fixed_lines)
        self.fixes_applied.append("Fixed indentation issues")
    
    def _fix_missing_newlines(self):
        """Fix missing newlines between functions"""
        # Fix missing newlines before function/class definitions
        self.content = re.sub(r'(\w+)\s+def\s+(\w+)', r'\1\n    \n    def \2', self.content)
        self.content = re.sub(r'(\w+)\s+class\s+(\w+)', r'\1\n    \n    class \2', self.content)
        
        self.fixes_applied.append("Fixed missing newlines")
    
    def _fix_escaped_quotes(self):
        """Fix escaped quotes"""
        self.content = self.content.replace('\\\'', "'")
        self.content = self.content.replace('\\"', '"')
        
        self.fixes_applied.append("Fixed escaped quotes")
    
    def _fix_specific_issues(self):
        """Fix specific known issues"""
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
        
        if re.search(pattern, self.content, re.DOTALL):
            self.content = re.sub(pattern, replacement + r'\2', self.content, flags=re.DOTALL)
            self.fixes_applied.append("Fixed _aggregate_meta_histories method")
    
    def save_file(self):
        """Save the fixed content back to file"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                f.write(self.content)
            print(f"✅ Saved fixed content to {self.filename}")
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            sys.exit(1)
    
    def verify_fixes(self):
        """Verify that all fixes were applied correctly"""
        print("🔍 Verifying fixes...")
        
        try:
            ast.parse(self.content)
            print("✅ Syntax verification passed")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax verification failed: {e}")
            return False
    
    def run(self):
        """Run the complete fix process"""
        print("🚀 Starting Smart Code Fixer...")
        
        # Load file
        self.load_file()
        
        # Diagnose issues
        self.diagnose_issues()
        
        if not self.issues_found:
            print("✅ No issues found! Code is already clean.")
            return
        
        # Apply fixes
        self.fix_all_issues()
        
        # Save file
        self.save_file()
        
        # Verify fixes
        if self.verify_fixes():
            print("🎉 All issues fixed successfully!")
        else:
            print("⚠️  Some issues may remain. Please check manually.")

def main():
    """Main function"""
    fixer = SmartCodeFixer("main.py")
    fixer.run()

if __name__ == "__main__":
    main()









