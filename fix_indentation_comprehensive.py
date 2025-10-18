#!/usr/bin/env python3
"""
Comprehensive indentation fix for main.py
"""

def fix_indentation():
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix specific indentation issues
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix the training sequence creation indentation
        if i >= 735 and i <= 742 and 'create_sequences' in line:
            if 'X_train_seq, y_train_seq = self.preprocessor.create_sequences(' in line:
                fixed_lines.append('                    X_train_seq, y_train_seq = self.preprocessor.create_sequences(\n')
            elif 'X_train_subset,' in line:
                fixed_lines.append('                        X_train_subset,\n')
            elif 'y_train_subset,' in line:
                fixed_lines.append('                        y_train_subset,\n')
            elif 'sequence_length=self.config.sequence_length,' in line:
                fixed_lines.append('                        sequence_length=self.config.sequence_length,\n')
            elif 'stride=self.config.sequence_stride,' in line:
                fixed_lines.append('                        stride=self.config.sequence_stride,\n')
            elif 'zero_pad=True' in line:
                fixed_lines.append('                        zero_pad=True\n')
            elif ')' in line and i == 742:
                fixed_lines.append('                    )\n')
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Indentation fixed successfully!")

if __name__ == "__main__":
    fix_indentation()