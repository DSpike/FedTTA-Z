#!/usr/bin/env python3
"""
Automated Dataset Switching Script
Switches between UNSW-NB15 and CICIDS2017 datasets with a single command.
Improved version with better error handling and safer parsing.
"""

import sys
import os
import re

def get_config_section(content, start_pattern, end_pattern):
    """Extract a section between two patterns"""
    start_idx = content.find(start_pattern)
    if start_idx == -1:
        return None, -1, -1
    end_idx = content.find(end_pattern, start_idx + len(start_pattern))
    if end_idx == -1:
        end_idx = len(content)
    return content[start_idx:end_idx + len(end_pattern)], start_idx, end_idx

def switch_to_cicids():
    """Switch configuration to CICIDS2017 dataset"""
    print("🔄 Switching to CICIDS2017 dataset...\n")
    
    # 1. Update config.py
    config_path = "config.py"
    if not os.path.exists(config_path):
        print(f"❌ Error: {config_path} not found!")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update data paths
    content = re.sub(
        r'data_path: str = "[^"]*"',
        'data_path: str = "CICIDS2017_train.csv"',
        content
    )
    content = re.sub(
        r'test_path: str = "[^"]*"',
        'test_path: str = "CICIDS2017_test.csv"',
        content
    )
    
    # Update zero_day_attack
    content = re.sub(
        r'zero_day_attack: str = "[^"]*"',
        'zero_day_attack: str = "PortScan"  # CICIDS2017 attack type',
        content
    )
    
    # Replace attack_types - find the active one and replace it
    # Pattern to match attack_types block (handles both UNSW and CICIDS)
    attack_pattern = r'(    # Attack type mapping[^\n]*\n)(.*?)(    \})'
    
    # Find where attack_types starts
    unsw_comment = '# Attack type mapping (UNSW-NB15 dataset)'
    cicids_comment = '# Attack type mapping (CICIDS2017 dataset)'
    
    # Comment out UNSW if active
    if unsw_comment in content and 'attack_types = {' in content:
        # Find and comment out the UNSW block
        unsw_section = re.search(
            r'(    # Attack type mapping \(UNSW-NB15 dataset\)[^\n]*\n)(    attack_types = \{.*?\n    \})',
            content,
            re.DOTALL
        )
        if unsw_section:
            unsw_block = unsw_section.group(0)
            commented_unsw = re.sub(
                r'^(    )(attack_types|# [A-Z])',
                r'\1# \2',
                unsw_block,
                flags=re.MULTILINE
            )
            content = content.replace(unsw_block, commented_unsw.replace('attack_types', '# attack_types', 1))
    
    # Ensure CICIDS attack_types is active (uncommented)
    cicids_attack_types = """    # Attack type mapping (CICIDS2017 dataset)
    attack_types = {
        'BENIGN': 0,
        'Bot': 1,
        'DDoS': 2,
        'DoS GoldenEye': 3,
        'DoS Hulk': 4,
        'DoS Slowhttptest': 5,
        'DoS slowloris': 6,
        'FTP-Patator': 7,
        'Heartbleed': 8,
        'Infiltration': 9,
        'PortScan': 10,
        'SSH-Patator': 11,
        'Web Attack': 12,
        'Web Attack  Brute Force': 12,
        'Web Attack  Sql Injection': 12,
        'Web Attack  XSS': 12
    }"""
    
    # Remove any commented CICIDS blocks (triple-quoted or # commented)
    content = re.sub(
        r"    '''\s*# CICIDS dataset attack types.*?'''",
        '',
        content,
        flags=re.DOTALL
    )
    content = re.sub(
        r"    # Attack type mapping \(CICIDS2017 dataset\) - COMMENTED OUT.*?    # \}",
        '',
        content,
        flags=re.DOTALL
    )
    
    # Find the @property line and insert CICIDS attack_types before it
    property_match = re.search(r'(\s+@property\s+def zero_day_attack_label)', content)
    if property_match:
        # Check if CICIDS is already there
        before_property = content[:property_match.start()]
        if "CICIDS2017 dataset" not in before_property or "# Attack type mapping (CICIDS2017 dataset)" not in before_property:
            # Insert CICIDS attack_types before @property
            content = content[:property_match.start()] + cicids_attack_types + '\n    \n' + content[property_match.start():]
    
    # Update zero_day_attack_label default
    content = re.sub(
        r'return self\.attack_types\.get\(self\.zero_day_attack, \d+\)  # Default to [^)]+\)',
        'return self.attack_types.get(self.zero_day_attack, 10)  # Default to PortScan=10 (CICIDS2017)',
        content
    )
    
    # Clean up any stray triple quotes that might comment out code
    content = re.sub(r"    '''\s*\n\s+@property", "    @property", content, flags=re.DOTALL)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Updated {config_path}")
    
    # 2. Update main.py
    main_path = "main.py"
    if not os.path.exists(main_path):
        print(f"❌ Error: {main_path} not found!")
        return False
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Comment out UNSW preprocessor (careful with multiline)
    unsw_block = """            logger.info("Initializing UNSW preprocessor...")
            from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
            self.preprocessor = UNSWPreprocessor(
                data_path=self.config.data_path,
                test_path=self.config.test_path
            )"""
    
    commented_unsw = """            # logger.info("Initializing UNSW preprocessor...")
            # from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
            # self.preprocessor = UNSWPreprocessor(
            #     data_path=self.config.data_path,
            #     test_path=self.config.test_path
            # )"""
    
    if unsw_block in content:
        content = content.replace(unsw_block, commented_unsw)
    
    # Uncomment CICIDS preprocessor
    cicids_comment_block = """            '''
            self.preprocessor = CICIDSPreprocessor(
                data_path=self.config.data_path,
                test_path=self.config.test_path
            )
            '''"""
    
    cicids_active = """            from preprocessing.blockchain_federated_cicids_preprocessor import CICIDSPreprocessor
            self.preprocessor = CICIDSPreprocessor(
                data_path=self.config.data_path,
                test_path=self.config.test_path
            )"""
    
    if cicids_comment_block in content:
        content = content.replace(cicids_comment_block, cicids_active)
    elif "# from blockchain_federated_cicids_preprocessor" not in content:
        # Add it if it doesn't exist at all
        content = content.replace(
            "            # 1. Initialize preprocessor",
            "            # 1. Initialize preprocessor\n            from preprocessing.blockchain_federated_cicids_preprocessor import CICIDSPreprocessor\n            self.preprocessor = CICIDSPreprocessor(\n                data_path=self.config.data_path,\n                test_path=self.config.test_path\n            )"
        )
    
    with open(main_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Updated {main_path}")
    
    return True


def switch_to_unsw():
    """Switch configuration to UNSW-NB15 dataset"""
    print("🔄 Switching to UNSW-NB15 dataset...\n")
    
    # 1. Update config.py
    config_path = "config.py"
    if not os.path.exists(config_path):
        print(f"❌ Error: {config_path} not found!")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update data paths
    content = re.sub(
        r'data_path: str = "[^"]*"',
        'data_path: str = "UNSW_NB15_training-set.csv"',
        content
    )
    content = re.sub(
        r'test_path: str = "[^"]*"',
        'test_path: str = "UNSW_NB15_testing-set.csv"',
        content
    )
    
    # Update zero_day_attack
    content = re.sub(
        r'zero_day_attack: str = "[^"]*"',
        'zero_day_attack: str = "Analysis"  # UNSW-NB15 attack type (label 3)',
        content
    )
    
    # Ensure UNSW attack_types is active
    unsw_attack_types = """    # Attack type mapping (UNSW-NB15 dataset)
    attack_types = {
        'Normal': 0,
        'Fuzzers': 1,
        'Analysis': 2,
        'Backdoor': 3,
        'DoS': 4,
        'Exploits': 5,
        'Generic': 6,
        'Reconnaissance': 7,
        'Shellcode': 8,
        'Worms': 9
    }"""
    
    # Comment out CICIDS attack_types
    cicids_pattern = r'(    # Attack type mapping \(CICIDS2017 dataset\)[^\n]*\n)(    attack_types = \{.*?\n    \})'
    cicids_match = re.search(cicids_pattern, content, re.DOTALL)
    if cicids_match:
        cicids_block = cicids_match.group(0)
        commented_cicids = re.sub(
            r'^(    )(attack_types|# [A-Z])',
            r'\1# \2',
            cicids_block,
            flags=re.MULTILINE
        )
        content = content.replace(cicids_block, commented_cicids.replace('attack_types', '# attack_types', 1))
    
    # Remove any triple-quoted CICIDS blocks
    content = re.sub(
        r"    '''\s*# CICIDS dataset attack types.*?'''",
        '',
        content,
        flags=re.DOTALL
    )
    
    # Find the @property line and ensure UNSW attack_types is before it
    property_match = re.search(r'(\s+@property\s+def zero_day_attack_label)', content)
    if property_match:
        before_property = content[:property_match.start()]
        # Check if UNSW is already active (not commented)
        if "# Attack type mapping (UNSW-NB15 dataset)" not in before_property or "# attack_types = {" in before_property.split('\n')[-20:]:
            # Uncomment if commented, or insert if missing
            # First, try to uncomment existing
            content = re.sub(
                r'(    # Attack type mapping \(UNSW-NB15 dataset\)[^\n]*\n)(    # attack_types = \{.*?\n    # \})',
                unsw_attack_types,
                content,
                flags=re.DOTALL
            )
            
            # If still not found, insert it
            if "UNSW-NB15 dataset" not in content or "# attack_types = {" in content.split("UNSW-NB15 dataset")[1].split("@property")[0]:
                content = content[:property_match.start()] + unsw_attack_types + '\n    \n' + content[property_match.start():]
    
    # Add commented CICIDS block for reference
    if "'''" not in content.split("@property")[0] or "CICIDS" not in content.split("@property")[0]:
        property_match = re.search(r'(\s+@property\s+def zero_day_attack_label)', content)
        if property_match:
            cicids_comment = """
    '''
    # CICIDS dataset attack types (commented out - use UNSW-NB15 instead)
    attack_types = {
        'BENIGN': 0,
        'Bot': 1,
        'DDoS': 2,
        'DoS GoldenEye': 3,
        'DoS Hulk': 4,
        'DoS Slowhttptest': 5,
        'DoS slowloris': 6,
        'FTP-Patator': 7,
        'Heartbleed': 8,
        'Infiltration': 9,
        'PortScan': 10,
        'SSH-Patator': 11,
        'Web Attack': 12,
        'Web Attack  Brute Force': 12,
        'Web Attack  Sql Injection': 12,
        'Web Attack  XSS': 12
    }
    '''
"""
            # Only add if not already present
            if "CICIDS dataset attack types" not in content.split("@property")[0]:
                content = content[:property_match.start()] + cicids_comment + content[property_match.start():]
    
    # Update zero_day_attack_label default
    content = re.sub(
        r'return self\.attack_types\.get\(self\.zero_day_attack, \d+\)  # Default to [^)]+\)',
        'return self.attack_types.get(self.zero_day_attack, 6)  # Default to Generic=6 (UNSW-NB15)',
        content
    )
    
    # Clean up any stray triple quotes
    content = re.sub(r"    '''\s*\n\s+@property", "    @property", content, flags=re.DOTALL)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Updated {config_path}")
    
    # 2. Update main.py
    main_path = "main.py"
    if not os.path.exists(main_path):
        print(f"❌ Error: {main_path} not found!")
        return False
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Uncomment UNSW preprocessor
    commented_unsw = """            # logger.info("Initializing UNSW preprocessor...")
            # from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
            # self.preprocessor = UNSWPreprocessor(
            #     data_path=self.config.data_path,
            #     test_path=self.config.test_path
            # )"""
    
    unsw_active = """            logger.info("Initializing UNSW preprocessor...")
            from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
            self.preprocessor = UNSWPreprocessor(
                data_path=self.config.data_path,
                test_path=self.config.test_path
            )"""
    
    if commented_unsw in content:
        content = content.replace(commented_unsw, unsw_active)
    
    # Comment out CICIDS preprocessor
    cicids_active = """            from preprocessing.blockchain_federated_cicids_preprocessor import CICIDSPreprocessor
            self.preprocessor = CICIDSPreprocessor(
                data_path=self.config.data_path,
                test_path=self.config.test_path
            )"""
    
    cicids_comment_block = """            '''
            self.preprocessor = CICIDSPreprocessor(
                data_path=self.config.data_path,
                test_path=self.config.test_path
            )
            '''"""
    
    if cicids_active in content:
        content = content.replace(cicids_active, cicids_comment_block)
    
    # Remove standalone CICIDS import if it exists
    content = re.sub(
        r'\s*from preprocessing\.blockchain_federated_cicids_preprocessor import CICIDSPreprocessor\n',
        '',
        content
    )
    content = re.sub(
        r'\s*from blockchain_federated_cicids_preprocessor import CICIDSPreprocessor\n',
        '',
        content
    )
    
    with open(main_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Updated {main_path}")
    
    return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("📋 Dataset Switching Script")
        print("=" * 60)
        print("Usage: python switch_dataset.py <dataset_name>")
        print("\nAvailable datasets:")
        print("  - UNSW    : Switch to UNSW-NB15 dataset")
        print("  - CICIDS  : Switch to CICIDS2017 dataset")
        print("\nExample:")
        print("  python switch_dataset.py CICIDS")
        print("  python switch_dataset.py UNSW")
        print("=" * 60)
        sys.exit(1)
    
    dataset_name = sys.argv[1].upper()
    
    print("\n" + "=" * 60)
    
    if dataset_name == "CICIDS":
        success = switch_to_cicids()
    elif dataset_name in ["UNSW", "UNSW-NB15"]:
        success = switch_to_unsw()
    else:
        print(f"❌ Error: Unknown dataset '{sys.argv[1]}'")
        print("   Use 'CICIDS' or 'UNSW'")
        sys.exit(1)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Dataset switch completed successfully!")
        print("=" * 60)
        print("\n📝 Next steps:")
        print("  1. Verify data files exist:")
        print("     - Check config.py for data_path and test_path")
        print("  2. Update zero_day_attack in config.py if needed")
        print("     - For CICIDS: PortScan, DDoS, Bot, etc.")
        print("     - For UNSW: Analysis, Exploits, Backdoor, etc.")
        print("  3. Run the system: python main.py")
        print("\n⚠️  Note: You may need to update input_dim in config.py")
        print("   after running preprocessing to match the actual feature count.\n")
    else:
        print("\n❌ Dataset switch failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
