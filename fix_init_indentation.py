#!/usr/bin/env python3
"""
Fix indentation for the __init__ method
"""

import re

def fix_init_indentation():
    """Fix indentation for the __init__ method"""
    
    print("🔧 Fixing __init__ method indentation...")
    
    # Read the file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the __init__ method specifically
    pattern = r'def __init__\(self, config: EnhancedSystemConfig\):.*?(?=\n    def|\n\n    def|\Z)'
    
    replacement = '''def __init__(self, config: EnhancedSystemConfig):
        """
        Initialize the enhanced blockchain federated learning system with incentives.
        
        Args:
            config: Enhanced system configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # GPU Memory Management
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            # Set memory fraction to allow the system to complete
            torch.cuda.set_per_process_memory_fraction(0.2)
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")

        logger.info(f"Initializing Enhanced Blockchain Federated Learning System with Incentives")
        
        # Initialize system components
        self.model = None
        self.coordinator = None
        self.blockchain_integration = None
        self.incentive_manager = None
        self.visualizer = None
        self.training_history = []
        self.incentive_history = []
        self.client_addresses = {}
        self.is_initialized = False
        
        # Initialize the system
        if not self._initialize_system():
            raise RuntimeError("Failed to initialize the enhanced system")
'''
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed __init__ method indentation!")

if __name__ == "__main__":
    fix_init_indentation()









