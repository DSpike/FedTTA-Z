#!/usr/bin/env python3
"""
Test script to validate configuration synchronization
Run this to check if configurations are aligned
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_validator import validate_configurations, ConfigValidator
from main import EnhancedSystemConfig
from config import get_config

def main():
    print("🔍 Testing Configuration Synchronization")
    print("=" * 50)
    
    # Test the validation
    validation = validate_configurations()
    
    if validation.is_valid:
        print("\n✅ All configurations are properly synchronized!")
        print("You can safely run your federated learning system.")
    else:
        print("\n❌ Configuration issues found!")
        print("Please fix the discrepancies before running the system.")
        
        # Show how to fix
        validator = ConfigValidator()
        enhanced_config = EnhancedSystemConfig()
        sync_code = validator.generate_sync_code(enhanced_config)
        
        print("\n💡 To fix automatically, run:")
        print("python -c \"from config_validator import ConfigValidator; from main import EnhancedSystemConfig; v = ConfigValidator(); e = EnhancedSystemConfig(); v.auto_fix_enhanced_config(e); print('Fixed!')\"")
        
        print("\n📝 Or manually update the EnhancedSystemConfig class with:")
        print(sync_code)
    
    return validation.is_valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
