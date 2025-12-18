#!/usr/bin/env python3
"""
Quick verification script to check if grouping is enabled
"""
import sys
from config_loader import get_dataset_config

# Test CICIDS2017 config
print("=" * 80)
print("VERIFYING GROUPING CONFIGURATION")
print("=" * 80)

try:
    # Force CICIDS2017
    config = get_dataset_config('CICIDS2017')
    
    print(f"\n✅ Configuration loaded successfully")
    print(f"\n📊 Key Settings:")
    print(f"   Dataset: CICIDS2017")
    print(f"   use_category_grouping: {config.use_category_grouping}")
    print(f"   zero_day_attack: {config.zero_day_attack}")
    print(f"   zero_day_attack_label: {config.zero_day_attack_label}")
    print(f"   zero_day_category: {config.zero_day_category}")
    
    print(f"\n📋 Category Mappings:")
    if config.attack_category_mapping:
        print(f"   Total mappings: {len(config.attack_category_mapping)}")
        # Show zero-day category mappings
        zero_day_category = config.zero_day_category
        zero_day_attacks = [k for k, v in config.attack_category_mapping.items() if v == zero_day_category]
        print(f"   {zero_day_category} category (zero-day) includes: {zero_day_attacks}")
    else:
        print("   ⚠️  No attack_category_mapping found!")
    
    print(f"\n🏷️  Category Types:")
    if config.category_types:
        print(f"   Categories: {config.category_types}")
    else:
        print("   ⚠️  No category_types found!")
    
    print(f"\n🔍 Verification:")
    if config.use_category_grouping:
        print("   ✅ GROUPING IS ENABLED")
        if config.attack_category_mapping and config.category_types:
            print("   ✅ Category mappings are available")
            if config.zero_day_attack in config.category_types:
                print(f"   ✅ Zero-day attack '{config.zero_day_attack}' is a valid category")
                print(f"   ✅ Zero-day label: {config.zero_day_attack_label}")
            else:
                category = config.attack_category_mapping.get(config.zero_day_attack)
                if category:
                    print(f"   ✅ Zero-day attack '{config.zero_day_attack}' maps to category '{category}'")
                    print(f"   ✅ Zero-day label: {config.zero_day_attack_label}")
                else:
                    print(f"   ⚠️  Zero-day attack '{config.zero_day_attack}' not found in mappings!")
        else:
            print("   ⚠️  Category mappings are NOT available!")
    else:
        print("   ❌ GROUPING IS DISABLED")
        print("   ⚠️  System will use fine-grained attack labels")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

