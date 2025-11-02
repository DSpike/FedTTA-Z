#!/usr/bin/env python3
"""Quick verification script to check TTT adaptation query size configuration"""

import sys
sys.path.insert(0, '.')

from config import SystemConfig

# Load config
config = SystemConfig()

print("=" * 60)
print("TTT Configuration Verification")
print("=" * 60)
print(f"ttt_adaptation_query_size: {config.ttt_adaptation_query_size}")
print(f"Expected: 750 (increased from 200)")
print(f"Status: {'✅ CORRECT' if config.ttt_adaptation_query_size >= 500 else '❌ TOO LOW'}")
print("=" * 60)

# Check if it's accessible via getattr
value = getattr(config, 'ttt_adaptation_query_size', 200)
print(f"getattr() test: {value}")
print(f"Status: {'✅ ACCESSIBLE' if value >= 500 else '❌ NOT ACCESSIBLE'}")
print("=" * 60)


