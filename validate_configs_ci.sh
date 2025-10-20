#!/bin/bash
# CI Configuration Validation Script

echo "🔍 Running configuration validation in CI..."

# Install dependencies if needed
pip install -r requirements.txt

# Run configuration validation
python test_config_sync.py

if [ $? -ne 0 ]; then
    echo "❌ Configuration validation failed in CI"
    echo "💡 Check configuration synchronization"
    exit 1
fi

echo "✅ Configuration validation passed in CI"
exit 0
