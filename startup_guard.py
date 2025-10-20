
# Configuration Startup Guard
# Add this to the top of your main.py or any startup script

from config_monitor import ConfigGuard

def ensure_system_ready():
    """Ensure system is ready to start with valid configurations"""
    guard = ConfigGuard()
    if not guard.ensure_config_validity():
        print("❌ System startup blocked due to configuration issues")
        print("💡 Run 'python auto_config_sync.py' to fix configurations")
        sys.exit(1)
    print("✅ System ready - configurations validated")

# Call this at the very beginning of your main function
ensure_system_ready()
