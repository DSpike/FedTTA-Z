"""
Configuration Monitoring and Auto-Fix System
Continuously monitors for configuration drift and automatically fixes it
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from config_validator import ConfigValidator
from auto_config_sync import AutoConfigSync

logger = logging.getLogger(__name__)

class ConfigMonitor:
    """Monitors configuration drift and automatically fixes it"""
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize configuration monitor
        
        Args:
            check_interval: Seconds between configuration checks
        """
        self.check_interval = check_interval
        self.validator = ConfigValidator()
        self.auto_sync = AutoConfigSync()
        self.running = False
        self.monitor_thread = None
        self.last_check = None
        self.drift_count = 0
        self.auto_fix_count = 0
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'drift_detected': 0,
            'auto_fixes_applied': 0,
            'manual_fixes_required': 0,
            'last_drift_time': None,
            'last_fix_time': None
        }
    
    def start_monitoring(self):
        """Start the configuration monitoring in a separate thread"""
        if self.running:
            logger.warning("⚠️ Configuration monitor is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"🔍 Configuration monitor started (checking every {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop the configuration monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Configuration monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_configuration()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"❌ Error in configuration monitor: {str(e)}")
                time.sleep(self.check_interval)
    
    def _check_configuration(self):
        """Check for configuration drift and fix if needed"""
        try:
            self.stats['total_checks'] += 1
            self.last_check = datetime.now()
            
            # Import here to avoid circular imports
            from main import EnhancedSystemConfig
            
            # Create temporary instance for validation
            enhanced_config = EnhancedSystemConfig()
            validation = self.validator.validate_enhanced_config(enhanced_config)
            
            if not validation.is_valid:
                self.stats['drift_detected'] += 1
                self.stats['last_drift_time'] = datetime.now()
                self.drift_count += 1
                
                logger.warning(f"⚠️ Configuration drift detected (check #{self.stats['total_checks']})")
                for field, expected, actual in validation.discrepancies:
                    logger.warning(f"  • {field}: expected {expected}, got {actual}")
                
                # Attempt automatic fix
                if self.auto_sync.auto_sync_configurations():
                    self.stats['auto_fixes_applied'] += 1
                    self.stats['last_fix_time'] = datetime.now()
                    self.auto_fix_count += 1
                    logger.info("✅ Configuration automatically synchronized")
                else:
                    self.stats['manual_fixes_required'] += 1
                    logger.error("❌ Manual configuration synchronization required")
                    self._send_alert("Configuration drift requires manual fix")
            else:
                # Reset drift count if configuration is valid
                if self.drift_count > 0:
                    logger.info("✅ Configuration drift resolved")
                    self.drift_count = 0
                
        except Exception as e:
            logger.error(f"❌ Configuration check failed: {str(e)}")
    
    def _send_alert(self, message: str):
        """Send alert about configuration issues"""
        # This could be extended to send emails, Slack messages, etc.
        logger.error(f"🚨 ALERT: {message}")
        
        # For now, just log to a file
        alert_file = Path(__file__).parent / "config_alerts.log"
        with open(alert_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics"""
        return {
            'running': self.running,
            'check_interval': self.check_interval,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'drift_count': self.drift_count,
            'auto_fix_count': self.auto_fix_count,
            'statistics': self.stats.copy()
        }
    
    def force_check(self) -> bool:
        """Force an immediate configuration check"""
        try:
            self._check_configuration()
            return True
        except Exception as e:
            logger.error(f"❌ Force check failed: {str(e)}")
            return False
    
    def reset_statistics(self):
        """Reset monitoring statistics"""
        self.stats = {
            'total_checks': 0,
            'drift_detected': 0,
            'auto_fixes_applied': 0,
            'manual_fixes_required': 0,
            'last_drift_time': None,
            'last_fix_time': None
        }
        self.drift_count = 0
        self.auto_fix_count = 0
        logger.info("📊 Monitoring statistics reset")

class ConfigGuard:
    """Configuration guard that prevents system startup with invalid configurations"""
    
    def __init__(self):
        self.validator = ConfigValidator()
        self.auto_sync = AutoConfigSync()
    
    def ensure_config_validity(self) -> bool:
        """
        Ensure configurations are valid before system startup
        
        Returns:
            bool: True if configurations are valid, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from main import EnhancedSystemConfig
            
            enhanced_config = EnhancedSystemConfig()
            validation = self.validator.validate_enhanced_config(enhanced_config)
            
            if not validation.is_valid:
                logger.error("❌ Configuration validation failed at startup")
                logger.error("Configuration discrepancies found:")
                for field, expected, actual in validation.discrepancies:
                    logger.error(f"  • {field}: expected {expected}, got {actual}")
                
                # Attempt automatic fix
                logger.info("🔧 Attempting automatic configuration synchronization...")
                if self.auto_sync.auto_sync_configurations():
                    logger.info("✅ Configuration automatically synchronized")
                    return True
                else:
                    logger.error("❌ Automatic synchronization failed - manual fix required")
                    return False
            else:
                logger.info("✅ Configuration validation passed at startup")
                return True
                
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {str(e)}")
            return False

def create_startup_guard():
    """Create a startup guard that can be imported and used"""
    guard_code = '''
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
'''
    
    guard_file = Path(__file__).parent / "startup_guard.py"
    with open(guard_file, 'w') as f:
        f.write(guard_code)
    
    logger.info(f"✅ Created startup guard: {guard_file}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Monitoring System")
    parser.add_argument("--start", action="store_true", help="Start configuration monitoring")
    parser.add_argument("--stop", action="store_true", help="Stop configuration monitoring")
    parser.add_argument("--status", action="store_true", help="Show monitoring status")
    parser.add_argument("--check", action="store_true", help="Force immediate configuration check")
    parser.add_argument("--reset-stats", action="store_true", help="Reset monitoring statistics")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--create-guard", action="store_true", help="Create startup guard")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.create_guard:
        create_startup_guard()
        return
    
    monitor = ConfigMonitor(check_interval=args.interval)
    
    if args.start:
        monitor.start_monitoring()
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    
    elif args.stop:
        monitor.stop_monitoring()
    
    elif args.status:
        status = monitor.get_status()
        print("🔍 Configuration Monitor Status")
        print("=" * 40)
        print(f"Running: {status['running']}")
        print(f"Check Interval: {status['check_interval']}s")
        print(f"Last Check: {status['last_check']}")
        print(f"Current Drift Count: {status['drift_count']}")
        print(f"Auto Fix Count: {status['auto_fix_count']}")
        print("\n📊 Statistics:")
        for key, value in status['statistics'].items():
            print(f"  {key}: {value}")
    
    elif args.check:
        success = monitor.force_check()
        if success:
            print("✅ Configuration check completed")
        else:
            print("❌ Configuration check failed")
            sys.exit(1)
    
    elif args.reset_stats:
        monitor.reset_statistics()
        print("📊 Statistics reset")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
