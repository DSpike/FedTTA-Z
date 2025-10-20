#!/usr/bin/env python3
"""
Comprehensive Configuration Management System
Prevents configuration drift and ensures consistency across all configuration files
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from config_validator import ConfigValidator
from auto_config_sync import AutoConfigSync
from config_monitor import ConfigMonitor, ConfigGuard

logger = logging.getLogger(__name__)

class ConfigManager:
    """Comprehensive configuration management system"""
    
    def __init__(self):
        self.validator = ConfigValidator()
        self.auto_sync = AutoConfigSync()
        self.monitor = ConfigMonitor()
        self.guard = ConfigGuard()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('config_management.log')
            ]
        )
    
    def validate_all_configs(self) -> bool:
        """Validate all configurations and show detailed report"""
        print("🔍 Comprehensive Configuration Validation")
        print("=" * 50)
        
        try:
            # Import here to avoid circular imports
            from main import EnhancedSystemConfig
            
            enhanced_config = EnhancedSystemConfig()
            validation = self.validator.validate_enhanced_config(enhanced_config)
            
            print(f"📊 Validation Results:")
            print(f"  • Valid: {validation.is_valid}")
            print(f"  • Discrepancies: {len(validation.discrepancies)}")
            print(f"  • Warnings: {len(validation.warnings)}")
            print(f"  • Suggestions: {len(validation.suggestions)}")
            
            if validation.discrepancies:
                print(f"\n❌ Configuration Discrepancies:")
                for field, expected, actual in validation.discrepancies:
                    print(f"  • {field}: expected {expected}, got {actual}")
            
            if validation.warnings:
                print(f"\n⚠️  Warnings:")
                for warning in validation.warnings:
                    print(f"  • {warning}")
            
            if validation.suggestions:
                print(f"\n💡 Suggestions:")
                for suggestion in validation.suggestions:
                    print(f"  • {suggestion}")
            
            return validation.is_valid
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return False
    
    def auto_fix_configs(self, dry_run: bool = False) -> bool:
        """Automatically fix configuration discrepancies"""
        print("🔧 Automatic Configuration Synchronization")
        print("=" * 50)
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        
        success = self.auto_sync.auto_sync_configurations(dry_run=dry_run)
        
        if success:
            print("✅ Configuration synchronization completed successfully")
        else:
            print("❌ Configuration synchronization failed")
        
        return success
    
    def setup_protection_system(self):
        """Setup comprehensive protection system"""
        print("🛡️ Setting Up Configuration Protection System")
        print("=" * 50)
        
        try:
            # 1. Create pre-commit hook
            print("1. Creating pre-commit hook...")
            self.auto_sync.create_pre_commit_hook()
            
            # 2. Create CI validation
            print("2. Creating CI validation script...")
            self.auto_sync.create_ci_validation()
            
            # 3. Create startup guard
            print("3. Creating startup guard...")
            from config_monitor import create_startup_guard
            create_startup_guard()
            
            # 4. Create configuration backup
            print("4. Creating configuration backup...")
            self._create_config_backup()
            
            print("✅ Protection system setup complete!")
            print("\n📋 What was created:")
            print("  • Pre-commit hook: .git/hooks/pre-commit")
            print("  • CI validation: validate_configs_ci.sh")
            print("  • Startup guard: startup_guard.py")
            print("  • Configuration backup: config_backups/")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup protection system: {str(e)}")
            return False
        
        return True
    
    def _create_config_backup(self):
        """Create backup of current configurations"""
        backup_dir = Path(__file__).parent / "config_backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup config.py
        config_backup = backup_dir / f"config_{timestamp}.py"
        with open("config.py", 'r') as src, open(config_backup, 'w') as dst:
            dst.write(src.read())
        
        # Backup main.py (EnhancedSystemConfig part)
        main_backup = backup_dir / f"main_enhanced_config_{timestamp}.py"
        with open("main.py", 'r') as src:
            content = src.read()
            # Extract EnhancedSystemConfig class
            import re
            class_match = re.search(r'class EnhancedSystemConfig:.*?(?=class|\Z)', content, re.DOTALL)
            if class_match:
                with open(main_backup, 'w') as dst:
                    dst.write(class_match.group(0))
        
        logger.info(f"📁 Configuration backup created: {backup_dir}")
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous configuration monitoring"""
        print(f"👀 Starting Configuration Monitoring (interval: {interval}s)")
        print("=" * 50)
        
        self.monitor = ConfigMonitor(check_interval=interval)
        self.monitor.start_monitoring()
        
        print("✅ Configuration monitoring started")
        print("💡 Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self.monitor.stop_monitoring()
            print("\n👋 Configuration monitoring stopped")
    
    def show_status(self):
        """Show current configuration status"""
        print("📊 Configuration Management Status")
        print("=" * 50)
        
        # Validate configurations
        is_valid = self.validate_all_configs()
        
        # Show monitoring status
        if hasattr(self, 'monitor') and self.monitor.running:
            status = self.monitor.get_status()
            print(f"\n👀 Monitoring Status:")
            print(f"  • Running: {status['running']}")
            print(f"  • Check Interval: {status['check_interval']}s")
            print(f"  • Last Check: {status['last_check']}")
            print(f"  • Drift Count: {status['drift_count']}")
            print(f"  • Auto Fix Count: {status['auto_fix_count']}")
        
        # Show system readiness
        print(f"\n🚀 System Readiness:")
        if is_valid:
            print("  ✅ System ready to run")
        else:
            print("  ❌ System not ready - configuration issues detected")
            print("  💡 Run 'python manage_configs.py --fix' to resolve issues")
    
    def create_config_report(self):
        """Create detailed configuration report"""
        report_file = f"config_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Configuration Management Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # SystemConfig values
            f.write("## SystemConfig Values\n\n")
            config = get_config()
            for field in config.__dataclass_fields__:
                value = getattr(config, field)
                f.write(f"- `{field}`: `{value}`\n")
            
            # EnhancedSystemConfig values
            f.write("\n## EnhancedSystemConfig Values\n\n")
            try:
                from main import EnhancedSystemConfig
                enhanced = EnhancedSystemConfig()
                for field in enhanced.__dataclass_fields__:
                    value = getattr(enhanced, field)
                    f.write(f"- `{field}`: `{value}`\n")
            except Exception as e:
                f.write(f"Error reading EnhancedSystemConfig: {str(e)}\n")
            
            # Validation results
            f.write("\n## Validation Results\n\n")
            is_valid = self.validate_all_configs()
            f.write(f"- Valid: {is_valid}\n")
        
        print(f"📄 Configuration report created: {report_file}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Configuration Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_configs.py --validate          # Validate all configurations
  python manage_configs.py --fix              # Auto-fix configuration issues
  python manage_configs.py --fix --dry-run    # Show what would be fixed
  python manage_configs.py --setup            # Setup protection system
  python manage_configs.py --monitor          # Start monitoring
  python manage_configs.py --status           # Show current status
  python manage_configs.py --report           # Create configuration report
        """
    )
    
    parser.add_argument("--validate", action="store_true", help="Validate all configurations")
    parser.add_argument("--fix", action="store_true", help="Auto-fix configuration issues")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--setup", action="store_true", help="Setup protection system")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--report", action="store_true", help="Create configuration report")
    
    args = parser.parse_args()
    
    # Create configuration manager
    manager = ConfigManager()
    
    if args.validate:
        success = manager.validate_all_configs()
        sys.exit(0 if success else 1)
    
    elif args.fix:
        success = manager.auto_fix_configs(dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    
    elif args.setup:
        success = manager.setup_protection_system()
        sys.exit(0 if success else 1)
    
    elif args.monitor:
        manager.start_monitoring(interval=args.interval)
    
    elif args.status:
        manager.show_status()
    
    elif args.report:
        manager.create_config_report()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
