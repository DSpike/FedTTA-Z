"""
Automatic Configuration Synchronization System
Prevents configuration drift by automatically updating EnhancedSystemConfig when SystemConfig changes
"""

import os
import sys
import logging
import ast
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, fields
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig, get_config
from config_validator import ConfigValidator

logger = logging.getLogger(__name__)

@dataclass
class ConfigChange:
    """Represents a configuration change"""
    field_name: str
    old_value: Any
    new_value: Any
    line_number: int

class AutoConfigSync:
    """Automatically synchronizes configurations to prevent drift"""
    
    def __init__(self):
        self.validator = ConfigValidator()
        self.main_py_path = Path(__file__).parent / "main.py"
        self.config_py_path = Path(__file__).parent / "config.py"
        self.backup_dir = Path(__file__).parent / "config_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Track critical fields that must be synchronized
        self.critical_fields = [
            'num_clients', 'num_rounds', 'local_epochs', 'learning_rate',
            'input_dim', 'hidden_dim', 'embedding_dim', 'num_classes',
            'n_way', 'k_shot', 'n_query', 'use_tcn', 'sequence_length',
            'sequence_stride', 'meta_epochs', 'ttt_base_steps', 'ttt_max_steps',
            'ttt_lr', 'support_weight', 'test_weight'
        ]
    
    def detect_config_changes(self) -> List[ConfigChange]:
        """Detect changes in SystemConfig that need to be synchronized"""
        changes = []
        
        try:
            # Get current SystemConfig values
            current_config = get_config()
            
            # Parse main.py to find EnhancedSystemConfig values
            with open(self.main_py_path, 'r', encoding='utf-8') as f:
                main_content = f.read()
            
            # Find EnhancedSystemConfig class definition
            enhanced_config_values = self._extract_enhanced_config_values(main_content)
            
            # Compare with current SystemConfig
            for field_name in self.critical_fields:
                if hasattr(current_config, field_name):
                    current_value = getattr(current_config, field_name)
                    enhanced_value = enhanced_config_values.get(field_name)
                    
                    if enhanced_value is not None and current_value != enhanced_value:
                        line_num = self._find_field_line_number(main_content, field_name)
                        changes.append(ConfigChange(
                            field_name=field_name,
                            old_value=enhanced_value,
                            new_value=current_value,
                            line_number=line_num
                        ))
            
            return changes
            
        except Exception as e:
            logger.error(f"Failed to detect config changes: {str(e)}")
            return []
    
    def _extract_enhanced_config_values(self, content: str) -> Dict[str, Any]:
        """Extract field values from EnhancedSystemConfig class"""
        values = {}
        
        # Find the EnhancedSystemConfig class
        class_match = re.search(r'class EnhancedSystemConfig:.*?(?=class|\Z)', content, re.DOTALL)
        if not class_match:
            return values
        
        class_content = class_match.group(0)
        
        # Extract field assignments
        for field_name in self.critical_fields:
            pattern = rf'{field_name}:\s*\w+\s*=\s*([^#\n]+)'
            match = re.search(pattern, class_content)
            if match:
                value_str = match.group(1).strip()
                try:
                    # Try to evaluate the value
                    value = ast.literal_eval(value_str)
                    values[field_name] = value
                except (ValueError, SyntaxError):
                    # If evaluation fails, store as string
                    values[field_name] = value_str.strip('"\'')
        
        return values
    
    def _find_field_line_number(self, content: str, field_name: str) -> int:
        """Find the line number of a field in EnhancedSystemConfig"""
        lines = content.split('\n')
        in_enhanced_config = False
        
        for i, line in enumerate(lines, 1):
            if 'class EnhancedSystemConfig:' in line:
                in_enhanced_config = True
                continue
            
            if in_enhanced_config and line.strip().startswith('class '):
                break
            
            if in_enhanced_config and f'{field_name}:' in line:
                return i
        
        return 0
    
    def auto_sync_configurations(self, dry_run: bool = False) -> bool:
        """
        Automatically synchronize EnhancedSystemConfig with SystemConfig
        
        Args:
            dry_run: If True, only show what would be changed without making changes
            
        Returns:
            bool: True if synchronization was successful or no changes needed
        """
        try:
            changes = self.detect_config_changes()
            
            if not changes:
                logger.info("✅ No configuration changes detected - everything is synchronized")
                return True
            
            logger.info(f"🔍 Found {len(changes)} configuration changes that need synchronization")
            
            if dry_run:
                logger.info("🔍 DRY RUN - Changes that would be made:")
                for change in changes:
                    logger.info(f"  • {change.field_name}: {change.old_value} → {change.new_value} (line {change.line_number})")
                return True
            
            # Create backup before making changes
            self._create_backup()
            
            # Apply changes to main.py
            success = self._apply_changes_to_main_py(changes)
            
            if success:
                logger.info("✅ Configuration synchronization completed successfully")
                return True
            else:
                logger.error("❌ Failed to synchronize configurations")
                return False
                
        except Exception as e:
            logger.error(f"❌ Auto-sync failed: {str(e)}")
            return False
    
    def _create_backup(self):
        """Create backup of main.py before making changes"""
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"main_py_backup_{timestamp}.py"
        shutil.copy2(self.main_py_path, backup_path)
        logger.info(f"📁 Created backup: {backup_path}")
    
    def _apply_changes_to_main_py(self, changes: List[ConfigChange]) -> bool:
        """Apply configuration changes to main.py"""
        try:
            with open(self.main_py_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Apply changes in reverse order to maintain line numbers
            for change in sorted(changes, key=lambda x: x.line_number, reverse=True):
                if change.line_number > 0 and change.line_number <= len(lines):
                    line_index = change.line_number - 1
                    old_line = lines[line_index]
                    
                    # Update the line with new value
                    new_line = re.sub(
                        rf'({change.field_name}:\s*\w+\s*=\s*)[^#\n]+',
                        rf'\g<1>{change.new_value}',
                        old_line
                    )
                    
                    if new_line != old_line:
                        lines[line_index] = new_line
                        logger.info(f"✅ Updated {change.field_name}: {change.old_value} → {change.new_value}")
                    else:
                        logger.warning(f"⚠️ Could not update {change.field_name} - pattern not found")
            
            # Write updated content
            with open(self.main_py_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply changes to main.py: {str(e)}")
            return False
    
    def setup_file_watcher(self):
        """Set up file watcher to automatically sync when config.py changes"""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class ConfigChangeHandler(FileSystemEventHandler):
                def __init__(self, auto_sync):
                    self.auto_sync = auto_sync
                
                def on_modified(self, event):
                    if event.src_path.endswith('config.py'):
                        logger.info("🔍 config.py changed - checking for synchronization needs")
                        self.auto_sync.auto_sync_configurations()
            
            event_handler = ConfigChangeHandler(self)
            observer = Observer()
            observer.schedule(event_handler, path=str(self.config_py_path.parent), recursive=False)
            observer.start()
            
            logger.info("👀 File watcher started - will auto-sync when config.py changes")
            return observer
            
        except ImportError:
            logger.warning("⚠️ watchdog not installed - file watcher not available")
            logger.info("💡 Install with: pip install watchdog")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to setup file watcher: {str(e)}")
            return None
    
    def create_pre_commit_hook(self):
        """Create a pre-commit hook to validate configurations"""
        hook_content = '''#!/bin/bash
# Pre-commit hook for configuration validation

echo "🔍 Validating configurations before commit..."

# Run configuration validation
python test_config_sync.py

if [ $? -ne 0 ]; then
    echo "❌ Configuration validation failed - commit blocked"
    echo "💡 Run 'python auto_config_sync.py' to fix configurations"
    exit 1
fi

echo "✅ Configuration validation passed"
exit 0
'''
        
        hook_path = Path(__file__).parent / ".git" / "hooks" / "pre-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(hook_path, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        os.chmod(hook_path, 0o755)
        logger.info(f"✅ Created pre-commit hook: {hook_path}")
    
    def create_ci_validation(self):
        """Create CI validation script"""
        ci_script = '''#!/bin/bash
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
'''
        
        ci_path = Path(__file__).parent / "validate_configs_ci.sh"
        with open(ci_path, 'w') as f:
            f.write(ci_script)
        
        os.chmod(ci_path, 0o755)
        logger.info(f"✅ Created CI validation script: {ci_path}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automatic Configuration Synchronization")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--watch", action="store_true", help="Start file watcher for automatic sync")
    parser.add_argument("--setup-hooks", action="store_true", help="Setup pre-commit hooks and CI validation")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    auto_sync = AutoConfigSync()
    
    if args.setup_hooks:
        print("🔧 Setting up configuration management hooks...")
        auto_sync.create_pre_commit_hook()
        auto_sync.create_ci_validation()
        print("✅ Hooks and CI validation setup complete!")
        return
    
    if args.watch:
        print("👀 Starting file watcher for automatic configuration sync...")
        observer = auto_sync.setup_file_watcher()
        if observer:
            try:
                observer.join()
            except KeyboardInterrupt:
                observer.stop()
                print("👋 File watcher stopped")
        return
    
    # Run synchronization
    success = auto_sync.auto_sync_configurations(dry_run=args.dry_run)
    
    if success:
        print("✅ Configuration synchronization completed successfully")
    else:
        print("❌ Configuration synchronization failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
