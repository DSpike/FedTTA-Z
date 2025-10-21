"""
Configuration Validation and Synchronization System
Prevents configuration drift between SystemConfig instances
"""

import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, fields
from config import SystemConfig, get_config

logger = logging.getLogger(__name__)

@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    discrepancies: List[Tuple[str, Any, Any]]  # (field_name, expected, actual)
    warnings: List[str]
    suggestions: List[str]

class ConfigValidator:
    """Validates and synchronizes configurations to prevent drift"""
    
    def __init__(self):
        self.central_config = get_config()
        self.critical_fields = [
            'num_clients', 'num_rounds', 'input_dim', 'learning_rate',
            'n_way', 'k_shot', 'n_query', 'use_tcn', 'sequence_length'
        ]
    
    def validate_enhanced_config(self, enhanced_config) -> ConfigValidationResult:
        """
        Validate SystemConfig against centralized SystemConfig
        
        Args:
            enhanced_config: SystemConfig instance to validate
            
        Returns:
            ConfigValidationResult with validation details
        """
        discrepancies = []
        warnings = []
        suggestions = []
        
        # Check critical fields
        for field_name in self.critical_fields:
            if hasattr(self.central_config, field_name) and hasattr(enhanced_config, field_name):
                central_value = getattr(self.central_config, field_name)
                enhanced_value = getattr(enhanced_config, field_name)
                
                if central_value != enhanced_value:
                    discrepancies.append((field_name, central_value, enhanced_value))
                    
                    # Add specific suggestions based on field type
                    if field_name in ['num_clients', 'num_rounds']:
                        suggestions.append(
                            f"Update {field_name} from {enhanced_value} to {central_value} "
                            f"for better federated learning performance"
                        )
                    elif field_name in ['k_shot', 'n_query']:
                        suggestions.append(
                            f"Update {field_name} from {enhanced_value} to {central_value} "
                            f"for consistent few-shot learning configuration"
                        )
                    elif field_name == 'input_dim':
                        suggestions.append(
                            f"Update {field_name} from {enhanced_value} to {central_value} "
                            f"to match IGRF-RFE feature selection results"
                        )
        
        # Check for missing fields in enhanced config
        central_fields = {f.name for f in fields(self.central_config)}
        enhanced_fields = {f.name for f in fields(enhanced_config)}
        missing_fields = central_fields - enhanced_fields
        
        if missing_fields:
            warnings.append(f"SystemConfig missing fields: {missing_fields}")
            suggestions.append("Consider adding missing fields to SystemConfig")
        
        # Check for extra fields in enhanced config
        extra_fields = enhanced_fields - central_fields
        if extra_fields:
            warnings.append(f"SystemConfig has extra fields: {extra_fields}")
            suggestions.append("Consider if extra fields should be in centralized config")
        
        is_valid = len(discrepancies) == 0
        
        return ConfigValidationResult(
            is_valid=is_valid,
            discrepancies=discrepancies,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def auto_fix_enhanced_config(self, enhanced_config) -> bool:
        """
        Automatically fix SystemConfig to match SystemConfig
        
        Args:
            enhanced_config: SystemConfig instance to fix
            
        Returns:
            bool: True if fixes were applied, False otherwise
        """
        try:
            fixes_applied = 0
            
            for field_name in self.critical_fields:
                if hasattr(self.central_config, field_name) and hasattr(enhanced_config, field_name):
                    central_value = getattr(self.central_config, field_name)
                    enhanced_value = getattr(enhanced_config, field_name)
                    
                    if central_value != enhanced_value:
                        setattr(enhanced_config, field_name, central_value)
                        fixes_applied += 1
                        logger.info(f"Fixed {field_name}: {enhanced_value} → {central_value}")
            
            if fixes_applied > 0:
                logger.info(f"Applied {fixes_applied} configuration fixes")
                return True
            else:
                logger.info("No configuration fixes needed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to auto-fix configuration: {str(e)}")
            return False
    
    def generate_sync_code(self, enhanced_config) -> str:
        """
        Generate code to synchronize SystemConfig with SystemConfig
        
        Args:
            enhanced_config: SystemConfig instance
            
        Returns:
            str: Python code to fix the configuration
        """
        validation = self.validate_enhanced_config(enhanced_config)
        
        if validation.is_valid:
            return "# No synchronization needed - configurations are already aligned"
        
        code_lines = ["# Auto-generated configuration synchronization code"]
        code_lines.append("from config import get_config")
        code_lines.append("")
        code_lines.append("def sync_enhanced_config(enhanced_config):")
        code_lines.append("    \"\"\"Synchronize SystemConfig with centralized config\"\"\"")
        code_lines.append("    central_config = get_config()")
        code_lines.append("")
        
        for field_name, expected_value, _ in validation.discrepancies:
            code_lines.append(f"    enhanced_config.{field_name} = central_config.{field_name}  # {expected_value}")
        
        code_lines.append("    return enhanced_config")
        
        return "\n".join(code_lines)
    
    def create_config_guard(self) -> str:
        """
        Create a configuration guard that can be added to main.py
        
        Returns:
            str: Configuration guard code
        """
        guard_code = '''
# Configuration Guard - Add this to the top of main.py
def ensure_config_sync():
    """Ensure SystemConfig is synchronized with SystemConfig"""
    from config_validator import ConfigValidator
    from config import get_config
    
    validator = ConfigValidator()
    central_config = get_config()
    
    # Create a temporary SystemConfig for validation
    temp_enhanced = SystemConfig()
    
    # Validate configuration
    validation = validator.validate_enhanced_config(temp_enhanced)
    
    if not validation.is_valid:
        logger.warning("Configuration drift detected!")
        for field, expected, actual in validation.discrepancies:
            logger.warning(f"  {field}: expected {expected}, got {actual}")
        
        # Auto-fix if possible
        if validator.auto_fix_enhanced_config(temp_enhanced):
            logger.info("Configuration automatically synchronized")
        else:
            logger.error("Manual configuration synchronization required")
            raise ValueError("Configuration drift detected - manual fix required")
    
    return True

# Call this at the start of your main function
ensure_config_sync()
'''
        return guard_code

def validate_configurations():
    """Standalone function to validate all configurations"""
    from config import SystemConfig
    
    validator = ConfigValidator()
    enhanced_config = SystemConfig()
    
    validation = validator.validate_enhanced_config(enhanced_config)
    
    print("🔍 Configuration Validation Report")
    print("=" * 50)
    
    if validation.is_valid:
        print("✅ All configurations are synchronized!")
    else:
        print("❌ Configuration discrepancies found:")
        for field, expected, actual in validation.discrepancies:
            print(f"  • {field}: expected {expected}, got {actual}")
    
    if validation.warnings:
        print("\n⚠️  Warnings:")
        for warning in validation.warnings:
            print(f"  • {warning}")
    
    if validation.suggestions:
        print("\n💡 Suggestions:")
        for suggestion in validation.suggestions:
            print(f"  • {suggestion}")
    
    return validation

if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_configurations()
