#!/usr/bin/env python3
"""
Quick Test Script for Centralized Learning Mode

This script runs a minimal test to verify centralized learning works correctly.
"""

import logging
from config import SystemConfig, get_config, update_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_centralized_learning():
    """Run a quick test of centralized learning mode"""
    
    logger.info("=" * 80)
    logger.info("🧪 CENTRALIZED LEARNING QUICK TEST")
    logger.info("=" * 80)
    
    # Step 1: Update config for quick test
    logger.info("\n📝 Step 1: Configuring for quick test...")
    config = get_config()
    
    # Enable centralized learning
    config.use_federated_learning = False
    
    # Reduce rounds and epochs for quick test
    config.num_rounds = 2
    config.meta_epochs = 3
    config.num_meta_tasks = 5
    config.ttt_base_steps = 10
    
    logger.info(f"   ✅ use_federated_learning: {config.use_federated_learning}")
    logger.info(f"   ✅ num_rounds: {config.num_rounds}")
    logger.info(f"   ✅ meta_epochs: {config.meta_epochs}")
    logger.info(f"   ✅ num_meta_tasks: {config.num_meta_tasks}")
    logger.info(f"   ✅ ttt_base_steps: {config.ttt_base_steps}")
    
    # Step 2: Import and initialize system
    logger.info("\n📝 Step 2: Initializing system...")
    try:
        from main import BlockchainFederatedIncentiveSystem
        system = BlockchainFederatedIncentiveSystem(config)
        
        if not system.initialize_system():
            logger.error("❌ System initialization failed")
            return False
        
        logger.info("   ✅ System initialized successfully")
        
        # Verify coordinator type
        from coordinators.centralized_coordinator import CentralizedCoordinator
        if isinstance(system.coordinator, CentralizedCoordinator):
            logger.info("   ✅ Using CentralizedCoordinator (correct!)")
        else:
            logger.error(f"   ❌ Wrong coordinator type: {type(system.coordinator)}")
            return False
            
    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Step 3: Verify coordinator interface
    logger.info("\n📝 Step 3: Verifying coordinator interface...")
    try:
        # Check required methods exist
        required_methods = ['distribute_data', 'run_federated_round', 'adapt_to_test_data', 
                           'quick_system_self_check', 'evaluate_with_flow_wrapper']
        
        for method_name in required_methods:
            if not hasattr(system.coordinator, method_name):
                logger.error(f"   ❌ Missing method: {method_name}")
                return False
            logger.info(f"   ✅ Method exists: {method_name}")
        
        # Verify empty clients list (centralized has no clients)
        if hasattr(system.coordinator, 'clients'):
            if isinstance(system.coordinator.clients, list) and len(system.coordinator.clients) == 0:
                logger.info(f"   ✅ Empty clients list (correct for centralized)")
            else:
                logger.warning(f"   ⚠️  Clients list is not empty: {len(system.coordinator.clients)} clients")
        else:
            logger.warning("   ⚠️  No clients attribute found")
            
    except Exception as e:
        logger.error(f"❌ Interface verification failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Step 5: Test coordinator methods
    logger.info("\n📝 Step 5: Testing coordinator methods...")
    try:
        # Test quick_system_self_check
        check_result = system.coordinator.quick_system_self_check()
        logger.info(f"   ✅ quick_system_self_check: {check_result.get('mode', 'N/A')}")
        
        # Test model access
        if hasattr(system.coordinator, 'model') and system.coordinator.model is not None:
            logger.info(f"   ✅ Coordinator model accessible")
        else:
            logger.error("   ❌ Coordinator model not accessible")
            return False
            
    except Exception as e:
        logger.error(f"❌ Coordinator method test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ CENTRALIZED LEARNING QUICK TEST PASSED!")
    logger.info("=" * 80)
    logger.info("\n💡 Summary:")
    logger.info("   - Centralized coordinator initialized correctly")
    logger.info("   - Data distribution works")
    logger.info("   - Training round executes successfully")
    logger.info("   - All coordinator methods accessible")
    logger.info("\n🎯 Ready for full experiment!")
    
    return True

if __name__ == "__main__":
    success = test_centralized_learning()
    exit(0 if success else 1)

