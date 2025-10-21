#!/usr/bin/env python3
"""
Debug script to check the actual incentive data structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BlockchainFederatedIncentiveSystem, EnhancedSystemConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_incentive_data():
    """Debug the incentive data structure"""
    logger.info("Debugging incentive data structure...")
    
    # Create a minimal config
    config = EnhancedSystemConfig(
        data_path="data/UNSW_NB15_training-set.csv",
        test_path="data/UNSW_NB15_testing-set.csv",
        num_clients=3,
        num_rounds=2,
        device="cpu"
    )
    
    # Initialize the system
    system = BlockchainFederatedIncentiveSystem(config)
    
    # Check incentive history structure
    logger.info(f"Incentive history length: {len(system.incentive_history)}")
    for i, record in enumerate(system.incentive_history):
        logger.info(f"Record {i}: {record}")
    
    # Check incentive manager
    if system.incentive_manager:
        logger.info("Incentive manager available")
        logger.info(f"Incentive manager type: {type(system.incentive_manager)}")
        
        # Check if it has get_round_summary method
        if hasattr(system.incentive_manager, 'get_round_summary'):
            logger.info("Has get_round_summary method")
            # Try to get a round summary
            for i, record in enumerate(system.incentive_history):
                try:
                    round_summary = system.incentive_manager.get_round_summary(record['round_number'])
                    logger.info(f"Round {record['round_number']} summary: {round_summary}")
                except Exception as e:
                    logger.error(f"Error getting round {record['round_number']} summary: {e}")
        else:
            logger.warning("No get_round_summary method")
    else:
        logger.warning("No incentive manager available")
    
    # Check incentive summary
    try:
        summary = system.get_incentive_summary()
        logger.info(f"Incentive summary: {summary}")
    except Exception as e:
        logger.error(f"Error getting incentive summary: {e}")

if __name__ == "__main__":
    debug_incentive_data()









