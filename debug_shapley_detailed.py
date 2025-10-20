#!/usr/bin/env python3
"""
Debug script to test Shapley value calculation with actual client performance data
"""

import sys
import os
import logging
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from incentives.shapley_value_calculator import ShapleyValueCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_shapley_detailed():
    print("🔍 DEBUGGING SHAPLEY VALUE CALCULATION IN DETAIL")
    print("=" * 60)

    # Get configuration
    config = get_config()
    print(f"📋 Configuration: {config.num_clients} clients")
    
    # Use the actual performance data from the recent run
    global_performance = 0.3554
    individual_performances = {
        'client_1': 0.594,
        'client_2': 0.496, 
        'client_3': 0.556,
        'client_4': 0.650,
        'client_5': 0.555,
        'client_6': 0.526
    }
    
    data_quality_scores = {
        'client_1': 0.8,
        'client_2': 0.75,
        'client_3': 0.85,
        'client_4': 0.9,
        'client_5': 0.8,
        'client_6': 0.7
    }
    
    participation_data = {
        'client_1': 1.0,
        'client_2': 1.0,
        'client_3': 1.0,
        'client_4': 1.0,
        'client_5': 1.0,
        'client_6': 1.0
    }
    
    print(f"📊 Input Data:")
    print(f"  Global performance: {global_performance}")
    print(f"  Individual performances: {individual_performances}")
    print(f"  Data quality scores: {data_quality_scores}")
    print(f"  Participation data: {participation_data}")
    print()
    
    # Create Shapley calculator with actual client IDs
    actual_client_ids = list(individual_performances.keys())
    calculator = ShapleyValueCalculator(
        num_clients=len(actual_client_ids),
        evaluation_metric='accuracy',
        client_ids=actual_client_ids
    )
    
    print(f"🔧 Calculator initialized with {len(actual_client_ids)} clients: {actual_client_ids}")
    print()
    
    # Test coalition performance calculation
    print("🧪 Testing coalition performance calculation...")
    for i, client_id in enumerate(actual_client_ids):
        coalition = [client_id]
        performance = calculator.calculate_coalition_performance(
            coalition, global_performance, individual_performances
        )
        print(f"  Single client {client_id}: {performance:.6f}")
    
    # Test with all clients
    all_clients = list(actual_client_ids)
    all_performance = calculator.calculate_coalition_performance(
        all_clients, global_performance, individual_performances
    )
    print(f"  All clients: {all_performance:.6f}")
    print()
    
    # Calculate Shapley values
    print("🧮 Calculating Shapley values...")
    try:
        contributions = calculator.calculate_shapley_values(
            global_performance=global_performance,
            individual_performances=individual_performances,
            client_data_quality=data_quality_scores,
            client_participation=participation_data
        )
        
        print(f"✅ Shapley calculation successful!")
        print(f"📊 Results:")
        for contrib in contributions:
            print(f"  {contrib.client_id}: {contrib.shapley_value:.6f}")
        
        # Check if values are different
        shapley_values = [contrib.shapley_value for contrib in contributions]
        unique_values = len(set(shapley_values))
        print(f"\n🔍 Analysis:")
        print(f"  Unique Shapley values: {unique_values}")
        print(f"  All values equal: {unique_values == 1}")
        print(f"  Value range: {min(shapley_values):.6f} to {max(shapley_values):.6f}")
        
        if unique_values == 1:
            print("❌ PROBLEM: All Shapley values are identical!")
            print("   This means the calculation is falling back to equal distribution")
        else:
            print("✅ SUCCESS: Shapley values are differentiated!")
            
    except Exception as e:
        print(f"❌ Error calculating Shapley values: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_shapley_detailed()
