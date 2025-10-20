#!/usr/bin/env python3
"""
Fix Token Distribution Tracking Issue
This script fixes the problem where individual token rewards are not being properly tracked
"""

import sys
import os
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BlockchainFederatedIncentiveSystem
from config import get_config

logger = logging.getLogger(__name__)

def fix_token_distribution_tracking():
    """Fix the token distribution tracking issue"""
    
    print("🔧 Fixing Token Distribution Tracking Issue")
    print("=" * 50)
    
    try:
        # Initialize the system
        config = get_config()
        system = BlockchainFederatedIncentiveSystem(config)
        
        print("✅ System initialized successfully")
        
        # Check current incentive history
        if hasattr(system, 'incentive_history') and system.incentive_history:
            print(f"📊 Found {len(system.incentive_history)} incentive records")
            
            # Analyze the records
            for i, record in enumerate(system.incentive_history):
                print(f"\n📋 Record {i+1}:")
                print(f"   Round: {record.get('round_number', 'N/A')}")
                print(f"   Total Rewards: {record.get('total_rewards', 0)}")
                print(f"   Individual Rewards: {record.get('individual_rewards', {})}")
                
                # Check if individual_rewards is missing or empty
                if 'individual_rewards' not in record or not record['individual_rewards']:
                    print("   ❌ Missing individual rewards data!")
                    
                    # Try to reconstruct from Shapley values if available
                    if 'shapley_values' in record:
                        print("   🔧 Attempting to reconstruct from Shapley values...")
                        shapley_values = record['shapley_values']
                        total_rewards = record.get('total_rewards', 0)
                        
                        # Reconstruct individual rewards
                        individual_rewards = {}
                        for client_id, shapley_value in shapley_values.items():
                            # Calculate token amount based on Shapley value
                            token_amount = int(total_rewards * shapley_value)
                            individual_rewards[client_id] = token_amount
                        
                        # Update the record
                        record['individual_rewards'] = individual_rewards
                        print(f"   ✅ Reconstructed individual rewards: {individual_rewards}")
                    else:
                        print("   ⚠️ No Shapley values available for reconstruction")
                else:
                    print("   ✅ Individual rewards data present")
        
        else:
            print("❌ No incentive history found")
            return False
        
        # Test the fix by running a quick simulation
        print("\n🧪 Testing Token Distribution Visualization")
        print("-" * 40)
        
        # Get incentive summary
        incentive_summary = system.get_incentive_summary()
        
        print(f"📊 Incentive Summary:")
        print(f"   Total Rounds: {incentive_summary['total_rounds']}")
        print(f"   Total Rewards: {incentive_summary['total_rewards_distributed']}")
        print(f"   Participant Rewards: {incentive_summary['participant_rewards']}")
        
        # Check if participant rewards are now available
        if incentive_summary['participant_rewards']:
            print("✅ SUCCESS: Individual rewards are now available!")
            
            # Test visualization
            try:
                from visualization.performance_visualization import PerformanceVisualizer
                visualizer = PerformanceVisualizer(output_dir=".", attack_name="Test")
                
                # Generate token distribution plot
                plot_path = visualizer.plot_token_distribution(incentive_summary, save=True)
                if plot_path:
                    print(f"✅ Token distribution plot generated: {plot_path}")
                else:
                    print("❌ Failed to generate token distribution plot")
                    
            except Exception as e:
                print(f"❌ Error generating visualization: {str(e)}")
        else:
            print("❌ Individual rewards still not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing token distribution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_enhanced_incentive_tracking():
    """Create enhanced incentive tracking system"""
    
    print("\n🔧 Creating Enhanced Incentive Tracking System")
    print("=" * 50)
    
    # Create a patch for the main.py file
    patch_code = '''
# Enhanced Incentive Tracking Patch
def _enhanced_process_round_incentives(self, round_num: int, round_results: Dict, 
                                     previous_accuracy: float, current_accuracy: float):
    """Enhanced version with better individual rewards tracking"""
    
    try:
        if not hasattr(self, 'incentive_manager') or not self.incentive_manager:
            logger.warning("Incentive manager not available")
            return
        
        # Calculate contribution scores
        contribution_scores = self._calculate_contribution_scores(round_results)
        
        # Calculate Shapley values
        shapley_values = self._calculate_shapley_values(round_results, contribution_scores)
        
        # Create client contributions
        client_contributions = []
        for client_id, score in contribution_scores.items():
            if client_id in shapley_values:
                client_contributions.append({
                    'client_id': client_id,
                    'contribution_score': score,
                    'shapley_value': shapley_values[client_id],
                    'accuracy_improvement': round_results.get('accuracy_improvement', 0),
                    'data_quality': round_results.get('data_quality', 0),
                    'reliability': round_results.get('reliability', 0)
                })
        
        # Process contributions and distribute rewards
        if client_contributions:
            reward_distributions = self.incentive_manager.process_round_contributions(
                round_num, client_contributions, shapley_values=shapley_values
            )
            
            if reward_distributions:
                success = self.incentive_manager.distribute_rewards(round_num, reward_distributions)
                
                if success:
                    total_tokens = sum(rd.token_amount for rd in reward_distributions)
                    
                    # Enhanced individual rewards tracking
                    individual_rewards = {}
                    for reward_dist in reward_distributions:
                        individual_rewards[reward_dist.recipient_address] = reward_dist.token_amount
                    
                    # Store comprehensive incentive record
                    incentive_record = {
                        'round_number': round_num,
                        'total_rewards': total_tokens,
                        'num_rewards': len(reward_distributions),
                        'individual_rewards': individual_rewards,
                        'shapley_values': shapley_values,
                        'contribution_scores': contribution_scores,
                        'timestamp': time.time()
                    }
                    
                    # Thread-safe storage
                    with self.lock:
                        self.incentive_history.append(incentive_record)
                    
                    logger.info(f"✅ Enhanced incentive tracking for round {round_num}: {len(reward_distributions)} rewards, Total: {total_tokens} tokens")
                    logger.info(f"📊 Individual rewards: {individual_rewards}")
                else:
                    logger.error(f"❌ Failed to distribute rewards for round {round_num}")
            else:
                logger.warning(f"No reward distributions for round {round_num}")
        else:
            logger.warning(f"No client contributions for round {round_num}")
            
    except Exception as e:
        logger.error(f"Error in enhanced incentive processing: {str(e)}")
        import traceback
        traceback.print_exc()
'''
    
    # Save the patch
    with open('incentive_tracking_patch.py', 'w') as f:
        f.write(patch_code)
    
    print("✅ Enhanced incentive tracking patch created: incentive_tracking_patch.py")
    print("💡 This patch can be applied to improve individual rewards tracking")

def main():
    """Main function"""
    print("🚀 Token Distribution Fix Tool")
    print("=" * 50)
    
    # Fix the tracking issue
    success = fix_token_distribution_tracking()
    
    if success:
        print("\n✅ Token distribution tracking fixed successfully!")
        print("🎯 Individual rewards should now be properly tracked")
        print("📊 Token distribution visualizations should work correctly")
    else:
        print("\n❌ Failed to fix token distribution tracking")
        print("💡 Try running the enhanced tracking patch")
        
        # Create enhanced tracking system
        create_enhanced_incentive_tracking()
    
    print("\n🔧 Next Steps:")
    print("1. Run your main system: python main.py")
    print("2. Check for token distribution visualizations")
    print("3. Verify individual rewards are being tracked")

if __name__ == "__main__":
    main()
