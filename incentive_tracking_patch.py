
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
