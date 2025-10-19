"""
Shapley Value Calculator for Federated Learning Incentive Distribution

This module implements the Shapley value calculation for fair contribution evaluation
in blockchain-enabled federated learning systems. The Shapley value ensures that
each client receives rewards proportional to their true marginal contribution to
the global model performance.

References:
- Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games
- Ghorbani, A., & Zou, J. (2019). Data Shapley: Equitable Valuation of Data for Machine Learning
- Wang, T., et al. (2020). A Shapley Value-based Approach for Fair Federated Learning
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from itertools import combinations, permutations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ClientContribution:
    """Represents a client's contribution to the federated learning system"""
    client_id: str
    shapley_value: float
    marginal_contributions: List[float]
    participation_rate: float
    data_quality: float
    performance_improvement: float

class ShapleyValueCalculator:
    """
    Calculates Shapley values for fair contribution evaluation in federated learning.
    
    The Shapley value measures the average marginal contribution of each client
    across all possible coalitions, ensuring fair and mathematically sound
    incentive distribution.
    """
    
    def __init__(self, num_clients: int = 3, evaluation_metric: str = 'accuracy', client_ids: List[str] = None):
        """
        Initialize the Shapley value calculator.
        
        Args:
            num_clients: Number of clients in the federated learning system
            evaluation_metric: Metric used for performance evaluation ('accuracy', 'f1', 'mcc')
            client_ids: List of actual client IDs (if None, will generate default ones)
        """
        self.num_clients = num_clients
        self.evaluation_metric = evaluation_metric
        if client_ids is not None:
            self.client_ids = client_ids
        else:
            self.client_ids = [f"client_{i+1}" for i in range(num_clients)]
        
        logger.info(f"Initialized Shapley value calculator for {num_clients} clients: {self.client_ids}")
    
    def calculate_coalition_performance(self, 
                                      client_subset: List[str], 
                                      global_performance: float,
                                      individual_performances: Dict[str, float]) -> float:
        """
        Calculate the performance of a coalition of clients.
        
        This is a simplified model that approximates coalition performance
        based on individual contributions and synergy effects.
        
        Args:
            client_subset: List of client IDs in the coalition
            global_performance: Performance when all clients participate
            individual_performances: Individual performance of each client
            
        Returns:
            Estimated coalition performance
        """
        if not client_subset:
            return 0.0
        
        if len(client_subset) == self.num_clients:
            return global_performance
        
        # Calculate base performance as weighted average of individual performances
        individual_perfs = [individual_performances[client_id] for client_id in client_subset]
        
        # Simple weighted average that preserves individual differences
        individual_avg = np.mean(individual_perfs)
        
        # Apply synergy factor - larger coalitions have better performance
        synergy_factor = 1.0 + (len(client_subset) - 1) * 0.1  # 10% synergy per additional client
        
        # Apply diversity factor - more diverse coalitions perform better
        diversity_factor = 1.0 + (len(client_subset) / self.num_clients) * 0.2  # Up to 20% diversity bonus
        
        coalition_performance = individual_avg * synergy_factor * diversity_factor
        
        # Scale coalition performance to be more realistic relative to global performance
        # Single clients should perform proportionally to their individual performance
        if len(client_subset) == 1:
            # Single client: use their individual performance directly (scaled to global performance range)
            individual_perf = individual_perfs[0]
            # Scale individual performance to be in the range of global performance
            # Map [0.5, 1.0] to [0, global_performance]
            normalized_perf = (individual_perf - 0.5) / 0.5  # Normalize to [0, 1]
            coalition_performance = normalized_perf * global_performance
        
        # Ensure coalition performance is reasonable
        max_reasonable = global_performance * 1.5  # Allow up to 50% above global performance
        return min(coalition_performance, max_reasonable)
    
    def calculate_shapley_values(self, 
                               global_performance: float,
                               individual_performances: Dict[str, float],
                               client_data_quality: Dict[str, float],
                               client_participation: Dict[str, float]) -> List[ClientContribution]:
        """
        Calculate Shapley values for all clients.
        
        Args:
            global_performance: Performance of the global model
            individual_performances: Individual performance of each client
            client_data_quality: Data quality score for each client
            client_participation: Participation rate for each client
            
        Returns:
            List of ClientContribution objects with Shapley values
        """
        logger.info("Starting Shapley value calculation...")
        
        # Generate all possible coalitions (2^n - 1, excluding empty set)
        all_coalitions = []
        for r in range(1, self.num_clients + 1):
            for coalition in combinations(self.client_ids, r):
                all_coalitions.append(list(coalition))
        
        logger.info(f"Evaluating {len(all_coalitions)} possible coalitions")
        
        # Calculate performance for each coalition
        coalition_performances = {}
        for coalition in all_coalitions:
            coalition_key = tuple(sorted(coalition))
            performance = self.calculate_coalition_performance(
                coalition, global_performance, individual_performances
            )
            coalition_performances[coalition_key] = performance
        
        # Calculate Shapley values
        shapley_values = {}
        marginal_contributions = {client_id: [] for client_id in self.client_ids}
        
        for client_id in self.client_ids:
            shapley_value = 0.0
            
            # Calculate marginal contribution for each coalition
            for coalition in all_coalitions:
                if client_id in coalition:
                    # Coalition with the client
                    coalition_with = tuple(sorted(coalition))
                    performance_with = coalition_performances[coalition_with]
                    
                    # Coalition without the client
                    coalition_without = tuple(sorted([c for c in coalition if c != client_id]))
                    if coalition_without:
                        performance_without = coalition_performances[coalition_without]
                    else:
                        performance_without = 0.0
                    
                    # Marginal contribution
                    marginal_contribution = performance_with - performance_without
                    marginal_contributions[client_id].append(marginal_contribution)
                    
                    # Weight by coalition size (Shapley value formula)
                    coalition_size = len(coalition)
                    # Correct Shapley value weight: 1/n * C(n-1, s-1) where s is coalition size
                    weight = 1.0 / (self.num_clients * self._binomial_coefficient(self.num_clients - 1, coalition_size - 1))
                    
                    shapley_value += weight * marginal_contribution
            
            shapley_values[client_id] = shapley_value
        
        # Normalize Shapley values to ensure they sum to global performance
        total_shapley = sum(shapley_values.values())
        if total_shapley > 0:
            normalization_factor = global_performance / total_shapley
            for client_id in self.client_ids:
                shapley_values[client_id] *= normalization_factor
        
        # Create ClientContribution objects
        contributions = []
        for client_id in self.client_ids:
            contribution = ClientContribution(
                client_id=client_id,
                shapley_value=shapley_values[client_id],
                marginal_contributions=marginal_contributions[client_id],
                participation_rate=client_participation.get(client_id, 1.0),
                data_quality=client_data_quality.get(client_id, 0.5),
                performance_improvement=individual_performances.get(client_id, 0.0)
            )
            contributions.append(contribution)
        
        logger.info("Shapley value calculation completed")
        return contributions
    
    def _binomial_coefficient(self, n: int, k: int) -> int:
        """Calculate binomial coefficient C(n, k)"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    def calculate_token_rewards(self, 
                              contributions: List[ClientContribution],
                              total_tokens: int = 100000,
                              base_reward_ratio: float = 0.1) -> Dict[str, int]:
        """
        Calculate token rewards based on Shapley values.
        
        Args:
            contributions: List of client contributions with Shapley values
            total_tokens: Total tokens to distribute
            base_reward_ratio: Ratio of tokens for base participation (vs performance)
            
        Returns:
            Dictionary mapping client_id to token reward
        """
        logger.info(f"Calculating token rewards for {len(contributions)} clients")
        
        # Calculate base rewards (equal for all participants)
        base_tokens = int(total_tokens * base_reward_ratio)
        base_reward_per_client = base_tokens // len(contributions)
        
        # Calculate performance-based rewards
        performance_tokens = total_tokens - base_tokens
        
        # Normalize Shapley values for token distribution
        total_shapley = sum(contrib.shapley_value for contrib in contributions)
        if total_shapley == 0:
            # Fallback to equal distribution if no Shapley values
            performance_reward_per_client = performance_tokens // len(contributions)
        else:
            # Distribute performance tokens based on Shapley values
            performance_rewards = {}
            for contrib in contributions:
                performance_rewards[contrib.client_id] = int(
                    (contrib.shapley_value / total_shapley) * performance_tokens
                )
        
        # Calculate total rewards
        token_rewards = {}
        for contrib in contributions:
            if total_shapley == 0:
                total_reward = base_reward_per_client + performance_tokens // len(contributions)
            else:
                total_reward = base_reward_per_client + performance_rewards[contrib.client_id]
            
            token_rewards[contrib.client_id] = total_reward
        
        logger.info(f"Token rewards calculated: {token_rewards}")
        return token_rewards
    
    def get_contribution_analysis(self, contributions: List[ClientContribution]) -> Dict[str, Any]:
        """
        Generate detailed analysis of client contributions.
        
        Args:
            contributions: List of client contributions
            
        Returns:
            Dictionary with contribution analysis
        """
        analysis = {
            'total_clients': len(contributions),
            'shapley_values': {contrib.client_id: contrib.shapley_value for contrib in contributions},
            'marginal_contributions': {contrib.client_id: contrib.marginal_contributions for contrib in contributions},
            'participation_rates': {contrib.client_id: contrib.participation_rate for contrib in contributions},
            'data_quality_scores': {contrib.client_id: contrib.data_quality for contrib in contributions},
            'performance_improvements': {contrib.client_id: contrib.performance_improvement for contrib in contributions},
            'fairness_metrics': self._calculate_fairness_metrics(contributions)
        }
        
        return analysis
    
    def _calculate_fairness_metrics(self, contributions: List[ClientContribution]) -> Dict[str, float]:
        """Calculate fairness metrics for the contribution distribution."""
        shapley_values = [contrib.shapley_value for contrib in contributions]
        
        if not shapley_values or all(v == 0 for v in shapley_values):
            return {'gini_coefficient': 0.0, 'max_min_ratio': 1.0, 'variance': 0.0}
        
        # Gini coefficient (measure of inequality)
        shapley_values_sorted = sorted(shapley_values)
        n = len(shapley_values_sorted)
        cumsum = np.cumsum(shapley_values_sorted)
        gini_coefficient = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Max-min ratio
        max_min_ratio = max(shapley_values) / min(shapley_values) if min(shapley_values) > 0 else float('inf')
        
        # Variance
        variance = np.var(shapley_values)
        
        return {
            'gini_coefficient': gini_coefficient,
            'max_min_ratio': max_min_ratio,
            'variance': variance
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the Shapley value calculator
    calculator = ShapleyValueCalculator(num_clients=3)
    
    # Example data
    global_performance = 0.95
    individual_performances = {
        'client_1': 0.92,
        'client_2': 0.88,
        'client_3': 0.90
    }
    client_data_quality = {
        'client_1': 0.9,
        'client_2': 0.85,
        'client_3': 0.88
    }
    client_participation = {
        'client_1': 1.0,
        'client_2': 0.95,
        'client_3': 1.0
    }
    
    # Calculate contributions
    contributions = calculator.calculate_shapley_values(
        global_performance, individual_performances, 
        client_data_quality, client_participation
    )
    
    # Calculate token rewards
    token_rewards = calculator.calculate_token_rewards(contributions, total_tokens=100000)
    
    # Print results
    print("Shapley Value Results:")
    for contrib in contributions:
        print(f"{contrib.client_id}: Shapley={contrib.shapley_value:.4f}, Tokens={token_rewards[contrib.client_id]}")
    
    # Print analysis
    analysis = calculator.get_contribution_analysis(contributions)
    print(f"\nFairness Metrics: {analysis['fairness_metrics']}")

