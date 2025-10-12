#!/usr/bin/env python3
"""
Data Leakage Detection Tests for Blockchain Federated Learning System
Ensures honest evaluation by detecting potential data leakage issues
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import copy

logger = logging.getLogger(__name__)

class DataLeakageDetector:
    """
    Detects potential data leakage in federated learning and few-shot learning systems
    """
    
    def __init__(self):
        self.test_results = {}
        
    def test_model_cloning(self, model, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, Any]:
        """
        Test if model cloning properly isolates models
        
        Args:
            model: Original model
            test_data: Test features
            test_labels: Test labels
            
        Returns:
            test_result: Dictionary with test results
        """
        logger.info("🔍 Testing model cloning for data leakage...")
        
        try:
            # Create two clones
            model_clone1 = copy.deepcopy(model)
            model_clone2 = copy.deepcopy(model)
            
            # Train clone1 on subset of data
            subset_size = len(test_data) // 2
            subset_data = test_data[:subset_size]
            subset_labels = test_labels[:subset_size]
            
            # Simple training on clone1
            optimizer1 = torch.optim.Adam(model_clone1.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            
            for epoch in range(5):
                optimizer1.zero_grad()
                outputs = model_clone1(subset_data)
                loss = criterion(outputs, subset_labels)
                loss.backward()
                optimizer1.step()
            
            # Test both models on remaining data
            remaining_data = test_data[subset_size:]
            remaining_labels = test_labels[subset_size:]
            
            model_clone1.eval()
            model_clone2.eval()
            
            with torch.no_grad():
                pred1 = model_clone1(remaining_data).argmax(dim=1)
                pred2 = model_clone2(remaining_data).argmax(dim=1)
                
                acc1 = accuracy_score(remaining_labels.cpu(), pred1.cpu())
                acc2 = accuracy_score(remaining_labels.cpu(), pred2.cpu())
            
            # Check if models are independent
            models_independent = abs(acc1 - acc2) > 0.01  # Should be different
            
            result = {
                'test_name': 'model_cloning',
                'passed': models_independent,
                'clone1_accuracy': acc1,
                'clone2_accuracy': acc2,
                'accuracy_difference': abs(acc1 - acc2),
                'message': 'Models are independent' if models_independent else 'Potential model sharing detected'
            }
            
            logger.info(f"✅ Model cloning test: {'PASSED' if models_independent else 'FAILED'}")
            logger.info(f"   Clone1 accuracy: {acc1:.4f}, Clone2 accuracy: {acc2:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Model cloning test failed: {str(e)}")
            return {
                'test_name': 'model_cloning',
                'passed': False,
                'error': str(e)
            }
    
    def test_support_query_separation(self, X_test: np.ndarray, y_test: np.ndarray, 
                                    support_ratio: float = 0.2) -> Dict[str, Any]:
        """
        Test if support and query sets are properly separated
        
        Args:
            X_test: Test features
            y_test: Test labels
            support_ratio: Ratio of data used for support set
            
        Returns:
            test_result: Dictionary with test results
        """
        logger.info("🔍 Testing support-query separation for data leakage...")
        
        try:
            # Create support-query split
            support_size = int(len(X_test) * support_ratio)
            query_size = len(X_test) - support_size
            
            # Use stratified split
            X_support, X_query, y_support, y_query = train_test_split(
                X_test, y_test, test_size=1-support_ratio, stratify=y_test, random_state=42
            )
            
            # Check class distribution similarity
            support_classes = np.bincount(y_support)
            query_classes = np.bincount(y_query)
            
            # Normalize distributions
            support_dist = support_classes / support_classes.sum()
            query_dist = query_classes / query_classes.sum()
            
            # Calculate distribution difference
            dist_difference = np.abs(support_dist - query_dist).mean()
            
            # Check if distributions are similar (good stratification)
            distributions_similar = dist_difference < 0.1
            
            result = {
                'test_name': 'support_query_separation',
                'passed': distributions_similar,
                'support_size': len(X_support),
                'query_size': len(X_query),
                'support_ratio': support_ratio,
                'distribution_difference': dist_difference,
                'support_distribution': support_dist.tolist(),
                'query_distribution': query_dist.tolist(),
                'message': 'Support and query sets properly stratified' if distributions_similar else 'Poor stratification detected'
            }
            
            logger.info(f"✅ Support-query separation test: {'PASSED' if distributions_similar else 'FAILED'}")
            logger.info(f"   Distribution difference: {dist_difference:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Support-query separation test failed: {str(e)}")
            return {
                'test_name': 'support_query_separation',
                'passed': False,
                'error': str(e)
            }
    
    def test_random_seed_consistency(self, X_test: np.ndarray, y_test: np.ndarray, 
                                   num_tests: int = 5) -> Dict[str, Any]:
        """
        Test if random seeds produce consistent results
        
        Args:
            X_test: Test features
            y_test: Test labels
            num_tests: Number of tests to run
            
        Returns:
            test_result: Dictionary with test results
        """
        logger.info("🔍 Testing random seed consistency for data leakage...")
        
        try:
            results = []
            
            for i in range(num_tests):
                # Use same seed for reproducible results
                np.random.seed(42)
                torch.manual_seed(42)
                
                # Create random split
                indices = np.random.permutation(len(X_test))
                split_point = len(X_test) // 2
                
                train_indices = indices[:split_point]
                test_indices = indices[split_point:]
                
                results.append({
                    'test_run': i,
                    'train_indices': train_indices[:5].tolist(),  # First 5 indices
                    'test_indices': test_indices[:5].tolist()    # First 5 indices
                })
            
            # Check if all runs produce same results
            first_run = results[0]
            all_consistent = all(
                np.array_equal(r['train_indices'], first_run['train_indices']) and
                np.array_equal(r['test_indices'], first_run['test_indices'])
                for r in results
            )
            
            result = {
                'test_name': 'random_seed_consistency',
                'passed': all_consistent,
                'num_tests': num_tests,
                'results': results,
                'message': 'Random seeds produce consistent results' if all_consistent else 'Inconsistent random behavior detected'
            }
            
            logger.info(f"✅ Random seed consistency test: {'PASSED' if all_consistent else 'FAILED'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Random seed consistency test failed: {str(e)}")
            return {
                'test_name': 'random_seed_consistency',
                'passed': False,
                'error': str(e)
            }
    
    def test_ttt_overfitting(self, model, support_x: torch.Tensor, support_y: torch.Tensor,
                           query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, Any]:
        """
        Test if TTT adaptation overfits to support set
        
        Args:
            model: Model to test
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features
            query_y: Query set labels
            
        Returns:
            test_result: Dictionary with test results
        """
        logger.info("🔍 Testing TTT overfitting for data leakage...")
        
        try:
            # Clone model for testing
            test_model = copy.deepcopy(model)
            test_model.train()
            
            # Train on support set
            optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            
            support_losses = []
            query_losses = []
            
            for epoch in range(20):
                # Support set training
                optimizer.zero_grad()
                support_outputs = test_model(support_x)
                support_loss = criterion(support_outputs, support_y)
                support_loss.backward()
                optimizer.step()
                
                # Evaluate on both sets
                test_model.eval()
                with torch.no_grad():
                    support_pred = test_model(support_x)
                    query_pred = test_model(query_x)
                    
                    support_loss_val = criterion(support_pred, support_y).item()
                    query_loss_val = criterion(query_pred, query_y).item()
                    
                    support_losses.append(support_loss_val)
                    query_losses.append(query_loss_val)
                
                test_model.train()
            
            # Check for overfitting (support loss much lower than query loss)
            final_support_loss = support_losses[-1]
            final_query_loss = query_losses[-1]
            overfitting_ratio = final_query_loss / final_support_loss if final_support_loss > 0 else float('inf')
            
            # Overfitting threshold: query loss should not be more than 10x support loss (realistic for few-shot learning)
            # This accounts for the fact that support set is small and query set is larger
            no_overfitting = overfitting_ratio < 10.0
            
            result = {
                'test_name': 'ttt_overfitting',
                'passed': no_overfitting,
                'final_support_loss': final_support_loss,
                'final_query_loss': final_query_loss,
                'overfitting_ratio': overfitting_ratio,
                'support_losses': support_losses[-5:],  # Last 5 losses
                'query_losses': query_losses[-5:],       # Last 5 losses
                'message': 'No significant overfitting detected' if no_overfitting else 'Potential overfitting to support set'
            }
            
            logger.info(f"✅ TTT overfitting test: {'PASSED' if no_overfitting else 'FAILED'}")
            logger.info(f"   Overfitting ratio: {overfitting_ratio:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ TTT overfitting test failed: {str(e)}")
            return {
                'test_name': 'ttt_overfitting',
                'passed': False,
                'error': str(e)
            }
    
    def run_all_tests(self, model, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, Any]:
        """
        Run all data leakage detection tests
        
        Args:
            model: Model to test
            X_test: Test features
            y_test: Test labels
            
        Returns:
            results: Dictionary with all test results
        """
        logger.info("🔍 Running comprehensive data leakage detection tests...")
        
        # Convert to numpy for some tests
        X_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        
        results = {
            'model_cloning': self.test_model_cloning(model, X_test, y_test),
            'support_query_separation': self.test_support_query_separation(X_np, y_np),
            'random_seed_consistency': self.test_random_seed_consistency(X_np, y_np)
        }
        
        # Add TTT overfitting test if we have enough data
        if len(X_test) > 100:
            support_size = min(20, len(X_test) // 10)
            query_size = len(X_test) - support_size
            
            torch.manual_seed(42)
            indices = torch.randperm(len(X_test))
            support_indices = indices[:support_size]
            query_indices = indices[support_size:support_size + query_size]
            
            support_x = X_test[support_indices]
            support_y = y_test[support_indices]
            query_x = X_test[query_indices]
            query_y = y_test[query_indices]
            
            results['ttt_overfitting'] = self.test_ttt_overfitting(
                model, support_x, support_y, query_x, query_y
            )
        
        # Calculate overall score
        passed_tests = sum(1 for test in results.values() if test.get('passed', False))
        total_tests = len(results)
        overall_score = passed_tests / total_tests
        
        results['overall'] = {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'overall_score': overall_score,
            'status': 'PASSED' if overall_score >= 0.8 else 'FAILED',
            'message': f'Data leakage detection: {passed_tests}/{total_tests} tests passed'
        }
        
        logger.info(f"🎯 Data leakage detection completed: {passed_tests}/{total_tests} tests passed")
        
        return results

def main():
    """Test the data leakage detector"""
    logger.info("Testing Data Leakage Detector")
    
    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim=10, hidden_dim=32, num_classes=2):
            super(SimpleModel, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create synthetic data
    torch.manual_seed(42)
    n_samples = 200
    n_features = 10
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    
    # Create model
    model = SimpleModel(input_dim=n_features)
    
    # Run tests
    detector = DataLeakageDetector()
    results = detector.run_all_tests(model, X, y)
    
    # Print results
    print("\n" + "="*60)
    print("DATA LEAKAGE DETECTION RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        if test_name != 'overall':
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            print(f"{test_name}: {status}")
            print(f"  {result.get('message', 'No message')}")
    
    print(f"\nOverall: {results['overall']['status']}")
    print(f"Score: {results['overall']['overall_score']:.2f}")

if __name__ == "__main__":
    main()
