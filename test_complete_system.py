#!/usr/bin/env python3
"""
Comprehensive Unit Test for Complete Federated Learning System
Tests: Meta-learning, Global Aggregation, RL-guided SSL-TTT, Evaluation, and Visualization
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import shutil
import logging
from unittest.mock import patch, MagicMock
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transductive_fewshot_model import TransductiveLearner
from coordinators.simple_fedavg_coordinator import SimpleFedAVGCoordinator, SimpleFederatedClient
from visualization.performance_visualization import PerformanceVisualizer
# from preprocessing.blockchain_federated_unsw_preprocessor import BlockchainFederatedUNSWPreprocessor
import config

# Configure logging for test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestCompleteFederatedSystem(unittest.TestCase):
    """Comprehensive test for the complete federated learning system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock config object
        self.config = MagicMock()
        self.config.num_clients = 3
        self.config.num_rounds = 2
        self.config.zero_day_attack = "Worms"
        self.config.use_tcn = True
        self.config.sequence_length = 30
        self.config.sequence_stride = 15
        self.config.local_epochs = 50
        self.config.learning_rate = 0.001
        self.config.ttt_base_steps = 100
        self.config.ttt_max_steps = 300
        self.config.ttt_lr = 0.0005
        self.config.ttt_lr_min = 1e-6
        self.config.n_way = 2
        self.config.k_shot = 50
        self.config.n_query = 100
        self.config.n_tasks = 20
        
        # Create test data
        self.create_test_data()
        
        # Initialize components
        self.setup_components()
        
        logger.info("✅ Test environment setup completed")
    
    def create_test_data(self):
        """Create synthetic test data for testing"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic network traffic data
        n_samples = 1000
        n_features = 23
        
        # Generate features (normalized)
        self.X_train = torch.randn(n_samples, n_features)
        self.X_val = torch.randn(200, n_features)
        self.X_test = torch.randn(300, n_features)
        
        # Generate binary labels (0: Normal, 1: Attack)
        # Create some structure in the data
        normal_mask = self.X_train[:, 0] > 0.5
        self.y_train = torch.where(normal_mask, torch.zeros(n_samples), torch.ones(n_samples))
        
        normal_mask_val = self.X_val[:, 0] > 0.5
        self.y_val = torch.where(normal_mask_val, torch.zeros(200), torch.ones(200))
        
        normal_mask_test = self.X_test[:, 0] > 0.5
        self.y_test = torch.where(normal_mask_test, torch.zeros(300), torch.ones(300))
        
        # Create zero-day attack samples (different pattern)
        zero_day_mask = self.X_test[:, 1] > 0.8
        self.zero_day_indices = torch.where(zero_day_mask)[0]
        
        logger.info(f"✅ Test data created: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")
        logger.info(f"   Zero-day samples: {len(self.zero_day_indices)}")
    
    def setup_components(self):
        """Initialize all system components"""
        # Initialize model
        self.model = TransductiveLearner(
            input_dim=23,
            hidden_dim=128,
            embedding_dim=64,
            num_classes=2,
            support_weight=0.7,
            test_weight=0.3,
            sequence_length=30
        )
        
        # Initialize coordinator
        self.coordinator = SimpleFedAVGCoordinator(
            model=self.model,
            config=self.config,
            device='cpu'
        )
        
        # Initialize clients
        self.clients = []
        for i in range(3):
            client = SimpleFederatedClient(
                client_id=f"test_client_{i}",
                model=self.model,
                config=self.config,
                device='cpu'
            )
            # Set training data
            client.set_training_data(
                self.X_train[i*300:(i+1)*300],
                self.y_train[i*300:(i+1)*300]
            )
            self.clients.append(client)
        
        # Initialize visualizer
        self.visualizer = PerformanceVisualizer(
            output_dir=self.temp_dir,
            attack_name="test_attack"
        )
        
        logger.info("✅ All components initialized")
    
    def test_meta_learning_at_clients(self):
        """Test that clients perform proper meta-learning"""
        logger.info("🧪 Testing meta-learning at clients...")
        
        for i, client in enumerate(self.clients):
            # Test meta-learning training
            update = client.train_local_model(epochs=2)
            
            # Verify update structure
            self.assertIsNotNone(update)
            self.assertEqual(update.client_id, f"test_client_{i}")
            self.assertIsInstance(update.model_parameters, dict)
            self.assertIsInstance(update.training_loss, float)
            self.assertIsInstance(update.validation_accuracy, float)
            self.assertGreater(update.validation_accuracy, 0.0)
            
            logger.info(f"   Client {i}: Loss={update.training_loss:.4f}, Accuracy={update.validation_accuracy:.4f}")
        
        logger.info("✅ Meta-learning at clients test passed")
    
    def test_global_aggregation(self):
        """Test global model aggregation"""
        logger.info("🧪 Testing global model aggregation...")
        
        # Get client updates
        client_updates = []
        for client in self.clients:
            update = client.train_local_model(epochs=2)
            client_updates.append(update)
        
        # Test aggregation
        aggregated_model = self.coordinator.aggregate_models(client_updates)
        
        # Verify aggregation worked
        self.assertIsNotNone(aggregated_model)
        
        # Test that aggregated model can make predictions
        with torch.no_grad():
            test_output = aggregated_model(self.X_test[:10])
            self.assertEqual(test_output.shape, (10, 2))
        
        logger.info("✅ Global aggregation test passed")
    
    def test_rl_guided_ssl_ttt_adaptation(self):
        """Test RL-guided SSL-TTT adaptation"""
        logger.info("🧪 Testing RL-guided SSL-TTT adaptation...")
        
        # Create support and query sets
        support_size = 50
        query_size = 100
        
        support_x = self.X_test[:support_size]
        support_y = self.y_test[:support_size]
        query_x = self.X_test[support_size:support_size+query_size]
        
        # Test TTT adaptation
        adapted_model = self.coordinator._perform_advanced_ttt_adaptation(
            support_x, support_y, query_x, self.config
        )
        
        # Verify adaptation worked
        self.assertIsNotNone(adapted_model)
        
        # Test that adapted model can make predictions
        with torch.no_grad():
            test_output = adapted_model(query_x[:10])
            self.assertEqual(test_output.shape, (10, 2))
        
        # Verify TTT adaptation data was stored
        self.assertTrue(hasattr(adapted_model, 'ttt_adaptation_data'))
        self.assertIsInstance(adapted_model.ttt_adaptation_data, dict)
        
        logger.info("✅ RL-guided SSL-TTT adaptation test passed")
    
    def test_model_evaluation(self):
        """Test base model and TTT model evaluation"""
        logger.info("🧪 Testing model evaluation...")
        
        # Test base model evaluation
        base_results = self.model.evaluate_zero_day_detection(
            self.X_test, self.y_test, self.zero_day_indices
        )
        
        # Verify base model results
        self.assertIsInstance(base_results, dict)
        self.assertIn('accuracy', base_results)
        self.assertIn('f1_score', base_results)
        self.assertIn('roc_auc', base_results)
        self.assertIn('confusion_matrix', base_results)
        
        # Test TTT adapted model evaluation
        support_x = self.X_test[:50]
        support_y = self.y_test[:50]
        query_x = self.X_test[50:150]
        
        adapted_model = self.coordinator._perform_advanced_ttt_adaptation(
            support_x, support_y, query_x, self.config
        )
        
        ttt_results = adapted_model.evaluate_zero_day_detection(
            self.X_test, self.y_test, self.zero_day_indices
        )
        
        # Verify TTT model results
        self.assertIsInstance(ttt_results, dict)
        self.assertIn('accuracy', ttt_results)
        self.assertIn('f1_score', ttt_results)
        self.assertIn('roc_auc', ttt_results)
        
        logger.info(f"   Base Model - Accuracy: {base_results['accuracy']:.4f}, F1: {base_results['f1_score']:.4f}")
        logger.info(f"   TTT Model - Accuracy: {ttt_results['accuracy']:.4f}, F1: {ttt_results['f1_score']:.4f}")
        logger.info("✅ Model evaluation test passed")
    
    def test_visualization_methods(self):
        """Test visualization methods"""
        logger.info("🧪 Testing visualization methods...")
        
        # Create mock evaluation results
        evaluation_results = {
            'base_model': {
                'accuracy': 0.85,
                'f1_score': 0.82,
                'precision': 0.80,
                'recall': 0.84,
                'roc_auc': 0.90,
                'confusion_matrix': np.array([[100, 20], [15, 165]]),
                'predictions': np.random.randint(0, 2, 300),
                'probabilities': np.random.rand(300, 2)
            },
            'adapted_model': {
                'accuracy': 0.87,
                'f1_score': 0.84,
                'precision': 0.82,
                'recall': 0.86,
                'roc_auc': 0.92,
                'confusion_matrix': np.array([[105, 15], [12, 168]]),
                'predictions': np.random.randint(0, 2, 300),
                'probabilities': np.random.rand(300, 2)
            }
        }
        
        # Test training history plot
        training_history = {
            'rounds': [1, 2],
            'losses': [0.5, 0.3],
            'accuracies': [0.7, 0.8]
        }
        
        plot_path = self.visualizer.plot_training_history(training_history, save=True)
        self.assertTrue(os.path.exists(plot_path))
        logger.info(f"   Training history plot: {plot_path}")
        
        # Test confusion matrix plot
        plot_path = self.visualizer.plot_confusion_matrices(evaluation_results, save=True)
        self.assertTrue(os.path.exists(plot_path))
        logger.info(f"   Confusion matrix plot: {plot_path}")
        
        # Test client performance plot
        client_data = [
            {'client_id': 'client_1', 'accuracy': 0.85, 'f1_score': 0.82},
            {'client_id': 'client_2', 'accuracy': 0.87, 'f1_score': 0.84},
            {'client_id': 'client_3', 'accuracy': 0.83, 'f1_score': 0.80}
        ]
        
        plot_path = self.visualizer.plot_client_performance(client_data, save=True)
        self.assertTrue(os.path.exists(plot_path))
        logger.info(f"   Client performance plot: {plot_path}")
        
        # Test performance comparison plot
        plot_path = self.visualizer.plot_performance_comparison_with_annotations(
            evaluation_results['base_model'], 
            evaluation_results['adapted_model'],
            save=True
        )
        self.assertTrue(os.path.exists(plot_path))
        logger.info(f"   Performance comparison plot: {plot_path}")
        
        # Test metrics JSON save
        system_data = {
            'evaluation_results': evaluation_results,
            'training_history': training_history,
            'client_performance': client_data
        }
        
        json_path = self.visualizer.save_metrics_to_json(system_data)
        self.assertTrue(os.path.exists(json_path))
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
        self.assertIn('evaluation_results', saved_data)
        self.assertIn('training_history', saved_data)
        
        logger.info(f"   Metrics JSON: {json_path}")
        logger.info("✅ Visualization methods test passed")
    
    def test_complete_federated_learning_workflow(self):
        """Test complete federated learning workflow"""
        logger.info("🧪 Testing complete federated learning workflow...")
        
        # Step 1: Meta-learning at clients
        client_updates = []
        for client in self.clients:
            update = client.train_local_model(epochs=2)
            client_updates.append(update)
        
        # Step 2: Global aggregation
        aggregated_model = self.coordinator.aggregate_models(client_updates)
        
        # Step 3: TTT adaptation
        support_x = self.X_test[:50]
        support_y = self.y_test[:50]
        query_x = self.X_test[50:150]
        
        adapted_model = self.coordinator._perform_advanced_ttt_adaptation(
            support_x, support_y, query_x, self.config
        )
        
        # Step 4: Evaluation
        base_results = aggregated_model.evaluate_zero_day_detection(
            self.X_test, self.y_test, self.zero_day_indices
        )
        
        ttt_results = adapted_model.evaluate_zero_day_detection(
            self.X_test, self.y_test, self.zero_day_indices
        )
        
        # Step 5: Visualization
        evaluation_results = {
            'base_model': base_results,
            'adapted_model': ttt_results
        }
        
        # Generate all visualizations
        plot_paths = {}
        
        # Training history
        training_history = {
            'rounds': [1, 2],
            'losses': [0.5, 0.3],
            'accuracies': [0.7, 0.8]
        }
        plot_paths['training_history'] = self.visualizer.plot_training_history(training_history, save=True)
        
        # Confusion matrices
        plot_paths['confusion_matrix'] = self.visualizer.plot_confusion_matrices(evaluation_results, save=True)
        
        # Client performance
        client_data = [
            {'client_id': f'client_{i}', 'accuracy': update.validation_accuracy, 'f1_score': 0.8}
            for i, update in enumerate(client_updates)
        ]
        plot_paths['client_performance'] = self.visualizer.plot_client_performance(client_data, save=True)
        
        # Performance comparison
        plot_paths['performance_comparison'] = self.visualizer.plot_performance_comparison_with_annotations(
            base_results, ttt_results, save=True
        )
        
        # Metrics JSON
        system_data = {
            'evaluation_results': evaluation_results,
            'training_history': training_history,
            'client_performance': client_data
        }
        plot_paths['metrics_json'] = self.visualizer.save_metrics_to_json(system_data)
        
        # Verify all plots were created
        for plot_name, plot_path in plot_paths.items():
            self.assertTrue(os.path.exists(plot_path), f"Plot {plot_name} was not created: {plot_path}")
            logger.info(f"   {plot_name}: {plot_path}")
        
        # Verify performance metrics
        self.assertGreater(base_results['accuracy'], 0.0)
        self.assertGreater(ttt_results['accuracy'], 0.0)
        
        logger.info(f"   Base Model Performance: Accuracy={base_results['accuracy']:.4f}, F1={base_results['f1_score']:.4f}")
        logger.info(f"   TTT Model Performance: Accuracy={ttt_results['accuracy']:.4f}, F1={ttt_results['f1_score']:.4f}")
        logger.info("✅ Complete federated learning workflow test passed")
    
    def test_rl_agent_functionality(self):
        """Test RL agent functionality specifically"""
        logger.info("🧪 Testing RL agent functionality...")
        
        # Test RL agent initialization
        agent = self.coordinator.model.threshold_agent
        self.assertIsNotNone(agent)
        
        # Test state representation
        confidence_scores = torch.rand(100)
        probabilities = torch.rand(100, 2)
        state = agent.get_enhanced_state(confidence_scores, probabilities)
        
        self.assertEqual(state.shape, (15,))  # 15-dimensional state
        self.assertTrue(torch.all(torch.isfinite(state)))
        
        # Test action generation
        actions = agent.get_meta_control_actions(state)
        
        self.assertIn('confidence_threshold', actions)
        self.assertIn('entropy_weight', actions)
        self.assertIn('should_adapt', actions)
        self.assertIn('adaptation_intensity', actions)
        
        # Verify action ranges
        self.assertGreaterEqual(actions['confidence_threshold'], 0.1)
        self.assertLessEqual(actions['confidence_threshold'], 0.9)
        self.assertGreaterEqual(actions['entropy_weight'], 0.1)
        self.assertLessEqual(actions['entropy_weight'], 1.0)
        self.assertIsInstance(actions['should_adapt'], bool)
        self.assertGreaterEqual(actions['adaptation_intensity'], 0.1)
        self.assertLessEqual(actions['adaptation_intensity'], 2.0)
        
        logger.info(f"   RL Actions: {actions}")
        logger.info("✅ RL agent functionality test passed")
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        logger.info("✅ Test environment cleaned up")

def run_comprehensive_test():
    """Run the comprehensive test suite"""
    logger.info("🚀 Starting Comprehensive Federated Learning System Test")
    logger.info("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCompleteFederatedSystem)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    logger.info("=" * 80)
    if result.wasSuccessful():
        logger.info("🎉 ALL TESTS PASSED! Complete system is working correctly.")
        logger.info(f"   Tests run: {result.testsRun}")
        logger.info("   Failures: 0")
        logger.info("   Errors: 0")
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error(f"   Tests run: {result.testsRun}")
        logger.error(f"   Failures: {len(result.failures)}")
        logger.error(f"   Errors: {len(result.errors)}")
        
        for failure in result.failures:
            logger.error(f"   FAILURE: {failure[0]}")
            logger.error(f"   {failure[1]}")
        
        for error in result.errors:
            logger.error(f"   ERROR: {error[0]}")
            logger.error(f"   {error[1]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
