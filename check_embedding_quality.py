#!/usr/bin/env python3
"""
Embedding Quality Diagnostic Tool
Implements Recommendation #3: Check Embedding Quality

This script:
1. Visualizes embeddings using t-SNE
2. Verifies prototypes are well-separated
3. Checks if meta-training is working correctly
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingQualityChecker:
    """Diagnostic tool for checking embedding quality"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        logger.info(f"Initialized EmbeddingQualityChecker on device: {device}")
    
    def extract_embeddings(self, X):
        """Extract embeddings from model"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device) if not isinstance(X, torch.Tensor) else X.to(self.device)
            embeddings = self.model.extract_embeddings(X_tensor)
            return embeddings.cpu().numpy()
    
    def compute_prototypes(self, support_x, support_y):
        """Compute prototypes from support set"""
        support_x_tensor = torch.FloatTensor(support_x).to(self.device) if not isinstance(support_x, torch.Tensor) else support_x.to(self.device)
        support_y_tensor = torch.LongTensor(support_y).to(self.device) if not isinstance(support_y, torch.Tensor) else support_y.to(self.device)
        
        with torch.no_grad():
            prototypes, unique_labels = self.model.compute_prototypes(support_x_tensor, support_y_tensor)
            return prototypes.cpu().numpy(), unique_labels.cpu().numpy()
    
    def visualize_embeddings_tsne(self, embeddings, labels, title="Embeddings t-SNE Visualization", save_path=None):
        """Visualize embeddings using t-SNE"""
        logger.info(f"Computing t-SNE for {len(embeddings)} samples...")
        
        # Use PCA for initial dimensionality reduction if needed
        if embeddings.shape[1] > 50:
            logger.info(f"Reducing dimensions from {embeddings.shape[1]} to 50 using PCA first...")
            pca = PCA(n_components=50)
            embeddings_reduced = pca.fit_transform(embeddings)
            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        else:
            embeddings_reduced = embeddings
        
        # Compute t-SNE (use max_iter for newer scikit-learn versions)
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        except TypeError:
            # Fallback for older scikit-learn versions that use n_iter
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            except TypeError:
                # Even older versions without n_iter parameter
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_reduced)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot different classes with different colors
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = "Normal" if label == 0 else f"Attack_{label}"
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label_name, alpha=0.6, s=50)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved t-SNE visualization to {save_path}")
        
        plt.close()
        
        return embeddings_2d
    
    def visualize_prototypes(self, embeddings, labels, prototypes, unique_proto_labels, 
                           title="Embeddings with Prototypes", save_path=None):
        """Visualize embeddings with prototypes overlaid"""
        logger.info(f"Computing t-SNE for {len(embeddings)} samples and {len(prototypes)} prototypes...")
        
        # Combine embeddings and prototypes for consistent t-SNE
        combined_embeddings = np.vstack([embeddings, prototypes])
        combined_labels = np.hstack([labels, unique_proto_labels + 1000])  # Offset prototype labels
        
        # Use PCA for initial dimensionality reduction if needed
        if combined_embeddings.shape[1] > 50:
            logger.info(f"Reducing dimensions from {combined_embeddings.shape[1]} to 50 using PCA first...")
            pca = PCA(n_components=50)
            combined_embeddings_reduced = pca.fit_transform(combined_embeddings)
            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        else:
            combined_embeddings_reduced = combined_embeddings
        
        # Compute t-SNE (use max_iter for newer scikit-learn versions)
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        except TypeError:
            # Fallback for older scikit-learn versions that use n_iter
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            except TypeError:
                # Even older versions without n_iter parameter
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(combined_embeddings_reduced)
        
        # Split back into embeddings and prototypes
        n_samples = len(embeddings)
        embeddings_2d_samples = embeddings_2d[:n_samples]
        prototypes_2d = embeddings_2d[n_samples:]
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Plot embeddings by class
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = "Normal" if label == 0 else f"Attack_{label}"
            plt.scatter(embeddings_2d_samples[mask, 0], embeddings_2d_samples[mask, 1], 
                       c=[colors[i]], label=label_name, alpha=0.5, s=30)
        
        # Plot prototypes with different markers and larger size
        proto_colors = [colors[np.where(unique_labels == label)[0][0]] for label in unique_proto_labels]
        for i, (proto, label) in enumerate(zip(prototypes_2d, unique_proto_labels)):
            label_name = "Normal Prototype" if label == 0 else f"Attack_{label} Prototype"
            plt.scatter(proto[0], proto[1], 
                       c=[proto_colors[i]], label=label_name, 
                       marker='*', s=500, edgecolors='black', linewidths=2, 
                       alpha=0.9, zorder=10)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prototype visualization to {save_path}")
        
        plt.close()
        
        return embeddings_2d_samples, prototypes_2d
    
    def check_prototype_separation(self, prototypes, unique_labels):
        """Check if prototypes are well-separated"""
        logger.info(f"Checking prototype separation for {len(prototypes)} prototypes...")
        
        # Compute pairwise distances between prototypes
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(prototypes))
        
        # Get distances between different classes (exclude self-distances)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        # Find minimum inter-class distance
        min_inter_class_dist = np.min(distances)
        
        # Compute average inter-class distance
        avg_inter_class_dist = np.mean(distances[distances != np.inf])
        
        # Check if prototypes are well-separated (threshold: should be > 1.0)
        well_separated = min_inter_class_dist > 1.0
        
        results = {
            'num_prototypes': len(prototypes),
            'min_inter_class_distance': float(min_inter_class_dist),
            'avg_inter_class_distance': float(avg_inter_class_dist),
            'well_separated': bool(well_separated),
            'distances_matrix': distances.tolist()
        }
        
        logger.info(f"  Minimum inter-class distance: {min_inter_class_dist:.4f}")
        logger.info(f"  Average inter-class distance: {avg_inter_class_dist:.4f}")
        logger.info(f"  Well-separated: {'✅ YES' if well_separated else '❌ NO'}")
        
        return results
    
    def check_embedding_separability(self, embeddings, labels):
        """Check if embeddings are separable by class using silhouette score"""
        logger.info(f"Checking embedding separability for {len(embeddings)} samples...")
        
        # Use PCA for dimensionality reduction if needed (silhouette score needs reasonable dimensions)
        if embeddings.shape[1] > 50:
            logger.info(f"Reducing dimensions from {embeddings.shape[1]} to 50 using PCA...")
            pca = PCA(n_components=50)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        # Compute silhouette score
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(embeddings_reduced, labels)
            
            results = {
                'silhouette_score': float(silhouette),
                'well_separable': bool(silhouette > 0.3),  # Threshold for good separation
                'num_samples': len(embeddings),
                'num_classes': len(np.unique(labels))
            }
            
            logger.info(f"  Silhouette score: {silhouette:.4f}")
            logger.info(f"  Well-separable: {'✅ YES (score > 0.3)' if results['well_separable'] else '❌ NO (score ≤ 0.3)'}")
            
            return results
        else:
            logger.warning("  Only one class found, cannot compute silhouette score")
            return {'silhouette_score': 0.0, 'well_separable': False, 'error': 'Only one class'}
    
    def analyze_class_distributions(self, embeddings, labels):
        """Analyze embedding distributions per class"""
        logger.info("Analyzing class distributions...")
        
        unique_labels = np.unique(labels)
        results = {}
        
        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            
            # Compute statistics
            mean_norm = np.mean(np.linalg.norm(class_embeddings, axis=1))
            std_norm = np.std(np.linalg.norm(class_embeddings, axis=1))
            mean_embedding = np.mean(class_embeddings, axis=0)
            
            label_name = "Normal" if label == 0 else f"Attack_{label}"
            results[label_name] = {
                'count': int(mask.sum()),
                'mean_norm': float(mean_norm),
                'std_norm': float(std_norm),
                'mean_embedding': mean_embedding.tolist()
            }
            
            logger.info(f"  {label_name}: {mask.sum()} samples, mean norm: {mean_norm:.4f} ± {std_norm:.4f}")
        
        return results
    
    def compute_prototype_accuracy(self, embeddings, labels, prototypes, unique_proto_labels):
        """Compute accuracy of prototype-based predictions"""
        logger.info("Computing prototype-based prediction accuracy...")
        
        # Compute distances from each embedding to all prototypes
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings, prototypes)
        
        # Predict class of nearest prototype
        predicted_labels = unique_proto_labels[np.argmin(distances, axis=1)]
        
        # Compute accuracy
        accuracy = np.mean(predicted_labels == labels)
        
        # Compute per-class accuracy
        unique_labels = np.unique(labels)
        per_class_accuracy = {}
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                label_name = "Normal" if label == 0 else f"Attack_{label}"
                per_class_accuracy[label_name] = float(np.mean(predicted_labels[mask] == labels[mask]))
                logger.info(f"  {label_name} accuracy: {per_class_accuracy[label_name]:.4f}")
        
        results = {
            'overall_accuracy': float(accuracy),
            'per_class_accuracy': per_class_accuracy
        }
        
        logger.info(f"  Overall prototype-based accuracy: {accuracy:.4f}")
        
        return results
    
    def run_full_diagnostic(self, X_test, y_test, X_support, y_support, output_dir="embedding_quality_diagnostics"):
        """Run complete diagnostic analysis"""
        logger.info("=" * 80)
        logger.info("STARTING EMBEDDING QUALITY DIAGNOSTIC")
        logger.info("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert to numpy if needed
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.cpu().numpy()
        if isinstance(X_support, torch.Tensor):
            X_support = X_support.cpu().numpy()
        if isinstance(y_support, torch.Tensor):
            y_support = y_support.cpu().numpy()
        
        # Convert to binary labels if multiclass
        y_test_binary = (y_test != 0).astype(int)
        y_support_binary = (y_support != 0).astype(int)
        
        results = {}
        
        # Step 1: Extract embeddings
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Extracting Embeddings")
        logger.info("=" * 80)
        test_embeddings = self.extract_embeddings(X_test)
        support_embeddings = self.extract_embeddings(X_support)
        logger.info(f"  Test embeddings shape: {test_embeddings.shape}")
        logger.info(f"  Support embeddings shape: {support_embeddings.shape}")
        results['embedding_shapes'] = {
            'test': test_embeddings.shape,
            'support': support_embeddings.shape
        }
        
        # Step 2: Compute prototypes
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Computing Prototypes")
        logger.info("=" * 80)
        prototypes, unique_proto_labels = self.compute_prototypes(X_support, y_support_binary)
        logger.info(f"  Prototypes shape: {prototypes.shape}")
        logger.info(f"  Unique prototype labels: {unique_proto_labels}")
        results['prototypes'] = {
            'shape': prototypes.shape,
            'unique_labels': unique_proto_labels.tolist()
        }
        
        # Step 3: Check prototype separation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Checking Prototype Separation")
        logger.info("=" * 80)
        separation_results = self.check_prototype_separation(prototypes, unique_proto_labels)
        results['prototype_separation'] = separation_results
        
        # Step 4: Check embedding separability
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Checking Embedding Separability")
        logger.info("=" * 80)
        separability_results = self.check_embedding_separability(test_embeddings, y_test_binary)
        results['embedding_separability'] = separability_results
        
        # Step 5: Analyze class distributions
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Analyzing Class Distributions")
        logger.info("=" * 80)
        distribution_results = self.analyze_class_distributions(test_embeddings, y_test_binary)
        results['class_distributions'] = distribution_results
        
        # Step 6: Compute prototype-based accuracy
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Computing Prototype-Based Accuracy")
        logger.info("=" * 80)
        accuracy_results = self.compute_prototype_accuracy(test_embeddings, y_test_binary, prototypes, unique_proto_labels)
        results['prototype_accuracy'] = accuracy_results
        
        # Step 7: Visualizations
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Creating Visualizations")
        logger.info("=" * 80)
        
        # t-SNE of test embeddings
        logger.info("  Creating t-SNE visualization of test embeddings...")
        self.visualize_embeddings_tsne(
            test_embeddings, y_test_binary,
            title="Test Embeddings t-SNE Visualization",
            save_path=output_path / "test_embeddings_tsne.png"
        )
        
        # t-SNE with prototypes
        logger.info("  Creating t-SNE visualization with prototypes...")
        self.visualize_prototypes(
            test_embeddings, y_test_binary, prototypes, unique_proto_labels,
            title="Test Embeddings with Prototypes",
            save_path=output_path / "embeddings_with_prototypes.png"
        )
        
        # Save results to JSON
        results_file = output_path / "embedding_quality_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✅ Saved results to {results_file}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Prototypes well-separated: {separation_results['well_separated']}")
        logger.info(f"✅ Embeddings well-separable: {separability_results.get('well_separable', False)}")
        logger.info(f"✅ Prototype-based accuracy: {accuracy_results['overall_accuracy']:.4f}")
        logger.info("=" * 80)
        
        return results


def main():
    """Main function to run embedding quality check"""
    import sys
    import config
    from models.transductive_fewshot_model import TransductiveLearner
    import pickle
    
    logger.info("Loading configuration and model...")
    # config is a module, not a class
    
    # Load preprocessed data - try multiple possible filenames
    preprocessed_files = [
        "preprocessed_data.pkl",
        "preprocessed_data_cicids.pkl",
        "preprocessed_data_unsw.pkl",
    ]
    preprocessed_data = None
    preprocessed_file = None
    
    for pf in preprocessed_files:
        if Path(pf).exists():
            preprocessed_file = pf
            logger.info(f"Found preprocessed data file: {pf}")
            try:
                with open(pf, 'rb') as f:
                    preprocessed_data = pickle.load(f)
                logger.info(f"✅ Successfully loaded preprocessed data from {pf}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {pf}: {e}")
                continue
    
    if preprocessed_data is None:
        logger.error(f"❌ Preprocessed data file not found in: {preprocessed_files}")
        logger.error("   Please run main.py first to generate preprocessed data")
        logger.info("   Looking for any .pkl files in current directory...")
        pkl_files = list(Path('.').glob('*.pkl'))
        if pkl_files:
            logger.info(f"   Found {len(pkl_files)} .pkl files: {[str(f) for f in pkl_files[:5]]}")
        sys.exit(1)
    
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']
    X_val = preprocessed_data['X_val']
    y_val = preprocessed_data['y_val']
    
    # Convert to binary labels
    y_test_binary = (y_test != 0).astype(int)
    y_val_binary = (y_val != 0).astype(int)
    
    # Load model - try multiple methods
    model = None
    
    # Method 1: Try to load from main system if it exists
    try:
        from main import BlockchainFederatedIncentiveSystem
        logger.info("Attempting to load model from main system...")
        system = BlockchainFederatedIncentiveSystem(config)
        system.preprocessed_data = preprocessed_data
        # Initialize system to get coordinator
        if hasattr(system, 'initialize_system'):
            system.initialize_system()
        if hasattr(system, 'coordinator') and system.coordinator and hasattr(system.coordinator, 'model'):
            model = system.coordinator.model
            logger.info("✅ Loaded model from system coordinator")
    except Exception as e:
        logger.warning(f"Could not load from main system: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    # Method 2: Try to load from saved model file
    if model is None:
        model_files = [
            "best_global_model.pth",
            "global_model.pth",
            "coordinator_model.pth"
        ]
        for model_file in model_files:
            if Path(model_file).exists():
                try:
                    logger.info(f"Attempting to load from {model_file}...")
                    input_dim = X_test.shape[-1]
                    model = TransductiveLearner(
                        input_dim=input_dim,
                        hidden_dim=config.hidden_dim,
                        embedding_dim=config.embedding_dim,
                        num_classes=2,
                        sequence_length=config.sequence_length,
                        tcn_kernel_sizes=config.tcn_kernel_sizes
                    )
                    checkpoint = torch.load(model_file, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                    else:
                        model.load_state_dict(checkpoint)
                    logger.info(f"✅ Loaded model from {model_file}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load from {model_file}: {e}")
    
    # Method 3: Create new model (fallback)
    if model is None:
        logger.warning("⚠️  Could not load trained model, creating newly initialized model")
        logger.warning("   This will show embeddings from untrained model - results may not be meaningful")
        input_dim = X_test.shape[-1]
        model = TransductiveLearner(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_classes=2,
            sequence_length=config.sequence_length,
            tcn_kernel_sizes=config.tcn_kernel_sizes
        )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Use validation set as support set (similar to base model evaluation)
    support_size = min(200, len(X_val))
    support_indices = np.random.RandomState(42).permutation(len(X_val))[:support_size]
    X_support = X_val[support_indices]
    y_support = y_val_binary[support_indices]
    
    # Use subset of test set for faster computation
    test_size = min(1000, len(X_test))
    test_indices = np.random.RandomState(42).permutation(len(X_test))[:test_size]
    X_test_subset = X_test[test_indices]
    y_test_subset = y_test_binary[test_indices]
    
    logger.info(f"Using {len(X_support)} support samples and {len(X_test_subset)} test samples")
    
    # Run diagnostic
    checker = EmbeddingQualityChecker(model, device)
    results = checker.run_full_diagnostic(
        X_test_subset, y_test_subset,
        X_support, y_support,
        output_dir="embedding_quality_diagnostics"
    )
    
    logger.info("\n✅ Embedding quality diagnostic completed!")
    logger.info(f"   Results saved to: embedding_quality_diagnostics/")


if __name__ == "__main__":
    main()

