#!/usr/bin/env python3
"""
Simple Embedding Quality Diagnostic
Checks embedding separability and prototype separation
"""

import torch
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def check_embedding_quality(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    output_dir: str = "embedding_quality_diagnostics"
) -> Dict[str, Any]:
    """
    Check embedding quality: separability and prototype separation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_val: Validation features (for prototype computation)
        y_val: Validation labels (for prototype computation)
        output_dir: Output directory for diagnostics
        
    Returns:
        Dictionary with embedding quality metrics
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        device = next(model.parameters()).device
        model.eval()
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_np = y_test if isinstance(y_test, np.ndarray) else np.array(y_test)
        
        # Get test embeddings
        with torch.no_grad():
            test_embeddings = model(X_test_tensor)
            if isinstance(test_embeddings, tuple):
                test_embeddings = test_embeddings[0]
            test_embeddings = test_embeddings.cpu().numpy()
        
        # Compute prototypes from validation set (or test set if no validation)
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_np = y_val if isinstance(y_val, np.ndarray) else np.array(y_val)
            
            with torch.no_grad():
                val_embeddings = model(X_val_tensor)
                if isinstance(val_embeddings, tuple):
                    val_embeddings = val_embeddings[0]
                val_embeddings = val_embeddings.cpu().numpy()
            
            # Use validation for prototypes
            support_embeddings = val_embeddings
            support_labels = y_val_np
        else:
            # Fallback to test set
            support_embeddings = test_embeddings
            support_labels = y_test_np
        
        # Compute prototypes (mean embedding per class)
        unique_labels = np.unique(support_labels)
        prototypes = []
        prototype_labels = []
        
        for label in unique_labels:
            mask = support_labels == label
            if mask.sum() > 0:
                prototype = support_embeddings[mask].mean(axis=0)
                prototypes.append(prototype)
                prototype_labels.append(label)
        
        prototypes = np.array(prototypes)
        
        # 1. Prototype Separation
        num_prototypes = len(prototypes)
        if num_prototypes >= 2:
            # Compute pairwise distances
            distances = []
            for i in range(num_prototypes):
                for j in range(i + 1, num_prototypes):
                    dist = np.linalg.norm(prototypes[i] - prototypes[j])
                    distances.append(dist)
            
            min_inter_class_distance = min(distances) if distances else 0.0
            avg_inter_class_distance = np.mean(distances) if distances else 0.0
            
            # Well-separated if min distance > 1.0
            well_separated = min_inter_class_distance > 1.0
            
            # Distance matrix
            distances_matrix = np.full((num_prototypes, num_prototypes), np.inf)
            for i in range(num_prototypes):
                for j in range(num_prototypes):
                    if i != j:
                        distances_matrix[i, j] = np.linalg.norm(prototypes[i] - prototypes[j])
        else:
            min_inter_class_distance = 0.0
            avg_inter_class_distance = 0.0
            well_separated = False
            distances_matrix = np.array([[np.inf]])
        
        # 2. Embedding Separability (Silhouette Score)
        if len(unique_labels) >= 2 and len(test_embeddings) >= 2:
            try:
                silhouette = silhouette_score(test_embeddings, y_test_np)
                well_separable = silhouette > 0.3  # Threshold for good separability
            except Exception as e:
                logger.warning(f"Silhouette score computation failed: {e}")
                silhouette = 0.0
                well_separable = False
        else:
            silhouette = 0.0
            well_separable = False
        
        # 3. Prototype-based Accuracy
        # Classify test samples based on distance to prototypes
        test_distances = []
        for embedding in test_embeddings:
            dists = [np.linalg.norm(embedding - proto) for proto in prototypes]
            test_distances.append(dists)
        
        test_distances = np.array(test_distances)
        predicted_labels = np.array([prototype_labels[np.argmin(dists)] for dists in test_distances])
        
        prototype_accuracy = (predicted_labels == y_test_np).mean()
        
        # Class distributions
        class_distributions = {}
        for label in unique_labels:
            mask = y_test_np == label
            if mask.sum() > 0:
                class_embeddings = test_embeddings[mask]
                class_norms = np.linalg.norm(class_embeddings, axis=1)
                class_distributions[f"Class_{label}"] = {
                    "count": int(mask.sum()),
                    "mean_norm": float(np.mean(class_norms)),
                    "std_norm": float(np.std(class_norms)),
                    "mean_embedding": class_embeddings.mean(axis=0).tolist()
                }
        
        # Create results dictionary
        results = {
            "embedding_shapes": {
                "test": list(test_embeddings.shape),
                "support": list(support_embeddings.shape)
            },
            "prototypes": {
                "shape": list(prototypes.shape),
                "unique_labels": prototype_labels
            },
            "prototype_separation": {
                "num_prototypes": num_prototypes,
                "min_inter_class_distance": float(min_inter_class_distance),
                "avg_inter_class_distance": float(avg_inter_class_distance),
                "well_separated": well_separated,
                "distances_matrix": distances_matrix.tolist()
            },
            "embedding_separability": {
                "silhouette_score": float(silhouette),
                "well_separable": well_separable,
                "num_samples": len(test_embeddings),
                "num_classes": len(unique_labels)
            },
            "prototype_accuracy": {
                "overall_accuracy": float(prototype_accuracy),
                "num_samples": len(y_test_np)
            },
            "class_distributions": class_distributions
        }
        
        # Save results
        results_path = os.path.join(output_dir, "embedding_quality_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create t-SNE visualization
        try:
            if len(test_embeddings) > 10:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_embeddings) - 1))
                embeddings_2d = tsne.fit_transform(test_embeddings)
                
                plt.figure(figsize=(10, 8))
                for label in unique_labels:
                    mask = y_test_np == label
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                              label=f"Class {label}", alpha=0.6)
                
                # Plot prototypes
                if len(prototypes) > 0:
                    proto_2d = tsne.fit_transform(prototypes)
                    plt.scatter(proto_2d[:, 0], proto_2d[:, 1], 
                              marker='*', s=500, c='red', edgecolors='black', 
                              linewidths=2, label='Prototypes', zorder=5)
                
                plt.title("t-SNE Visualization of Test Embeddings with Prototypes")
                plt.xlabel("t-SNE Component 1")
                plt.ylabel("t-SNE Component 2")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                tsne_path = os.path.join(output_dir, "test_embeddings_tsne.png")
                plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"t-SNE visualization failed: {e}")
        
        logger.info(f"✅ Embedding quality check completed")
        logger.info(f"   Prototype separation: {min_inter_class_distance:.4f} (well-separated: {well_separated})")
        logger.info(f"   Embedding separability: {silhouette:.4f} (well-separable: {well_separable})")
        logger.info(f"   Prototype-based accuracy: {prototype_accuracy:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Embedding quality check failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            "error": str(e),
            "prototype_separation": {"well_separated": False},
            "embedding_separability": {"well_separable": False},
            "prototype_accuracy": {"overall_accuracy": 0.0}
        }
