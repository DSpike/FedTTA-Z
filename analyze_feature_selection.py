#!/usr/bin/env python3
"""
Analyze IGRF-RFE Feature Selection Results
"""

import pandas as pd
import numpy as np
import torch
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_feature_selection():
    """Analyze the IGRF-RFE feature selection results"""
    
    logger.info("🔍 Analyzing IGRF-RFE Feature Selection Results")
    logger.info("=" * 60)
    
    # Initialize preprocessor
    preprocessor = UNSWPreprocessor()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    # Get feature importance scores
    if hasattr(preprocessor, 'feature_importance_scores'):
        feature_scores = preprocessor.feature_importance_scores
        
        logger.info(f"\n📊 Feature Selection Analysis:")
        logger.info(f"Total features before selection: {len(feature_scores)}")
        logger.info(f"Selected features: {len(preprocessor.selected_features)}")
        
        # Show top 20 features by hybrid score
        logger.info(f"\n🏆 Top 20 Features by Hybrid Score:")
        top_features = feature_scores.head(20)
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']:25} | IG: {row['ig_score']:.4f} | RF: {row['rf_importance']:.4f} | Hybrid: {row['hybrid_score']:.4f}")
        
        # Show bottom 10 features
        logger.info(f"\n🔻 Bottom 10 Features by Hybrid Score:")
        bottom_features = feature_scores.tail(10)
        for idx, row in bottom_features.iterrows():
            logger.info(f"  {row['feature']:25} | IG: {row['ig_score']:.4f} | RF: {row['rf_importance']:.4f} | Hybrid: {row['hybrid_score']:.4f}")
        
        # Analyze feature types
        logger.info(f"\n📈 Feature Type Analysis:")
        selected_features = preprocessor.selected_features
        
        # Count by feature type
        feature_types = {}
        for feature in selected_features:
            if 'proto' in feature:
                feature_types['Protocol'] = feature_types.get('Protocol', 0) + 1
            elif 'service' in feature:
                feature_types['Service'] = feature_types.get('Service', 0) + 1
            elif 'state' in feature:
                feature_types['State'] = feature_types.get('State', 0) + 1
            elif 'packet' in feature or 'tcp' in feature:
                feature_types['Network'] = feature_types.get('Network', 0) + 1
            elif 'dur' in feature or 'sbytes' in feature or 'dbytes' in feature:
                feature_types['Traffic'] = feature_types.get('Traffic', 0) + 1
            else:
                feature_types['Other'] = feature_types.get('Other', 0) + 1
        
        for ftype, count in feature_types.items():
            logger.info(f"  {ftype}: {count} features")
        
        # Check if important features were selected
        important_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'sload', 'dload']
        selected_important = [f for f in important_features if f in selected_features]
        logger.info(f"\n🎯 Important Features Selected: {len(selected_important)}/{len(important_features)}")
        for f in selected_important:
            logger.info(f"  ✅ {f}")
        for f in important_features:
            if f not in selected_features:
                logger.info(f"  ❌ {f}")
    
    else:
        logger.warning("Feature importance scores not available")
    
    # Compare with original performance
    logger.info(f"\n📊 Performance Comparison:")
    logger.info(f"Current results (with IGRF-RFE):")
    logger.info(f"  Base Model Accuracy: 0.7105")
    logger.info(f"  TTT Model Accuracy: 0.8208")
    logger.info(f"  Zero-day Detection Rate: 0.5542")
    
    # Check if we can run without feature selection for comparison
    logger.info(f"\n🔬 Testing without feature selection...")
    
    # Create a simple test to compare
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    # Get the data
    X_train = data['X_train'].numpy()
    y_train = data['y_train'].numpy()
    X_test = data['X_test'].numpy()
    y_test = data['y_test'].numpy()
    
    # Train a simple Random Forest on selected features
    rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selected.fit(X_train, y_train)
    
    # Predictions
    y_pred_selected = rf_selected.predict(X_test)
    
    acc_selected = accuracy_score(y_test, y_pred_selected)
    f1_selected = f1_score(y_test, y_pred_selected)
    
    logger.info(f"Random Forest with IGRF-RFE selected features:")
    logger.info(f"  Accuracy: {acc_selected:.4f}")
    logger.info(f"  F1-Score: {f1_selected:.4f}")
    
    # Feature importance from the trained model
    feature_importance = rf_selected.feature_importances_
    feature_names = data['feature_names']
    
    # Show top 10 most important features according to the trained model
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\n🎯 Top 10 Most Important Features (from trained model):")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:25} | Importance: {row['importance']:.4f}")
    
    return {
        'selected_features': preprocessor.selected_features,
        'feature_scores': feature_scores if hasattr(preprocessor, 'feature_importance_scores') else None,
        'rf_accuracy': acc_selected,
        'rf_f1': f1_selected,
        'feature_importance': importance_df
    }

if __name__ == "__main__":
    results = analyze_feature_selection()





