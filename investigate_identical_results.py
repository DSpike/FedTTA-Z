#!/usr/bin/env python3
"""
Investigate why base and TTT models show identical zero-day results
"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_identical_results():
    """
    Investigate why base and TTT models show identical zero-day results
    """
    logger.info("=" * 80)
    logger.info("INVESTIGATING IDENTICAL BASE/TTT ZERO-DAY RESULTS")
    logger.info("=" * 80)
    
    # Read main.py to check the evaluation flow
    logger.info("\n🔍 Checking TTT evaluation code...")
    
    issues_found = []
    
    # Issue 1: Model mismatch
    logger.info("\n1️⃣ CHECKING: Model consistency between base and TTT evaluation")
    logger.info("   Location: main.py _evaluate_ttt_model()")
    logger.info("   ⚠️  POTENTIAL ISSUE: Base predictions use 'binary_model' (line 6361)")
    logger.info("   ⚠️  POTENTIAL ISSUE: TTT adaptation adapts 'coordinator.model' (line 6382)")
    logger.info("   ⚠️  POTENTIAL ISSUE: These are DIFFERENT models!")
    issues_found.append({
        'issue': 'Model Mismatch',
        'description': 'Base predictions use binary_model, but TTT adapts coordinator.model',
        'location': 'main.py lines 6312-6382',
        'impact': 'Base and TTT predictions come from different models, making comparison invalid'
    })
    
    # Issue 2: TTT might not be adapting
    logger.info("\n2️⃣ CHECKING: TTT adaptation execution")
    logger.info("   Location: coordinators/centralized_coordinator.py adapt_to_test_data()")
    logger.info("   ⚠️  POTENTIAL ISSUE: TTT might skip adaptation if confidence is high")
    logger.info("   ⚠️  POTENTIAL ISSUE: If adaptation is skipped, adapted_model = base model")
    issues_found.append({
        'issue': 'TTT Adaptation Skipped',
        'description': 'TTT might skip adaptation if base model confidence > threshold',
        'location': 'coordinators/centralized_coordinator.py',
        'impact': 'If skipped, adapted_model is identical to base model'
    })
    
    # Issue 3: Binary conversion making predictions identical
    logger.info("\n3️⃣ CHECKING: Binary prediction conversion")
    logger.info("   Location: main.py lines 6742, 7044")
    logger.info("   ⚠️  POTENTIAL ISSUE: Both use (predictions != 0) for binary conversion")
    logger.info("   ⚠️  POTENTIAL ISSUE: If multiclass predictions are same, binary will be identical")
    issues_found.append({
        'issue': 'Binary Conversion',
        'description': 'Both models use (predictions != 0) for binary conversion',
        'location': 'main.py lines 6742, 7044',
        'impact': 'Even if probabilities differ, binary predictions might be identical'
    })
    
    # Issue 4: Same threshold used
    logger.info("\n4️⃣ CHECKING: Threshold selection")
    logger.info("   Location: main.py lines 6543-6634")
    logger.info("   ⚠️  POTENTIAL ISSUE: Both might use same optimal threshold")
    logger.info("   ⚠️  POTENTIAL ISSUE: If threshold is same, binary predictions will be identical")
    issues_found.append({
        'issue': 'Same Threshold',
        'description': 'Both models might use the same optimal threshold',
        'location': 'main.py threshold optimization',
        'impact': 'Same threshold + similar probabilities = identical binary predictions'
    })
    
    # Issue 5: TTT entropy minimization doesn't change zero-day predictions
    logger.info("\n5️⃣ CHECKING: TTT optimization strategy")
    logger.info("   Location: coordinators/centralized_coordinator.py lines 495-513")
    logger.info("   ⚠️  POTENTIAL ISSUE: Entropy minimization optimizes for overall confidence")
    logger.info("   ⚠️  POTENTIAL ISSUE: Zero-day samples are minority (30%), gradient dominated by majority (70%)")
    logger.info("   ⚠️  POTENTIAL ISSUE: TTT might not change zero-day predictions significantly")
    issues_found.append({
        'issue': 'TTT Optimization Bias',
        'description': 'Entropy minimization is dominated by non-zero-day samples (70%)',
        'location': 'coordinators/centralized_coordinator.py TTT loop',
        'impact': 'TTT doesn\'t specifically improve zero-day detection'
    })
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY OF ISSUES FOUND")
    logger.info("=" * 80)
    
    for i, issue in enumerate(issues_found, 1):
        logger.info(f"\n{i}. {issue['issue']}")
        logger.info(f"   Description: {issue['description']}")
        logger.info(f"   Location: {issue['location']}")
        logger.info(f"   Impact: {issue['impact']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDED FIXES")
    logger.info("=" * 80)
    
    logger.info("\n1. FIX MODEL CONSISTENCY:")
    logger.info("   - Use same model for base and TTT predictions")
    logger.info("   - Either: Adapt binary_model for TTT, OR use coordinator.model for base predictions")
    
    logger.info("\n2. VERIFY TTT ADAPTATION:")
    logger.info("   - Check if TTT adaptation is actually running")
    logger.info("   - Log model parameter changes before/after TTT")
    logger.info("   - Verify adapted_model != base model")
    
    logger.info("\n3. ADD PREDICTION COMPARISON:")
    logger.info("   - Log prediction differences: (base_predictions != ttt_predictions).sum()")
    logger.info("   - Log probability differences for zero-day samples")
    logger.info("   - Check if predictions are actually identical or just metrics are same")
    
    logger.info("\n4. IMPROVE TTT FOR ZERO-DAY:")
    logger.info("   - Weight zero-day samples more heavily in TTT loss")
    logger.info("   - Use zero-day weighted entropy minimization")
    logger.info("   - Add explicit zero-day detection loss component")
    
    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    investigate_identical_results()



