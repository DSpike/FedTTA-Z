"""
QUICK TTT DIAGNOSTIC SCRIPT

Run this IMMEDIATELY to find the problem!

Usage:
    python ttt_quick_diagnostic.py
"""

import torch
import torch.nn as nn
import numpy as np


def quick_diagnostic(coordinator, query_x, query_y, config):
    """
    5-minute diagnostic to find why TTT isn't working
    
    Args:
        coordinator: Your SimpleFedAVGCoordinator instance
        query_x: Test data (torch.Tensor)
        query_y: Test labels (torch.Tensor)
        config: Your config object
    """
    
    print("\n" + "=" * 80)
    print("TTT QUICK DIAGNOSTIC")
    print("=" * 80 + "\n")
    
    issues_found = []
    
    # ==========================================
    # CHECK 1: BATCH NORM (MOST COMMON ISSUE)
    # ==========================================
    print("CHECK 1: Batch Norm Layers")
    print("-" * 80)
    
    bn_count = 0
    for module in coordinator.model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_count += 1
    
    print(f"Batch norm layers found: {bn_count}")
    
    if bn_count == 0:
        print("❌ CRITICAL: NO BATCH NORM LAYERS!")
        print("   → This is why TTT doesn't work")
        print("   → TENT REQUIRES batch norm to adapt")
        print("\n   FIX: Add batch norm layers to your model:")
        print("   ```python")
        print("   self.bn1 = nn.BatchNorm1d(hidden_dim)")
        print("   x = torch.relu(self.bn1(self.fc1(x)))")
        print("   ```\n")
        issues_found.append("NO_BATCH_NORM")
        return issues_found  # Critical - can't continue
    else:
        print("✅ Batch norm layers present\n")
    
    # ==========================================
    # CHECK 2: BASE MODEL PERFORMANCE
    # ==========================================
    print("CHECK 2: Base Model Performance")
    print("-" * 80)
    
    coordinator.model.eval()
    with torch.no_grad():
        base_outputs = coordinator.model(query_x)
        base_preds = base_outputs.argmax(dim=1)
        base_acc = (base_preds == query_y).float().mean().item()
        base_probs = torch.softmax(base_outputs, dim=1)
        base_conf = base_probs.max(dim=1)[0].mean().item()
        base_entropy = -torch.sum(base_probs * torch.log(base_probs + 1e-8), dim=1).mean().item()
    
    print(f"Base accuracy:   {base_acc:.3f}")
    print(f"Base confidence: {base_conf:.3f}")
    print(f"Base entropy:    {base_entropy:.3f}")
    
    if base_acc < 0.55:
        print("❌ PROBLEM: Base model too poor (<55%)")
        print("   → TTT can't fix a fundamentally bad model")
        print("   → FIX: Improve base model training first\n")
        issues_found.append("BASE_TOO_POOR")
    elif base_acc > 0.95:
        print("⚠️  Base model already excellent (>95%)")
        print("   → Not much room for improvement")
        print("   → TTT might only give +0-2%\n")
        issues_found.append("BASE_TOO_GOOD")
    elif base_conf > 0.92:
        print("⚠️  Base model very confident (>92%)")
        print("   → Might skip adaptation (by design)")
        print("   → This is expected behavior\n")
        issues_found.append("BASE_TOO_CONFIDENT")
    else:
        print("✅ Base model in good range for TTT\n")
    
    # ==========================================
    # CHECK 3: DATA QUALITY
    # ==========================================
    print("CHECK 3: Data Quality")
    print("-" * 80)
    
    print(f"Number of samples: {len(query_x)}")
    print(f"Number of classes: {len(torch.unique(query_y))}")
    print(f"Class distribution: {torch.bincount(query_y).tolist()}")
    
    if len(query_x) < 50:
        print("⚠️  Very few samples (<50)")
        print("   → TTT works better with 100+ samples")
        print("   → Might see unstable results\n")
        issues_found.append("FEW_SAMPLES")
    
    if len(torch.unique(query_y)) < 2:
        print("❌ CRITICAL: Only one class!")
        print("   → Need samples from multiple classes")
        print("   → Cannot evaluate properly\n")
        issues_found.append("SINGLE_CLASS")
        return issues_found
    else:
        print("✅ Data quality acceptable\n")
    
    # ==========================================
    # CHECK 4: WHICH METHOD ARE YOU USING?
    # ==========================================
    print("CHECK 4: Method Check")
    print("-" * 80)
    print("Are you calling the RIGHT method?")
    print()
    print("❌ OLD (broken):")
    print("   coordinator._perform_advanced_ttt_adaptation(...)")
    print()
    print("✅ NEW (correct):")
    print("   coordinator.adapt_to_test_data(query_x, method='tent_pseudo')")
    print()
    print("⚠️  Make sure you're using the NEW method!\n")
    
    # ==========================================
    # CHECK 5: QUICK ADAPTATION TEST
    # ==========================================
    print("CHECK 5: Quick Adaptation Test")
    print("-" * 80)
    print("Running 10-step adaptation test...")
    
    try:
        # Test with small subset
        test_size = min(100, len(query_x))
        test_x = query_x[:test_size]
        test_y = query_y[:test_size]
        
        # Import and use TENTPseudoLabels
        try:
            from tent_pseudo_labels_implementation import TENTPseudoLabels
        except:
            print("❌ Cannot import TENTPseudoLabels")
            print("   Make sure tent_pseudo_labels_implementation.py is in path")
            issues_found.append("IMPORT_ERROR")
            return issues_found
        
        # Create adapter
        adapter = TENTPseudoLabels(
            model=coordinator.model,
            initial_threshold=0.9,
            min_threshold=0.7,
            pseudo_label_weight=1.0,
            entropy_weight=0.1,
            use_temporal_consistency=True
        )
        
        # Quick adaptation (only 10 steps for speed)
        adapted_model, stats = adapter.adapt(
            query_x=test_x,
            query_y=test_y,
            num_steps=10,
            batch_size=32,
            lr=0.00025
        )
        
        # Evaluate adapted model
        adapted_model.eval()
        with torch.no_grad():
            adapted_outputs = adapted_model(test_x)
            adapted_preds = adapted_outputs.argmax(dim=1)
            adapted_acc = (adapted_preds == test_y).float().mean().item()
            adapted_probs = torch.softmax(adapted_outputs, dim=1)
            adapted_conf = adapted_probs.max(dim=1)[0].mean().item()
        
        print(f"\nResults after 10 steps:")
        print(f"  Base accuracy:    {base_acc:.3f}")
        print(f"  Adapted accuracy: {adapted_acc:.3f}")
        print(f"  Change:           {adapted_acc - base_acc:+.3f}")
        print(f"  Base confidence:    {base_conf:.3f}")
        print(f"  Adapted confidence: {adapted_conf:.3f}")
        
        # Check pseudo-labels
        pseudo_counts = stats['pseudo_labels_generated']
        avg_pseudo = np.mean(pseudo_counts)
        print(f"\n  Pseudo-labels per step: {avg_pseudo:.1f} / {test_size}")
        print(f"  Pseudo-label ratio: {avg_pseudo/test_size:.1%}")
        
        if avg_pseudo < test_size * 0.05:
            print("\n  ⚠️  Very few pseudo-labels (<5%)")
            print("     → Model not confident enough")
            print("     → Try: config.pseudo_threshold = 0.85")
            issues_found.append("FEW_PSEUDO_LABELS")
        
        # Check entropy change
        initial_entropy = stats['entropy_history'][0]
        final_entropy = stats['entropy_history'][-1]
        entropy_change = final_entropy - initial_entropy
        
        print(f"\n  Initial entropy: {initial_entropy:.3f}")
        print(f"  Final entropy:   {final_entropy:.3f}")
        print(f"  Change:          {entropy_change:+.3f}")
        
        if entropy_change >= 0:
            print("\n  ❌ Entropy not decreasing!")
            print("     → Model not learning")
            print("     → Try: config.ttt_lr = 0.0001 (lower LR)")
            issues_found.append("NO_LEARNING")
        
        # Final verdict
        print()
        if adapted_acc > base_acc + 0.01:
            print("✅ ADAPTATION WORKING!")
            print("   → Seeing improvement after just 10 steps")
            print("   → Try full adaptation: config.ttt_steps = 100")
        elif adapted_acc > base_acc - 0.01 and adapted_acc < base_acc + 0.01:
            print("⚠️  NO CHANGE")
            print("   → Need more steps or better config")
            print("   → Try: config.ttt_steps = 150")
            issues_found.append("NO_IMPROVEMENT")
        else:
            print("❌ GETTING WORSE!")
            print("   → Something is wrong")
            print("   → Check issues below")
            issues_found.append("DEGRADATION")
        
    except Exception as e:
        print(f"\n❌ ERROR during adaptation test:")
        print(f"   {str(e)}")
        print("\n   Stacktrace:")
        import traceback
        traceback.print_exc()
        issues_found.append("ADAPTATION_ERROR")
    
    print()
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print()
    
    if len(issues_found) == 0:
        print("✅ NO MAJOR ISSUES FOUND")
        print()
        print("If TTT still not working:")
        print("1. Run with more steps: config.ttt_steps = 150")
        print("2. Try lower LR: config.ttt_lr = 0.0001")
        print("3. Check you're using the correct method")
    else:
        print(f"Found {len(issues_found)} issue(s):")
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. {issue}")
        
        print()
        print("RECOMMENDED FIXES:")
        print()
        
        if "NO_BATCH_NORM" in issues_found:
            print("🔥 CRITICAL: Add batch norm layers to your model!")
            print("   This is the #1 reason TTT doesn't work")
        
        if "BASE_TOO_POOR" in issues_found:
            print("→ Improve base model training first")
        
        if "BASE_TOO_GOOD" in issues_found or "BASE_TOO_CONFIDENT" in issues_found:
            print("→ Base model already good - TTT won't help much")
        
        if "FEW_PSEUDO_LABELS" in issues_found:
            print("→ Lower threshold: config.pseudo_threshold = 0.85")
        
        if "NO_LEARNING" in issues_found:
            print("→ Lower LR: config.ttt_lr = 0.0001")
            print("→ More steps: config.ttt_steps = 150")
        
        if "NO_IMPROVEMENT" in issues_found:
            print("→ More steps: config.ttt_steps = 150")
            print("→ Tune hyperparameters")
    
    print()
    print("=" * 80)
    
    return issues_found


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nTo use this diagnostic:")
    print()
    print("```python")
    print("from ttt_quick_diagnostic import quick_diagnostic")
    print()
    print("issues = quick_diagnostic(")
    print("    coordinator=your_coordinator,")
    print("    query_x=your_test_data,")
    print("    query_y=your_test_labels,")
    print("    config=your_config")
    print(")")
    print("```")
    print()
    print("The script will identify the problem and suggest fixes!")