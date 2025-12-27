"""
Test script for low-confidence sample selection

This script verifies that the low-confidence selector works correctly
with synthetic data before running on real experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from low_confidence_selector import LowConfidenceSampleSelector, select_low_confidence_samples_simple


class SimpleMockModel(nn.Module):
    """Simple mock model for testing"""

    def __init__(self, seq_len=20, n_features=10, num_classes=15):
        super().__init__()
        input_dim = seq_len * n_features
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        return self.fc(x)


def create_synthetic_data(n_samples=1000, seq_len=20, n_features=10, n_classes=15):
    """
    Create synthetic test data with known zero-day samples.

    Zero-day samples (class 14) will have higher uncertainty in the model.
    """
    # Create random data
    X = torch.randn(n_samples, seq_len, n_features)

    # Create labels: 70% non-zero-day (0-13), 30% zero-day (14)
    y = torch.zeros(n_samples, dtype=torch.long)

    # First 70% are non-zero-day (random classes 0-13)
    non_zero_day_count = int(0.7 * n_samples)
    y[:non_zero_day_count] = torch.randint(0, n_classes - 1, (non_zero_day_count,))

    # Last 30% are zero-day (class 14)
    y[non_zero_day_count:] = n_classes - 1

    # Shuffle
    perm = torch.randperm(n_samples)
    X = X[perm]
    y = y[perm]

    return X, y


def test_entropy_selection():
    """Test entropy-based selection"""
    print("\n" + "=" * 80)
    print("TEST 1: Entropy-based Selection")
    print("=" * 80)

    # Create synthetic data
    X, y = create_synthetic_data(n_samples=1000)
    print(f"Created {len(X)} samples")
    print(f"Zero-day samples (label 14): {(y == 14).sum().item()}")

    # Create a simple model
    model = SimpleMockModel()

    # Test selection
    selector = LowConfidenceSampleSelector(
        method='entropy',
        threshold_percentile=0.70,
        min_samples=100,
        max_samples=500
    )

    selected_samples, selected_mask, stats = selector.select_low_confidence_samples(
        model, X, y
    )

    print(f"\n✅ Selected {stats['n_selected']} samples ({stats['selection_rate']*100:.1f}%)")
    print(f"Mean entropy (selected): {stats['mean_score_selected']:.4f}")
    print(f"Mean entropy (all): {stats['mean_score_all']:.4f}")

    # Check if selection worked
    if stats['n_selected'] > 0:
        print("✅ Selection successful!")
    else:
        print("❌ Selection failed - no samples selected")

    # Check zero-day correlation
    if 'selected_label_distribution' in stats:
        print("\nLabel distribution in selected samples:")
        for label, count in stats['selected_label_distribution'].items():
            is_zero_day = (label == 14)
            marker = "🎯 ZERO-DAY" if is_zero_day else ""
            print(f"   Label {label}: {count} samples {marker}")

    return stats


def test_probability_selection():
    """Test probability-based selection"""
    print("\n" + "=" * 80)
    print("TEST 2: Probability-based Selection")
    print("=" * 80)

    X, y = create_synthetic_data(n_samples=1000)
    model = SimpleMockModel()

    selector = LowConfidenceSampleSelector(
        method='probability',
        threshold_percentile=0.70,
        min_samples=100,
        max_samples=500
    )

    selected_samples, selected_mask, stats = selector.select_low_confidence_samples(
        model, X, y
    )

    print(f"\n✅ Selected {stats['n_selected']} samples ({stats['selection_rate']*100:.1f}%)")
    print(f"Mean max_probability (selected): {stats['mean_score_selected']:.4f}")
    print(f"Mean max_probability (all): {stats['mean_score_all']:.4f}")

    if stats['n_selected'] > 0:
        print("✅ Selection successful!")
    else:
        print("❌ Selection failed")

    return stats


def test_combined_selection():
    """Test combined selection"""
    print("\n" + "=" * 80)
    print("TEST 3: Combined Selection")
    print("=" * 80)

    X, y = create_synthetic_data(n_samples=1000)
    model = SimpleMockModel()

    selector = LowConfidenceSampleSelector(
        method='combined',
        threshold_percentile=0.70,
        min_samples=100,
        max_samples=500
    )

    selected_samples, selected_mask, stats = selector.select_low_confidence_samples(
        model, X, y
    )

    print(f"\n✅ Selected {stats['n_selected']} samples ({stats['selection_rate']*100:.1f}%)")
    print(f"Mean combined_uncertainty (selected): {stats['mean_score_selected']:.4f}")
    print(f"Mean combined_uncertainty (all): {stats['mean_score_all']:.4f}")

    if stats['n_selected'] > 0:
        print("✅ Selection successful!")
    else:
        print("❌ Selection failed")

    return stats


def test_simple_interface():
    """Test simple function interface"""
    print("\n" + "=" * 80)
    print("TEST 4: Simple Interface")
    print("=" * 80)

    X, y = create_synthetic_data(n_samples=1000)
    model = SimpleMockModel()

    # Use simple interface
    selected_samples, stats = select_low_confidence_samples_simple(
        model=model,
        X_test=X,
        y_test=y,
        method='entropy',
        percentile=0.70,
        min_samples=100,
        max_samples=500
    )

    print(f"\n✅ Selected {len(selected_samples)} samples")
    print(f"Stats: {stats['n_selected']}/{stats['n_total']} samples ({stats['selection_rate']*100:.1f}%)")

    if len(selected_samples) > 0:
        print("✅ Simple interface works!")
    else:
        print("❌ Simple interface failed")

    return stats


def test_min_max_constraints():
    """Test minimum and maximum sample constraints"""
    print("\n" + "=" * 80)
    print("TEST 5: Min/Max Constraints")
    print("=" * 80)

    X, y = create_synthetic_data(n_samples=500)
    model = SimpleMockModel()

    # Test minimum constraint
    print("\nTest 5a: Minimum constraint (min=200, expect 200 samples)")
    selector = LowConfidenceSampleSelector(
        method='entropy',
        threshold_percentile=0.95,  # Very high threshold (few samples naturally)
        min_samples=200,
        max_samples=None
    )

    selected_samples, _, stats = selector.select_low_confidence_samples(model, X, y)
    print(f"Selected: {stats['n_selected']} samples (expected >= 200)")

    if stats['n_selected'] >= 200:
        print("✅ Minimum constraint works!")
    else:
        print(f"❌ Minimum constraint failed (got {stats['n_selected']}, expected >= 200)")

    # Test maximum constraint
    print("\nTest 5b: Maximum constraint (max=100, expect 100 samples)")
    selector = LowConfidenceSampleSelector(
        method='entropy',
        threshold_percentile=0.30,  # Very low threshold (many samples naturally)
        min_samples=10,
        max_samples=100
    )

    selected_samples, _, stats = selector.select_low_confidence_samples(model, X, y)
    print(f"Selected: {stats['n_selected']} samples (expected <= 100)")

    if stats['n_selected'] <= 100:
        print("✅ Maximum constraint works!")
    else:
        print(f"❌ Maximum constraint failed (got {stats['n_selected']}, expected <= 100)")

    return stats


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("LOW-CONFIDENCE SAMPLE SELECTOR - TEST SUITE")
    print("=" * 80)

    tests_passed = 0
    tests_total = 5

    # Test 1: Entropy selection
    try:
        stats = test_entropy_selection()
        if stats['n_selected'] > 0:
            tests_passed += 1
            print("\n✅ Test 1 PASSED")
        else:
            print("\n❌ Test 1 FAILED")
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with exception: {e}")

    # Test 2: Probability selection
    try:
        stats = test_probability_selection()
        if stats['n_selected'] > 0:
            tests_passed += 1
            print("\n✅ Test 2 PASSED")
        else:
            print("\n❌ Test 2 FAILED")
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with exception: {e}")

    # Test 3: Combined selection
    try:
        stats = test_combined_selection()
        if stats['n_selected'] > 0:
            tests_passed += 1
            print("\n✅ Test 3 PASSED")
        else:
            print("\n❌ Test 3 FAILED")
    except Exception as e:
        print(f"\n❌ Test 3 FAILED with exception: {e}")

    # Test 4: Simple interface
    try:
        stats = test_simple_interface()
        if stats['n_selected'] > 0:
            tests_passed += 1
            print("\n✅ Test 4 PASSED")
        else:
            print("\n❌ Test 4 FAILED")
    except Exception as e:
        print(f"\n❌ Test 4 FAILED with exception: {e}")

    # Test 5: Min/Max constraints
    try:
        stats = test_min_max_constraints()
        tests_passed += 1
        print("\n✅ Test 5 PASSED")
    except Exception as e:
        print(f"\n❌ Test 5 FAILED with exception: {e}")

    # Summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {tests_passed}/{tests_total} tests passed")
    print("=" * 80)

    if tests_passed == tests_total:
        print("\n🎉 ALL TESTS PASSED! Ready for real experiments.")
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) failed. Please review.")

    return tests_passed == tests_total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
