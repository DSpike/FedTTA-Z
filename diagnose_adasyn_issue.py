"""
Diagnose ADASYN failure: "No samples will be generated with the provided ratio settings"
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_adasyn_issue():
    """
    Identify why ADASYN fails with certain class distributions
    """
    print("=" * 80)
    print("🔍 ADASYN FAILURE DIAGNOSIS")
    print("=" * 80)
    
    print("\n📊 Common Reasons for ADASYN Failure:")
    print("\n1. **Insufficient Samples for n_neighbors**:")
    print("   - ADASYN requires at least (n_neighbors + 1) samples per class")
    print("   - If n_neighbors=5, class needs ≥ 6 samples")
    print("   - Error occurs when class has fewer samples than required")
    
    print("\n2. **Target Ratio Too Large**:")
    print("   - ADASYN calculates how many samples to generate")
    print("   - If target_count - current_count results in impossible ratio")
    print("   - ADASYN may fail with 'No samples will be generated'")
    
    print("\n3. **ADASYN Internal Ratio Calculation**:")
    print("   - ADASYN uses relative ratios (like 0.5, 1.0) internally")
    print("   - When you provide absolute counts, ADASYN converts to ratios")
    print("   - If the ratio calculation rounds to 0, it fails")
    
    print("\n4. **Feature Space Issues**:")
    print("   - ADASYN needs sufficient neighbors in feature space")
    print("   - If samples are too sparse, cannot find n_neighbors")
    print("   - High-dimensional or sparse data can cause failures")
    
    print("\n" + "=" * 80)
    print("🔧 SOLUTIONS")
    print("=" * 80)
    
    print("\n**Solution 1: Check Minimum Sample Requirements**")
    print("""
    Before calling ADASYN, check if each class has enough samples:
    
    ```python
    n_neighbors = 5
    min_samples_required = n_neighbors + 1  # At least 6 for n_neighbors=5
    
    for class_label, count in class_counts.items():
        if count < min_samples_required:
            logger.warning(f"Class {class_label} has only {count} samples, "
                          f"need {min_samples_required} for ADASYN")
            # Skip this class or use a different method
    ```""")
    
    print("\n**Solution 2: Use Relative Ratios Instead of Absolute Counts**")
    print("""
    ADASYN works better with ratios (0.5, 1.0, etc.) than absolute counts:
    
    ```python
    # Instead of: oversample_strategy = {0: 15170, 2: 15170}  # Absolute counts
    # Use ratios:
    oversample_strategy = {0: 0.5, 2: 0.5}  # Relative ratios
    # Or use 'auto', 'minority', etc.
    ```""")
    
    print("\n**Solution 3: Reduce n_neighbors for Small Classes**")
    print("""
    Adaptively reduce n_neighbors for classes with few samples:
    
    ```python
    # Calculate adaptive n_neighbors
    for class_label in oversample_strategy.keys():
        class_count = class_counts[class_label]
        adaptive_n_neighbors = min(5, max(1, class_count - 1))
        # Use adaptive_n_neighbors for this class
    ```""")
    
    print("\n**Solution 4: Filter Classes Before ADASYN**")
    print("""
    Only use ADASYN for classes with sufficient samples:
    
    ```python
    filtered_oversample_strategy = {}
    for class_label, target_count in oversample_strategy.items():
        current_count = class_counts[class_label]
        if current_count >= (n_neighbors + 1):
            filtered_oversample_strategy[class_label] = target_count
        else:
            logger.warning(f"Skipping class {class_label}: "
                          f"only {current_count} samples (need {n_neighbors+1})")
    ```""")
    
    print("\n**Solution 5: Use SMOTE Instead of ADASYN for Small Classes**")
    print("""
    SMOTE is more lenient and can work with fewer samples:
    
    ```python
    # Try SMOTE first for all classes
    # If it works, use it. Otherwise try ADASYN only for larger classes
    ```""")
    
    print("\n" + "=" * 80)
    print("📋 RECOMMENDED FIX")
    print("=" * 80)
    
    print("""
    Modify the ADASYN code to:
    
    1. Check if each class has enough samples (≥ n_neighbors + 1)
    2. Filter out classes that don't meet the requirement
    3. Use SMOTE for classes with insufficient samples
    4. Use ADASYN only for classes that meet requirements
    5. Add better error messages showing which classes failed
    """)
    
    # Example problematic scenario
    print("\n" + "=" * 80)
    print("📊 EXAMPLE PROBLEMATIC SCENARIO")
    print("=" * 80)
    
    print("""
    From the error log:
    - Worms (Label 9): 174 samples → target 15,170 (87x increase)
    - Shellcode (Label 8): 1,511 samples → target 15,170 (10x increase)
    - n_neighbors = 5
    
    **Why ADASYN fails:**
    1. Worms has 174 samples, which is > 6 (n_neighbors+1) ✅
    2. But generating 15,170 samples from 174 (87x) is extreme
    3. ADASYN calculates: samples_to_generate = 15,170 - 174 = 14,996
    4. ADASYN tries to generate ~14,996 samples using only 174 neighbors
    5. The internal ratio calculation may fail because it's too extreme
    
    **Solution:**
    - Use a more gradual approach (e.g., generate in stages)
    - Or use SMOTE which is more lenient
    - Or reduce the target count to something more reasonable
    """)

if __name__ == "__main__":
    diagnose_adasyn_issue()










