#!/usr/bin/env python3
"""
Test Identical Initialization of Miners
Verifies that all miners start with exactly the same model parameters
"""

import torch
import torch.nn as nn
import logging
import numpy as np
from decentralized_fl_system import DecentralizedFederatedLearningSystem
from models.transductive_fewshot_model import TransductiveFewShotModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_identical_initialization():
    """Test that miners start with identical initial parameters"""
    
    print("🧪 Testing Identical Initialization of Miners")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create the original model
    print("1️⃣ Creating original model...")
    original_model = TransductiveFewShotModel(
        input_dim=30,
        hidden_dim=64,
        embedding_dim=32,
        num_classes=2
    )
    
    # Get original model parameters
    original_params = {name: param.clone() for name, param in original_model.named_parameters()}
    print(f"   Original model created with {len(original_params)} parameter tensors")
    
    # Initialize decentralized system (this should create identical copies)
    print("\n2️⃣ Initializing decentralized system...")
    fl_system = DecentralizedFederatedLearningSystem(original_model, num_clients=3)
    
    # Get miner models
    miner_1_model = fl_system.miners["miner_1"].model
    miner_2_model = fl_system.miners["miner_2"].model
    
    print("   ✅ Decentralized system initialized")
    
    # Test 1: Check if miners have different model objects
    print("\n3️⃣ Testing Model Object Independence:")
    are_different_objects = miner_1_model is not miner_2_model
    print(f"   Miner 1 and Miner 2 are different objects: {are_different_objects}")
    
    # Test 2: Check if miners have identical parameters
    print("\n4️⃣ Testing Parameter Identicality:")
    identical_params = True
    param_differences = []
    
    for (name1, param1), (name2, param2) in zip(
        miner_1_model.named_parameters(),
        miner_2_model.named_parameters()
    ):
        if not torch.equal(param1, param2):
            identical_params = False
            diff = torch.norm(param1 - param2).item()
            param_differences.append((name1, diff))
            print(f"   ❌ Parameter mismatch: {name1}")
            print(f"      Miner 1: {param1[0, 0].item():.8f}")
            print(f"      Miner 2: {param2[0, 0].item():.8f}")
            print(f"      Difference: {diff:.8f}")
    
    if identical_params:
        print("   ✅ All parameters are identical between miners")
    else:
        print(f"   ❌ Found {len(param_differences)} parameter differences")
    
    # Test 3: Compare with original model
    print("\n5️⃣ Testing Against Original Model:")
    identical_with_original_1 = True
    identical_with_original_2 = True
    
    for (name_orig, param_orig), (name1, param1), (name2, param2) in zip(
        original_model.named_parameters(),
        miner_1_model.named_parameters(),
        miner_2_model.named_parameters()
    ):
        if not torch.equal(param_orig, param1):
            identical_with_original_1 = False
            print(f"   ❌ Miner 1 differs from original: {name_orig}")
            break
        
        if not torch.equal(param_orig, param2):
            identical_with_original_2 = False
            print(f"   ❌ Miner 2 differs from original: {name_orig}")
            break
    
    print(f"   Miner 1 identical to original: {identical_with_original_1}")
    print(f"   Miner 2 identical to original: {identical_with_original_2}")
    
    # Test 4: Detailed parameter comparison
    print("\n6️⃣ Detailed Parameter Analysis:")
    
    # Get parameter statistics
    def get_param_stats(model, name):
        params = [param for name, param in model.named_parameters()]
        all_params = torch.cat([p.flatten() for p in params])
        return {
            'mean': all_params.mean().item(),
            'std': all_params.std().item(),
            'min': all_params.min().item(),
            'max': all_params.max().item(),
            'shape': [p.shape for p in params]
        }
    
    original_stats = get_param_stats(original_model, "Original")
    miner_1_stats = get_param_stats(miner_1_model, "Miner 1")
    miner_2_stats = get_param_stats(miner_2_model, "Miner 2")
    
    print(f"   Original Model Stats:")
    print(f"      Mean: {original_stats['mean']:.8f}")
    print(f"      Std:  {original_stats['std']:.8f}")
    print(f"      Min:  {original_stats['min']:.8f}")
    print(f"      Max:  {original_stats['max']:.8f}")
    
    print(f"   Miner 1 Stats:")
    print(f"      Mean: {miner_1_stats['mean']:.8f}")
    print(f"      Std:  {miner_1_stats['std']:.8f}")
    print(f"      Min:  {miner_1_stats['min']:.8f}")
    print(f"      Max:  {miner_1_stats['max']:.8f}")
    
    print(f"   Miner 2 Stats:")
    print(f"      Mean: {miner_2_stats['mean']:.8f}")
    print(f"      Std:  {miner_2_stats['std']:.8f}")
    print(f"      Min:  {miner_2_stats['min']:.8f}")
    print(f"      Max:  {miner_2_stats['max']:.8f}")
    
    # Test 5: Memory address verification
    print("\n7️⃣ Memory Address Verification:")
    original_addresses = [id(param) for param in original_model.parameters()]
    miner_1_addresses = [id(param) for param in miner_1_model.parameters()]
    miner_2_addresses = [id(param) for param in miner_2_model.parameters()]
    
    print(f"   Original model parameter addresses: {original_addresses[:3]}...")
    print(f"   Miner 1 parameter addresses: {miner_1_addresses[:3]}...")
    print(f"   Miner 2 parameter addresses: {miner_2_addresses[:3]}...")
    
    # Check if addresses are different (they should be)
    addresses_different = (
        set(original_addresses).isdisjoint(set(miner_1_addresses)) and
        set(original_addresses).isdisjoint(set(miner_2_addresses)) and
        set(miner_1_addresses).isdisjoint(set(miner_2_addresses))
    )
    print(f"   All parameter addresses are different: {addresses_different}")
    
    # Test 6: Test independence by modifying one miner
    print("\n8️⃣ Testing Independence by Modification:")
    
    # Get initial parameter values
    initial_param_1 = list(miner_1_model.parameters())[0][0, 0].item()
    initial_param_2 = list(miner_2_model.parameters())[0][0, 0].item()
    
    print(f"   Initial Miner 1 param[0,0]: {initial_param_1:.8f}")
    print(f"   Initial Miner 2 param[0,0]: {initial_param_2:.8f}")
    
    # Modify miner 1
    with torch.no_grad():
        for param in miner_1_model.parameters():
            param.add_(0.1)
    
    # Check if miner 2 is affected
    modified_param_1 = list(miner_1_model.parameters())[0][0, 0].item()
    modified_param_2 = list(miner_2_model.parameters())[0][0, 0].item()
    
    print(f"   After modification Miner 1 param[0,0]: {modified_param_1:.8f}")
    print(f"   After modification Miner 2 param[0,0]: {modified_param_2:.8f}")
    
    miner_2_unchanged = abs(modified_param_2 - initial_param_2) < 1e-8
    print(f"   Miner 2 unchanged by Miner 1 modification: {miner_2_unchanged}")
    
    # Test 7: Test synchronization
    print("\n9️⃣ Testing Synchronization:")
    
    # Now miners should be different
    miners_different_before_sync = not torch.equal(
        list(miner_1_model.parameters())[0],
        list(miner_2_model.parameters())[0]
    )
    print(f"   Miners different before sync: {miners_different_before_sync}")
    
    # Synchronize
    fl_system.synchronize_models()
    
    # Check if they're identical again
    miners_identical_after_sync = True
    for (name1, param1), (name2, param2) in zip(
        miner_1_model.named_parameters(),
        miner_2_model.named_parameters()
    ):
        if not torch.equal(param1, param2):
            miners_identical_after_sync = False
            break
    
    print(f"   Miners identical after sync: {miners_identical_after_sync}")
    
    # Final summary
    print("\n📊 TEST RESULTS SUMMARY:")
    print("=" * 40)
    
    results = {
        "different_objects": are_different_objects,
        "identical_parameters": identical_params,
        "identical_with_original_1": identical_with_original_1,
        "identical_with_original_2": identical_with_original_2,
        "different_memory_addresses": addresses_different,
        "independent_modification": miner_2_unchanged,
        "synchronization_works": miners_identical_after_sync
    }
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Miners truly start with identical initial parameters")
        print("✅ Miners are independent objects")
        print("✅ Synchronization works correctly")
    else:
        print("\n⚠️ Some tests failed!")
        print("❌ There may be issues with the initialization")
    
    return results

def test_multiple_initializations():
    """Test that multiple initializations produce identical results"""
    
    print("\n🔄 Testing Multiple Initializations")
    print("=" * 50)
    
    results = []
    
    for i in range(3):
        print(f"\nInitialization {i+1}:")
        
        # Set same seed each time
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create model
        model = TransductiveFewShotModel(30, 64, 32, 2)
        
        # Get first parameter value
        first_param = list(model.parameters())[0][0, 0].item()
        print(f"   First parameter value: {first_param:.8f}")
        
        results.append(first_param)
    
    # Check if all initializations are identical
    all_identical = all(abs(r - results[0]) < 1e-8 for r in results)
    print(f"\nAll initializations identical: {all_identical}")
    
    if all_identical:
        print("✅ Reproducible initialization confirmed")
    else:
        print("❌ Initialization is not reproducible")
    
    return all_identical

def main():
    """Run all initialization tests"""
    try:
        print("🧪 COMPREHENSIVE INITIALIZATION TESTING")
        print("=" * 60)
        
        # Test 1: Identical initialization
        results = test_identical_initialization()
        
        # Test 2: Multiple initializations
        reproducible = test_multiple_initializations()
        
        # Final verdict
        print("\n🏆 FINAL VERDICT:")
        print("=" * 30)
        
        if all(results.values()) and reproducible:
            print("✅ PERFECT! Miners truly start with identical parameters")
            print("✅ System is working correctly")
            print("✅ No initialization issues detected")
        else:
            print("❌ ISSUES DETECTED!")
            print("❌ Miners may not be starting with identical parameters")
            print("❌ System may have initialization problems")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()







