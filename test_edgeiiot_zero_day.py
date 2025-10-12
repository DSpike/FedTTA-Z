#!/usr/bin/env python3
"""
Test script for Edge-IIoTset zero-day attack simulation
Demonstrates how to test different attack types as zero-day attacks
"""

import logging
from main_edgeiiot import test_zero_day_attack, test_multiple_zero_day_attacks, EdgeIIoTSystemConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_zero_day_attacks():
    """Demonstrate zero-day attack testing with Edge-IIoTset"""
    
    print("🎯 Edge-IIoTset Zero-Day Attack Testing Demo")
    print("=" * 60)
    
    # Available attack types with sample counts
    attack_types = {
        "DDoS_UDP": "121,568 samples - High frequency DDoS attack",
        "DDoS_ICMP": "116,436 samples - ICMP-based DDoS attack", 
        "SQL_injection": "51,203 samples - Database injection attack",
        "Password": "50,153 samples - Password-based attack",
        "Vulnerability_scanner": "50,110 samples - Network scanning attack",
        "DDoS_TCP": "50,062 samples - TCP-based DDoS attack",
        "DDoS_HTTP": "49,911 samples - HTTP-based DDoS attack",
        "Uploading": "37,634 samples - File upload attack",
        "Backdoor": "24,862 samples - Backdoor installation attack",
        "Port_Scanning": "22,564 samples - Port scanning attack",
        "XSS": "15,915 samples - Cross-site scripting attack",
        "Ransomware": "10,925 samples - Ransomware attack",
        "MITM": "1,214 samples - Man-in-the-middle attack",
        "Fingerprinting": "1,001 samples - System fingerprinting attack"
    }
    
    print("\n📊 Available Attack Types for Zero-Day Simulation:")
    for attack, description in attack_types.items():
        print(f"  • {attack}: {description}")
    
    print("\n🎯 Zero-Day Attack Simulation Process:")
    print("  1. Select an attack type (e.g., 'DDoS_UDP')")
    print("  2. Remove that attack from training data")
    print("  3. Use only that attack + normal samples for testing")
    print("  4. Train model on remaining attacks")
    print("  5. Test model's ability to detect the unseen attack")
    
    print("\n🚀 Testing Examples:")
    
    # Example 1: Test DDoS_UDP as zero-day
    print("\n1. Testing DDoS_UDP as zero-day attack...")
    try:
        success = test_zero_day_attack("DDoS_UDP")
        if success:
            print("   ✅ DDoS_UDP zero-day test completed successfully!")
        else:
            print("   ❌ DDoS_UDP zero-day test failed!")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Example 2: Test SQL_injection as zero-day
    print("\n2. Testing SQL_injection as zero-day attack...")
    try:
        success = test_zero_day_attack("SQL_injection")
        if success:
            print("   ✅ SQL_injection zero-day test completed successfully!")
        else:
            print("   ❌ SQL_injection zero-day test failed!")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n💡 Usage Examples:")
    print("  # Test single attack type")
    print("  python test_edgeiiot_zero_day.py")
    print("  ")
    print("  # Test multiple attack types")
    print("  # Uncomment test_multiple_zero_day_attacks() in main_edgeiiot.py")
    print("  ")
    print("  # Test specific attack in code")
    print("  from main_edgeiiot import test_zero_day_attack")
    print("  test_zero_day_attack('Backdoor')")

def show_configuration_options():
    """Show configuration options for zero-day testing"""
    
    print("\n⚙️ Configuration Options:")
    print("=" * 40)
    
    config = EdgeIIoTSystemConfig()
    
    print(f"Dataset Path: {config.data_path}")
    print(f"Default Zero-Day Attack: {config.zero_day_attack}")
    print(f"Available Attacks: {len(config.available_attacks)} types")
    print(f"Input Dimensions: {config.input_dim}")
    print(f"Hidden Dimensions: {config.hidden_dim}")
    print(f"Embedding Dimensions: {config.embedding_dim}")
    print(f"Number of Clients: {config.num_clients}")
    print(f"Number of Rounds: {config.num_rounds}")
    print(f"TTT Steps: {config.ttt_steps}")
    print(f"Max Samples per Client: {config.max_samples_per_client}")
    print(f"Use Data Sampling: {config.use_data_sampling}")
    
    print("\n🔧 Customization Examples:")
    print("  # Test with different attack type")
    print("  config = EdgeIIoTSystemConfig(zero_day_attack='Ransomware')")
    print("  ")
    print("  # Test with more samples")
    print("  config = EdgeIIoTSystemConfig(max_samples_per_client=100000)")
    print("  ")
    print("  # Test with more TTT steps")
    print("  config = EdgeIIoTSystemConfig(ttt_steps=500)")

if __name__ == "__main__":
    demonstrate_zero_day_attacks()
    show_configuration_options()
