# 🗑️ Federated Learning Removal Checklist

## ✅ **What Will Be Removed:**

### **1. Files to DELETE:**
- [ ] `coordinators/simple_fedavg_coordinator.py` (entire file)
- [ ] Check if `coordinators/decentralized_coordinator.py` needs deletion

### **2. In `main.py`:**

#### **Imports:**
- [ ] Remove `from coordinators.simple_fedavg_coordinator import SimpleFedAVGCoordinator`

#### **Classes to REMOVE:**
- [ ] `SecureBlockchainFederatedIncentiveSystem` class
- [ ] `BlockchainFederatedIncentiveSystem` class

#### **Methods to REMOVE:**
- [ ] `setup_federated_learning()` method
- [ ] `run_meta_training()` method
- [ ] `_aggregate_meta_histories()` method
- [ ] All client-related validation methods

#### **Code Sections to REMOVE:**
- [ ] Federated coordinator initialization (lines ~514-522)
- [ ] Federated round loops in `main()` function
- [ ] All conditional federated/centralized logic

### **3. In `config.py`:**

#### **Parameters to REMOVE:**
- [ ] `num_clients: int`
- [ ] `num_rounds: int` (or repurpose)
- [ ] `dirichlet_alpha: float`
- [ ] `use_fedprox: bool`
- [ ] `fedprox_mu: float`

### **4. Keep These:**
- ✅ `CentralizedCoordinator` (this is what we want!)
- ✅ All TTT logic
- ✅ All meta-learning logic
- ✅ All evaluation logic
- ✅ All preprocessing logic

---

## 🚀 **Ready to Start Removal?**

I'll proceed systematically. Should I continue?









