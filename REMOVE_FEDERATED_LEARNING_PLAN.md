# 🗑️ Remove Federated Learning Features - Implementation Plan

## 🎯 **Goal**

Remove all federated learning features from the codebase, keeping only centralized learning.

---

## 📋 **Components to Remove**

### **1. Coordinator Files**

#### **Files to DELETE:**
- ✅ `coordinators/simple_fedavg_coordinator.py` - Full federated coordinator implementation
- ⚠️ `coordinators/decentralized_coordinator.py` - Decentralized FL (if exists, check if needed)
- ✅ **KEEP**: `coordinators/centralized_coordinator.py` - This is what we want!

---

### **2. Main System File (`main.py`)**

#### **Classes/Methods to REMOVE:**
1. `SecureBlockchainFederatedIncentiveSystem` class (lines ~333-393)
2. `BlockchainFederatedIncentiveSystem` class (lines ~394-510)
3. `setup_federated_learning()` method (lines ~1332-1369)
4. `run_meta_training()` method (lines ~1373-1439) - Pre-federated training (redundant)
5. `_aggregate_meta_histories()` method (lines ~1441-1477)
6. All client-related methods in `SecureBlockchainFederatedIncentiveSystem`
7. All federated round logic in `main()` function

#### **Code Sections to REMOVE:**
- Federated learning coordinator initialization (lines ~514-523)
- Federated learning setup calls (lines calling `setup_federated_learning()`)
- Federated round loops in `main()` function
- Client update aggregation logic
- Dirichlet distribution data splitting

#### **Code Sections to KEEP/MODIFY:**
- ✅ Centralized coordinator initialization (already exists, lines ~524-530)
- ✅ `initialize_system()` method (modify to only use centralized)
- ✅ Data preprocessing (keep)
- ✅ TTT adaptation (keep)
- ✅ Evaluation logic (keep)

---

### **3. Configuration File (`config.py`)**

#### **Parameters to REMOVE:**
- `num_clients: int` - Number of federated clients
- `num_rounds: int` - Federated learning rounds (or repurpose for centralized)
- `dirichlet_alpha: float` - For non-IID data distribution
- `use_fedprox: bool` - FedProx regularization
- `fedprox_mu: float` - FedProx parameter

#### **Parameters to KEEP/MODIFY:**
- ✅ `use_federated_learning: bool = False` - Keep but always set to False
- ✅ All TTT parameters (keep)
- ✅ All meta-learning parameters (keep)
- ✅ All model architecture parameters (keep)

---

### **4. Preprocessing Files**

#### **Files to CHECK/MODIFY:**
- ⚠️ `preprocessing/blockchain_federated_unsw_preprocessor.py` - Check if has FL-specific logic
- ⚠️ `blockchain_federated_cicids_preprocessor.py` - Check if has FL-specific logic

**Action**: Remove Dirichlet distribution logic if present, keep preprocessing core functionality

---

### **5. Model Files**

#### **Files to CHECK:**
- `models/transductive_fewshot_model.py` - Should be FL-agnostic (keep)
- Check for any client-specific or federated-specific code

**Action**: Verify no FL dependencies, keep as-is if clean

---

## 🔧 **Implementation Steps**

### **Step 1: Simplify `main.py`**

1. Remove all federated learning classes
2. Remove `setup_federated_learning()` method
3. Remove `run_meta_training()` method
4. Remove `_aggregate_meta_histories()` method
5. Simplify `initialize_system()` to only create centralized coordinator
6. Remove federated round loops from `main()`
7. Update `main()` to only call centralized training

---

### **Step 2: Clean Up `config.py`**

1. Remove `num_clients` parameter
2. Remove `num_rounds` parameter (or repurpose as centralized training epochs)
3. Remove `dirichlet_alpha` parameter
4. Remove `use_fedprox` and `fedprox_mu` parameters
5. Set `use_federated_learning: bool = False` permanently
6. Add comment: "Federated learning removed - only centralized learning supported"

---

### **Step 3: Delete Coordinator Files**

1. Delete `coordinators/simple_fedavg_coordinator.py`
2. Check and delete `coordinators/decentralized_coordinator.py` if not needed
3. Keep `coordinators/centralized_coordinator.py`
4. Update `coordinators/__init__.py` to only export centralized coordinator

---

### **Step 4: Clean Up Imports**

1. Remove `from coordinators.simple_fedavg_coordinator import SimpleFedAVGCoordinator`
2. Keep `from coordinators.centralized_coordinator import CentralizedCoordinator`
3. Remove any federated-specific imports

---

### **Step 5: Update Documentation**

1. Update README to reflect centralized-only learning
2. Remove federated learning documentation
3. Update comments in code

---

## ✅ **Verification Checklist**

After removal, verify:
- [ ] System runs with centralized learning only
- [ ] No references to `SimpleFedAVGCoordinator`
- [ ] No references to federated clients
- [ ] No references to federated rounds
- [ ] No references to model aggregation
- [ ] No references to Dirichlet distribution
- [ ] All tests pass (if any)
- [ ] Documentation updated

---

## 🎯 **Expected Outcome**

**Before:**
- Mixed federated/centralized codebase
- Complex coordinator selection logic
- Client management overhead

**After:**
- Clean centralized-only codebase
- Single coordinator (`CentralizedCoordinator`)
- Simplified training flow
- Easier to maintain and understand

---

## ⚠️ **Important Notes**

1. **Keep TTT Logic**: Test-Time Training is not federated learning - keep it!
2. **Keep Meta-Learning**: Meta-learning is the training method - keep it!
3. **Keep Preprocessing**: Data preprocessing is separate - keep it!
4. **Keep Evaluation**: Performance evaluation is separate - keep it!

**What we're removing:**
- Client splitting and management
- Federated rounds and aggregation
- Dirichlet data distribution
- FedProx regularization
- Model weight aggregation

---

## 🚀 **Ready to Start?**

This plan systematically removes all federated learning features while preserving:
- ✅ Centralized training logic
- ✅ Meta-learning functionality
- ✅ TTT adaptation
- ✅ Evaluation and visualization









