# 🎯 How to Use Centralized Learning Mode

## 📊 **Quick Start**

To switch from **federated learning** to **centralized learning**, simply change **one line** in `config.py`:

```python
# In config.py
use_federated_learning: bool = False  # Change from True to False
```

That's it! The system will automatically:
- ✅ Use the full dataset directly (no client splitting)
- ✅ Train on all data in one place
- ✅ Use the same model architecture and training code
- ✅ Support all features (TTT, evaluation, visualization)

---

## 🔧 **How It Works**

### **Federated Learning Mode** (`use_federated_learning = True`)
- Data is split across multiple clients using Dirichlet distribution
- Each client trains locally on its subset
- Models are aggregated using FedAvg/FedProx
- Simulates distributed learning scenario

### **Centralized Learning Mode** (`use_federated_learning = False`)
- **All training data** is used directly (no splitting)
- Model trains on the **full dataset** at once
- No aggregation needed (single training process)
- Direct comparison baseline

---

## 📝 **Configuration**

Both modes use the same configuration parameters:

```python
# config.py
use_federated_learning: bool = False  # Switch between modes

# These parameters are used in both modes:
num_rounds: int = 15              # Number of training rounds
meta_epochs: int = 18             # Meta-learning epochs per round
k_shot: int = 118                 # Support set size
n_query: int = 20                 # Query set size
num_meta_tasks: int = 34          # Number of meta-tasks per round

# These are ONLY used in federated mode (ignored in centralized):
num_clients: int = 5              # Number of clients (not used in centralized)
dirichlet_alpha: float = 1.27     # Data distribution (not used in centralized)
```

---

## 🎯 **Key Differences**

| Aspect | Federated Learning | Centralized Learning |
|--------|-------------------|---------------------|
| **Data Distribution** | Split across clients (non-IID) | Full dataset in one place |
| **Training** | Each client trains locally | Single training on full data |
| **Aggregation** | FedAvg/FedProx aggregation | Not needed |
| **Privacy** | Simulates data privacy | No privacy (all data visible) |
| **Use Case** | Realistic distributed scenario | Performance baseline/comparison |

---

## ✅ **Benefits of Centralized Mode**

1. **Performance Baseline**: See maximum achievable performance
2. **Faster Experimentation**: No aggregation overhead
3. **Fair Comparison**: Compare federated vs centralized directly
4. **Debugging**: Easier to debug with full data access
5. **Same Codebase**: All features work the same way

---

## 🔍 **What Stays the Same**

- ✅ Model architecture (TransductiveFewShotModel)
- ✅ Meta-learning training process
- ✅ TTT adaptation
- ✅ Evaluation metrics and visualization
- ✅ All hyperparameters and configurations
- ✅ Zero-day attack detection logic

---

## 📊 **Example Usage**

### **Step 1: Switch to Centralized Mode**

```python
# config.py
use_federated_learning: bool = False  # Enable centralized learning
```

### **Step 2: Run the System**

```bash
python main.py
```

The system will automatically:
1. Load full training data (no splitting)
2. Train on all data directly
3. Perform TTT adaptation
4. Generate evaluation plots

### **Step 3: Compare Results**

You can now compare:
- **Federated results**: Performance with data distribution
- **Centralized results**: Maximum achievable performance
- **Gap analysis**: See the cost of federated learning

---

## ⚠️ **Important Notes**

1. **No Changes to Federated Code**: All federated learning code remains untouched
2. **Easy Switching**: Just change the config flag
3. **Same Features**: TTT, evaluation, visualization all work
4. **Fair Comparison**: Uses identical model architecture and hyperparameters

---

## 🎯 **Use Cases**

### **Research Experiments**
- Compare federated vs centralized performance
- Understand the cost of distributed learning
- Establish performance baselines

### **Quick Testing**
- Faster iteration (no aggregation overhead)
- Easier debugging with full data access
- Validate model architecture quickly

### **Paper Comparison**
- Show federated learning performance vs centralized
- Demonstrate privacy-performance trade-off
- Highlight the effectiveness of your federated approach

---

## 📝 **Troubleshooting**

### **Issue: System still uses federated learning**
- ✅ Check `config.py`: `use_federated_learning = False`
- ✅ Restart the Python process

### **Issue: Different results than expected**
- ✅ Check that hyperparameters are the same
- ✅ Verify same dataset is being used
- ✅ Ensure same random seed

---

## 🎉 **Summary**

**Centralized learning mode is now available!** Simply set `use_federated_learning = False` in `config.py` and run your experiments. All existing code remains unchanged - you get a new mode with zero modifications to the federated learning implementation.









