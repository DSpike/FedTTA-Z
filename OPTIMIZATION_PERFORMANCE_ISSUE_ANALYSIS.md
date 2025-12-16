# Optimization Performance Issue Analysis

## 🔴 **Critical Issues Identified**

### **Issue 1: Most Clients Skipped During Training**
**Problem**: During optimization, only **1 out of 9 clients** was actually training. The other 8 clients were skipped due to insufficient samples.

**Evidence from logs**:
```
⚠️ Client client_1: Insufficient samples (176 < 397). Skipping training...
⚠️ Client client_2: Insufficient samples (194 < 397). Skipping training...
...
📊 Active clients: 1/9 (skipped 8 clients with insufficient data)
```

**Root Cause**: 
- `k_shot = 191` requires **191 Normal samples per meta-task**
- `num_meta_tasks = 35` means each client needs: **191 × 35 = 6,685 Normal samples** minimum
- With `dirichlet_alpha = 4.741` and 9 clients, some clients get < 7,000 Normal samples total
- Minimum samples required = `k_shot × n_way + n_query = 191 × 2 + 15 = 397`
- Most clients have < 397 samples available

**Impact**: 
- Federated learning becomes **single-client training** (defeats the purpose)
- Model doesn't benefit from diverse data across clients
- Poor generalization

---

### **Issue 2: Extremely Small Test Set**
**Problem**: Evaluation used only **21 test samples** total (2 zero-day, 19 non-zero-day).

**Evidence from optimization logs**:
```
Evaluating base model on 21 test samples with 2 zero-day samples and 19 non-zero-day samples
```

**Root Cause**:
- Test set creation might be failing to create proper stratified subset
- Or the CICIDS dataset has limited zero-day samples available for test set

**Impact**:
- **Unreliable metrics** - 21 samples is way too small for statistical significance
- **Perfect zero-day detection (ZDR=1.0)** is meaningless with only 2 samples
- Metrics like F1, AUC-PR are highly unstable with such small test sets

---

### **Issue 3: TTT Adaptation Stopped Prematurely**
**Problem**: TTT was configured for 335 steps but stopped after only **6 steps**.

**Evidence from logs**:
```
TTT: 335 steps with batch size 8
⚠️ Diversity below threshold (0.6901 < 0.80) - stopping adaptation to prevent collapse (after 6 steps)
```

**Root Cause**:
- Diversity threshold (0.80) is too strict
- Model predictions became too concentrated after just 6 steps
- This might be due to the small test set (21 samples) causing rapid collapse

**Impact**:
- TTT adaptation doesn't have enough time to improve the model
- The optimized `ttt_base_steps = 335` parameter is not actually being used

---

### **Issue 4: Hyperparameter Incompatibility**
**Problem**: The optimized hyperparameters are **incompatible** with the federated learning setup.

**Specific Issues**:

1. **k_shot = 191 is too high**
   - Requires ~7,000 Normal samples per client
   - With 9 clients and dirichlet distribution, many clients can't meet this requirement
   - Result: Most clients skipped → Poor federated learning

2. **num_clients = 9 with dirichlet_alpha = 4.741**
   - 9 clients with moderate alpha creates high heterogeneity
   - Some clients get very few samples
   - Combined with high k_shot, most clients are excluded

3. **num_meta_tasks = 35 with k_shot = 191**
   - Each task needs 191 Normal samples
   - Total requirement: 6,685 Normal samples per client
   - Very few clients can meet this

---

### **Issue 5: Optimization Metric Bias**
**Problem**: The optimization achieved **perfect zero-day detection (ZDR=1.0)** but at the cost of **very poor overall performance**.

**Optimization Results**:
- Base Accuracy: **52.38%** (very poor)
- Base F1: **54.55%** (poor)
- Base AUC-PR: **39.30%** (poor)
- Base ZDR: **100%** (perfect, but only 2 samples!)
- Base Non-Zero-Day F1: **44.44%** (poor)

**Root Cause**:
- Balanced metric gives 40% weight to base F1, 30% to TTT ZDR, 30% to TTT F1
- With only 2 zero-day samples, achieving 100% ZDR is trivial
- The optimization overfits to zero-day at the expense of overall performance

---

## 🎯 **Recommended Solutions**

### **Solution 1: Reduce k_shot Requirement**
```python
# In config.py
k_shot: int = 50  # Reduced from 191 (more reasonable)
num_meta_tasks: int = 50  # Increase tasks to compensate for lower k_shot
```

**Rationale**: Lower k_shot allows more clients to participate in federated learning.

---

### **Solution 2: Adjust Client and Dirichlet Settings**
```python
num_clients: int = 5  # Reduce from 9 (fewer clients = more samples per client)
dirichlet_alpha: float = 2.0  # Increase heterogeneity slightly (was 4.741)
```

**Rationale**: Fewer clients with moderate heterogeneity ensures each client has enough samples.

---

### **Solution 3: Fix Test Set Size**
**Check why test set only has 21 samples**:
- Verify CICIDS dataset has enough zero-day samples
- Check test set creation logic in `_stratified_test_subset`
- Ensure target composition (60/30/10) is achievable

---

### **Solution 4: Adjust TTT Diversity Threshold**
```python
# In coordinators/simple_fedavg_coordinator.py
diversity_threshold: float = 0.50  # Reduce from 0.80 (less strict)
```

**Rationale**: Allow TTT to run for more steps before stopping.

---

### **Solution 5: Re-run Optimization with Constraints**
Add constraints to optimization to prevent incompatible hyperparameters:

```python
# In optimize_hyperparameters_cicids.py
def suggest_hyperparameters(self, trial):
    # ...
    k_shot = trial.suggest_int('k_shot', 30, 100)  # Cap at 100
    num_meta_tasks = trial.suggest_int('num_meta_tasks', 20, 50)
    
    # Constraint: Ensure clients can meet requirements
    # Estimate: With 9 clients and alpha=4.741, average client gets ~11% of samples
    # For CICIDS with ~100k samples, average client gets ~11k samples
    # Each task needs: k_shot × 2 (normal + attack) + n_query ≈ k_shot × 3
    # Maximum tasks per client: min_samples / (k_shot × 3)
    max_feasible_tasks = min(50, int(5000 / (k_shot * 3)))  # Conservative estimate
    if num_meta_tasks > max_feasible_tasks:
        num_meta_tasks = max_feasible_tasks
```

---

## 📊 **Immediate Actions**

1. **✅ Reduce k_shot to reasonable value** (50-80)
2. **✅ Reduce num_clients to 5**
3. **✅ Investigate why test set is so small** (21 samples)
4. **✅ Re-run optimization with constraints**
5. **✅ Lower TTT diversity threshold** to allow more adaptation steps

---

## 🔍 **Verification Steps**

After fixing, verify:
1. **At least 5/5 clients participate** in each round
2. **Test set has > 100 samples** (ideally 500+)
3. **TTT runs for > 50 steps** (not just 6)
4. **Base model accuracy > 70%** (not 52%)
5. **Base F1 > 0.65** (not 0.54)









