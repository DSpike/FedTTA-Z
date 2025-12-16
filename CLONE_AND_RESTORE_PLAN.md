# Plan: Clone GitHub Version & Re-add Pseudo-Labeling Features

## 🎯 Objective
1. Clone the original GitHub repository (FedTTA-Z)
2. Run it to establish baseline performance
3. Systematically re-add all pseudo-labeling features we developed
4. Verify improvements match our current implementation

---

## 📋 Step-by-Step Plan

### **PHASE 1: Clone & Setup GitHub Version**

#### Step 1.1: Clone Repository
```bash
# Navigate to parent directory
cd C:\Users\Dspike\Documents\PhD\TNN\exp1\

# Clone GitHub repo to a new directory
git clone https://github.com/DSpike/FedTTA-Z.git FedTTA-Z_original

# Navigate to cloned repo
cd FedTTA-Z_original
```

#### Step 1.2: Identify Repository Structure
- Check if it uses `blockchain_federated_learning_project/` structure or flat structure
- Identify entry point (`main.py` location)
- Check for `config.py` or configuration files
- Verify dependencies (`requirements.txt`)

#### Step 1.3: Setup Environment
```bash
# Create virtual environment
python -m venv venv_original
venv_original\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt  # or requirements_blockchain_fl.txt
```

#### Step 1.4: Run Baseline
```bash
python main.py  # or python src/main.py (depending on structure)
```
- **Document baseline metrics**: Accuracy, F1, ZDR, AUC-PR
- **Save baseline results**: Create `baseline_results.json`
- **Note configuration**: Document all hyperparameters used

---

### **PHASE 2: Identify Key Differences**

#### Step 2.1: Compare File Structures
**Current (Modified) Structure:**
```
Tgnn/
├── coordinators/
│   └── simple_fedavg_coordinator.py  # Contains TENTPseudoLabels class
├── models/
│   └── transductive_fewshot_model.py
├── config.py                          # Contains pseudo-labeling config
├── main.py
└── visualization/
    └── performance_visualization.py
```

**GitHub (Original) Structure:**
```
FedTTA-Z_original/
├── [Check actual structure]
├── coordinators/  (or blockchain_federated_learning_project/src/coordinators/)
│   └── simple_fedavg_coordinator.py  # Original TENT only
└── [Other files]
```

#### Step 2.2: Key Files to Compare
1. **`coordinators/simple_fedavg_coordinator.py`**
   - Original: Only `_perform_advanced_ttt_adaptation` (pure TENT)
   - Modified: Added `TENTPseudoLabels` class + `_perform_tent_pseudo_labels_adaptation`

2. **`config.py`**
   - Original: Basic TTT config
   - Modified: Added pseudo-labeling configs (lines 90-102)

3. **`main.py`**
   - Original: Calls `adapt_to_test_data(method='tent')`
   - Modified: Calls `adapt_to_test_data(method='tent_pseudo')`

4. **`models/transductive_fewshot_model.py`**
   - Check: Data leakage fix (support/query overlap prevention)

---

### **PHASE 3: Re-add Pseudo-Labeling Features**

#### Step 3.1: Add Configuration Options
**File: `config.py`**
```python
# === TENT + PSEUDO-LABELS CONFIGURATION ===
use_pseudo_labels: bool = True
pseudo_threshold: float = 0.92
pseudo_min_threshold: float = 0.85
ttt_max_pseudo_ratio: float = 0.6
pseudo_weight: float = 0.8
entropy_weight: float = 0.4
use_teacher: bool = True
ema_decay: float = 0.999
use_adaptive_threshold: bool = True
threshold_adaptation_mode: str = 'combined'
```

#### Step 3.2: Add TENTPseudoLabels Class
**File: `coordinators/simple_fedavg_coordinator.py`**
- Add entire `TENTPseudoLabels` class (lines 1091-1698 from current version)
- Key methods:
  - `__init__()`: Initialize with config
  - `_configure_model_for_tent()`: Setup trainable parameters
  - `_generate_pseudo_labels()`: Multi-strategy pseudo-labeling
  - `_adaptive_threshold()`: Curriculum learning
  - `_update_teacher()`: EMA teacher updates
  - `adapt()`: Main adaptation loop

#### Step 3.3: Add Wrapper Method
**File: `coordinators/simple_fedavg_coordinator.py`**
- Add `_perform_tent_pseudo_labels_adaptation()` method (lines 929-1000)
- This method creates `TENTPseudoLabels` adapter and calls `adapt()`

#### Step 3.4: Update Method Selection
**File: `coordinators/simple_fedavg_coordinator.py`**
- Modify `adapt_to_test_data()` method (lines 1020-1087)
- Add `method='tent_pseudo'` option
- Update method selection logic

#### Step 3.5: Update Main Entry Point
**File: `main.py`**
- Change `method='tent'` → `method='tent_pseudo'`
- Or add config-based selection: `method='tent_pseudo' if config.use_pseudo_labels else 'tent'`

#### Step 3.6: Fix Data Leakage (if needed)
**File: `models/transductive_fewshot_model.py`**
- Verify `create_meta_tasks()` excludes support samples from query set
- Add exclusion mask logic if missing

---

### **PHASE 4: Verification & Testing**

#### Step 4.1: Run Modified Version
```bash
python main.py
```

#### Step 4.2: Compare Results
- **Baseline (GitHub)**: Document metrics
- **Modified (with pseudo-labels)**: Document metrics
- **Expected improvement**: +8-12% vs +2-5% for pure TENT

#### Step 4.3: Verify Features
- ✅ Pseudo-label generation working
- ✅ Adaptive threshold decreasing
- ✅ EMA teacher updating
- ✅ Loss curves showing improvement
- ✅ Statistics tracking (pseudo_label_ratio, etc.)

---

## 🔍 Key Differences Summary

### **What GitHub Version Has (Pure TENT)**
1. ✅ Entropy minimization
2. ✅ Diversity regularization
3. ✅ BatchNorm adaptation
4. ❌ No pseudo-labeling
5. ❌ No EMA teacher
6. ❌ No adaptive threshold curriculum
7. ❌ No class-balanced thresholding

### **What We're Adding (Pseudo-Labeling Extension)**
1. ✅ Multi-strategy pseudo-label generation
2. ✅ Temperature sharpening
3. ✅ Class-balanced thresholds
4. ✅ Entropy-based uncertainty filtering
5. ✅ EMA teacher model (temporal consistency)
6. ✅ Adaptive threshold curriculum
7. ✅ Pseudo-label ratio capping
8. ✅ Enhanced statistics tracking

---

## 📝 Implementation Checklist

### Configuration (`config.py`)
- [ ] Add `use_pseudo_labels: bool = True`
- [ ] Add `pseudo_threshold: float = 0.92`
- [ ] Add `pseudo_min_threshold: float = 0.85`
- [ ] Add `ttt_max_pseudo_ratio: float = 0.6`
- [ ] Add `pseudo_weight: float = 0.8`
- [ ] Add `entropy_weight: float = 0.4`
- [ ] Add `use_teacher: bool = True`
- [ ] Add `ema_decay: float = 0.999`
- [ ] Add `use_adaptive_threshold: bool = True`

### Coordinator (`coordinators/simple_fedavg_coordinator.py`)
- [ ] Add `TENTPseudoLabels` class (entire class)
- [ ] Add `_perform_tent_pseudo_labels_adaptation()` method
- [ ] Update `adapt_to_test_data()` to support `method='tent_pseudo'`
- [ ] Import required modules (`copy`, `torch.nn.functional as F`, etc.)

### Main Entry (`main.py`)
- [ ] Update TTT method call to use `'tent_pseudo'`
- [ ] Or add conditional: `method='tent_pseudo' if config.use_pseudo_labels else 'tent'`

### Model (`models/transductive_fewshot_model.py`)
- [ ] Verify data leakage fix (support/query exclusion)
- [ ] Add if missing: exclusion mask in `create_meta_tasks()`

---

## 🚀 Execution Order

1. **Clone & Setup** (30 min)
   - Clone repo
   - Setup environment
   - Run baseline

2. **Compare & Document** (1 hour)
   - Compare file structures
   - Document differences
   - Identify exact locations for changes

3. **Implement Changes** (2-3 hours)
   - Add config options
   - Add TENTPseudoLabels class
   - Update method calls
   - Fix any compatibility issues

4. **Test & Verify** (1 hour)
   - Run modified version
   - Compare results
   - Verify all features working

---

## 📊 Expected Outcomes

### Baseline (GitHub Pure TENT)
- Accuracy improvement: +2-5%
- ZDR improvement: Moderate
- Loss curve: Basic entropy minimization

### With Pseudo-Labeling (Our Extension)
- Accuracy improvement: +8-12%
- ZDR improvement: Significant (+0.08-0.16)
- Loss curve: Shows pseudo-label + entropy + diversity components
- Pseudo-label ratio: ~60% of samples per step
- Adaptive threshold: Decreases from 0.92 → 0.85

---

## ⚠️ Potential Issues & Solutions

### Issue 1: Import Errors
- **Solution**: Ensure all imports match GitHub version structure

### Issue 2: Config Structure Differences
- **Solution**: Adapt config additions to match GitHub's config format

### Issue 3: Method Signature Mismatches
- **Solution**: Check GitHub version's method signatures and adapt

### Issue 4: Missing Dependencies
- **Solution**: Compare `requirements.txt` and install missing packages

---

## 📁 Files to Backup Before Changes

Before modifying GitHub version, backup:
1. `coordinators/simple_fedavg_coordinator.py` (original)
2. `config.py` (original)
3. `main.py` (original)
4. `models/transductive_fewshot_model.py` (original)

Create backup directory:
```bash
mkdir backups
cp coordinators/simple_fedavg_coordinator.py backups/
cp config.py backups/
cp main.py backups/
```

---

## ✅ Success Criteria

1. ✅ GitHub version runs successfully (baseline established)
2. ✅ All pseudo-labeling features added without breaking existing code
3. ✅ Modified version runs successfully
4. ✅ Performance improvement matches expectations (+8-12%)
5. ✅ All statistics tracking working (pseudo_label_ratio, etc.)
6. ✅ Loss curves show proper components (pseudo + entropy + diversity)

---

## 📝 Notes

- Keep current modified version (`Tgnn/`) as reference
- Document any structural differences between GitHub and current version
- Create git branch for modifications: `git checkout -b add-pseudo-labeling`
- Commit changes incrementally for easy rollback

