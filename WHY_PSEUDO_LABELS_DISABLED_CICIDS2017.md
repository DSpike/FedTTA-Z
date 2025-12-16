# Why Pseudo-Labels Were Disabled for CICIDS2017

## 🔍 **Important Clarification**

**I did NOT disable pseudo-labels** - this came from **Optuna optimization results**!

The optimization for CICIDS2017 found that `use_pseudo_labels: false` (pure TENT) performed better than `use_pseudo_labels: true` (TENT + pseudo-labels).

---

## 📊 **What the Optimization Found**

From `best_hyperparameters_cicids.json`:
```json
{
  "use_pseudo_labels": false,  // ← Optuna found this better
  "pseudo_weight": 3.1167946962329225,  // Still optimized (for if enabled)
  "entropy_weight": 0.8046137691733707,
  ...
}
```

**This means**: Optuna tested both `True` and `False` and found `False` gave better performance for CICIDS2017.

---

## 🤔 **Why Might Pure TENT Be Better for CICIDS2017?**

### **1. Dataset Characteristics**
- **CICIDS2017**: 78 features, 7 attack categories
- **Complex distribution**: May have overlapping attack patterns
- **Pure TENT** (entropy minimization only) might be more stable

### **2. Pseudo-Label Risks**
When pseudo-labels are enabled:
- **Risk**: If initial predictions are wrong, pseudo-labels reinforce errors
- **Risk**: Model may overfit to incorrect pseudo-labels
- **Risk**: Especially problematic for zero-day detection (unseen patterns)

### **3. Pure TENT Benefits**
- **Stable**: Only minimizes entropy, doesn't commit to specific labels
- **Flexible**: Adapts to test distribution without forcing predictions
- **Better for zero-day**: Doesn't assume initial predictions are correct

---

## 📋 **Comparison: Pure TENT vs TENT + Pseudo-Labels**

| Aspect | Pure TENT (`use_pseudo_labels: false`) | TENT + Pseudo-Labels (`use_pseudo_labels: true`) |
|--------|----------------------------------------|---------------------------------------------------|
| **Method** | Entropy minimization only | Entropy + pseudo-label cross-entropy |
| **Stability** | ✅ More stable | ⚠️ Can reinforce errors |
| **Zero-Day Detection** | ✅ Better for unseen patterns | ⚠️ May overfit to wrong labels |
| **Speed** | ✅ Faster (no pseudo-label generation) | ⚠️ Slightly slower |
| **When Good** | Complex distributions, zero-day detection | Clear class boundaries, high initial accuracy |
| **CICIDS2017** | ✅ **Optimized choice** | ⚠️ Not optimal for this dataset |

---

## 🔧 **Should You Enable Pseudo-Labels?**

### **Option 1: Keep Disabled (Current - Optimized)**
```python
'use_pseudo_labels': False,  # Pure TENT (optimized for CICIDS2017)
```

**Pros**:
- ✅ Optimized by Optuna for CICIDS2017
- ✅ More stable adaptation
- ✅ Better for zero-day detection

**Cons**:
- ⚠️ May miss benefits of pseudo-label supervision

### **Option 2: Enable and Test**
```python
'use_pseudo_labels': True,  # Test TENT + pseudo-labels
'pseudo_threshold': 0.8626973748208299,  # From optimization
'pseudo_weight': 3.1167946962329225,  # From optimization
```

**Pros**:
- ✅ May improve non-zero-day performance
- ✅ Provides supervision signal
- ✅ Can test if it works better

**Cons**:
- ⚠️ May overfit to incorrect predictions
- ⚠️ May hurt zero-day detection

---

## 💡 **Recommendation**

### **For CICIDS2017**:
1. **Keep disabled** (as optimized) - This is what Optuna found best
2. **If performance is poor**, try enabling with optimized parameters:
   ```python
   'use_pseudo_labels': True,
   'pseudo_threshold': 0.8626973748208299,
   'pseudo_weight': 3.1167946962329225,
   ```

### **Why Optuna Might Have Found This**:
- **CICIDS2017 has complex attack patterns** (7 categories, 78 features)
- **Zero-day detection** (PortScan) may benefit from flexible adaptation
- **Pure TENT** doesn't commit to potentially wrong labels
- **Entropy minimization** alone may be sufficient for this dataset

---

## 🎯 **Key Takeaway**

**This was NOT my decision** - it came from Optuna optimization results. The optimizer tested both options and found that **pure TENT (no pseudo-labels) performed better for CICIDS2017**.

If you want to test with pseudo-labels enabled, you can change it in `config_loader.py`:
```python
'use_pseudo_labels': True,  # Test this
```

But the optimized configuration suggests **pure TENT is better for CICIDS2017**.




