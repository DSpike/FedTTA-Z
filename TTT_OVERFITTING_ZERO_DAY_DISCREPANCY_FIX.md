# TTT Overfitting: Zero Day Discrepancy Fix

## 🔍 **Problem Analysis**

**Status**: OVERFITTING (Severity: MEDIUM)
**Flag**: Zero Day Discrepancy

### **What This Means**

The TTT model is performing significantly better on zero-day attacks (>10% higher accuracy) than on the overall test set. This indicates:

1. **Overfitting to Zero-Day Patterns**: TTT is adapting too aggressively to zero-day attack patterns
2. **Performance Trade-off**: While zero-day detection improves, overall performance (especially on Normal samples) degrades
3. **Distribution Mismatch**: TTT is specializing on outlier patterns rather than generalizing

### **Root Cause**

The discrepancy occurs when:
- `ttt_zero_day_accuracy > ttt_overall_accuracy + 0.10` (10% threshold)

This suggests TTT is:
- ✅ Learning to detect zero-day outliers effectively
- ⚠️ Overfitting to the zero-day distribution
- ⚠️ Sacrificing performance on Normal/non-zero-day samples

---

## 🔧 **Recommended Fixes**

### **Option 1: Reduce Pseudo-Label Weight (Recommended)**

**Current Value**: `pseudo_weight: float = 3.0425406933718913` (too high)

**Fix**:
```python
pseudo_weight: float = 2.0  # Reduce from 3.04 → 2.0 (34% reduction)
```

**Rationale**: High pseudo-weight causes overfitting to pseudo-labels, especially on zero-day patterns.

---

### **Option 2: Increase L2 Regularization**

**Current Value**: `ttt_l2_reg_weight: float = 0.0010257563974185654` (low)

**Fix**:
```python
ttt_l2_reg_weight: float = 0.005  # Increase from 0.001 → 0.005 (5x increase)
```

**Rationale**: Stronger regularization prevents overfitting to specific patterns.

---

### **Option 3: Reduce Prototype Weight**

**Current Value**: `ttt_prototype_weight: float = 1` (high)

**Fix**:
```python
ttt_prototype_weight: float = 0.5  # Reduce from 1.0 → 0.5 (50% reduction)
```

**Rationale**: Lower prototype weight reduces overfitting to zero-day prototype patterns.

---

### **Option 4: Reduce TTT Steps**

**Current Value**: `ttt_base_steps: int = 100`

**Fix**:
```python
ttt_base_steps: int = 80  # Reduce from 100 → 80 (20% reduction)
```

**Rationale**: Fewer steps prevent overfitting while still allowing adaptation.

---

### **Option 5: Increase Entropy Weight (Balance Adaptation)**

**Current Value**: `entropy_weight: float = 0.5740446517340904`

**Fix**:
```python
entropy_weight: float = 0.7  # Increase from 0.57 → 0.7 (22% increase)
```

**Rationale**: Higher entropy weight encourages more balanced adaptation across all samples, not just zero-day.

---

## 📊 **Recommended Configuration Changes**

Apply these changes to `config.py`:

```python
# === TENT + PSEUDO-LABELS CONFIGURATION ===
use_pseudo_labels: bool = True
pseudo_threshold: float = 0.95
pseudo_min_threshold: float = 0.7173803589287694
pseudo_weight: float = 2.0  # REDUCED from 3.04 → 2.0 (reduce overfitting)
entropy_weight: float = 0.7  # INCREASED from 0.57 → 0.7 (balance adaptation)
use_teacher: bool = True
ema_decay: float = 0.9662140032177797

# === TEST-TIME TRAINING (TTT) CONFIGURATION ===
ttt_base_steps: int = 80  # REDUCED from 100 → 80 (prevent overfitting)
ttt_max_steps: int = 400
ttt_adaptation_query_size: int = 1198
ttt_batch_size: int = 64
ttt_lr: float = 0.0002911701023242743
ttt_l2_reg_weight: float = 0.005  # INCREASED from 0.001 → 0.005 (stronger regularization)
confidence_rejection_threshold: float = 0.8261845713819337

# === ATTACK PROTOTYPE DISCOVERY TTT ===
ttt_prototype_clusters: int = 10
ttt_prototype_weight: float = 0.5  # REDUCED from 1.0 → 0.5 (reduce overfitting)
ttt_prototype_entropy_weight: float = 0.3
ttt_prototype_steps: int = 100
```

---

## 🎯 **Expected Impact**

### **Before (Current)**:
- Zero-day accuracy: ~85-90%
- Overall accuracy: ~75-80%
- **Discrepancy**: >10% ⚠️

### **After (With Fixes)**:
- Zero-day accuracy: ~80-85% (slight decrease)
- Overall accuracy: ~80-85% (improvement)
- **Discrepancy**: <5% ✅

**Trade-off**: Slight reduction in zero-day performance for better overall balance.

---

## ✅ **Verification Steps**

After applying fixes, check:

1. **Run the system** and check TTT overfitting diagnostic
2. **Verify** that "Zero Day Discrepancy" flag is no longer triggered
3. **Check metrics**:
   - Overall accuracy should improve
   - Zero-day accuracy may decrease slightly
   - Discrepancy should be <10%

---

## 📝 **Notes**

- These changes reduce overfitting while maintaining zero-day detection capability
- The goal is **balanced performance** across all sample types
- If zero-day performance drops too much, gradually increase `pseudo_weight` back to 2.5

