# CICIDS2017 Configuration Revert - Reason

## ⚠️ **Issue Found**

The Optuna optimization results in `best_hyperparameters_cicids.json` show:

```json
{
  "best_value": -Infinity, // ❌ This indicates optimization FAILED
  "best_trial_number": 0
}
```

**Problem**: `-Infinity` as the best value means the optimization study failed or didn't complete successfully. The "optimized" values are likely from an incomplete or failed run.

---

## 🔄 **Reverted Configuration**

### **TCN Settings**:

- `tcn_kernel_sizes`: `(3, 5, 7)` ✅ (reverted from `(3, 4, 4)`)
- `sequence_stride`: `15` ✅ (reverted from `12`)

### **Other Settings**:

- `hidden_dim`: `256` ✅ (reverted from `512`)
- `meta_epochs`: `20` ✅ (reverted from `22`)
- `k_shot`: `100` ✅ (reverted from `41`)
- `n_query`: `15` ✅ (reverted from `10`)
- `learning_rate`: `0.001` ✅ (reverted from `0.0015751320499779737`)

### **Kept**:

- `zero_day_attack`: `"PortScan"` (more common zero-day attack for CICIDS2017)

---

## 💡 **Why the Original Values?**

The original configuration `(3, 5, 7)` was based on:

1. **Hierarchical multi-scale pattern**: Captures fine → medium → coarse temporal patterns
2. **Dataset characteristics**: CICIDS2017 has 78 features (larger than KDD/UNSW)
3. **Common practice**: Multi-scale kernel sizes are standard in temporal CNNs

---

## 🎯 **Recommendation**

### **Option 1: Keep Current (Reverted) Values**

- Use the hierarchical `(3, 5, 7)` pattern
- These are reasonable defaults for CICIDS2017

### **Option 2: Run New Optuna Optimization**

If you want truly optimized values:

```bash
python optimize_hyperparameters_cicids.py --n_trials 50
```

This will run a proper optimization and save valid results.

### **Option 3: Use Values from Similar Dataset**

- **KDD**: `(2, 3, 3)` - smaller kernels
- **UNSW**: `(3, 3, 6)` - medium kernels
- **CICIDS2023**: `(3, 4, 5)` - medium-large kernels

You could try `(3, 4, 5)` as a middle ground.

---

## ✅ **Current Status**

Configuration has been **reverted to original values** that were working better. The Optuna results were from a failed optimization run and should not be used.



