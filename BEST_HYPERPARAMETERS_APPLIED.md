# Best Hyperparameters Applied to config.py

## ✅ **Status: Applied**

The best hyperparameters from **Multi-Objective Optimization (Trial 6)** have been successfully applied to `config.py`.

---

## 📊 **Optimization Results Summary**

**Best Trial**: 6  
**Multi-Objective Score**: 0.9371 (93.71%)  
**Optimization Target**: Balanced ZDR (40%) + F1-Score (30%) + Accuracy (30%)

### **Performance Achieved:**
- ✅ **Zero-Day Detection Rate**: 100.00% (Perfect!)
- ✅ **Overall Accuracy**: 88.00% (+30.00% improvement)
- ✅ **Overall F1-Score**: 91.04% (+35.73% improvement)
- ✅ **Non-Zero-Day F1**: 87.23% (+55.66% improvement)

---

## 🔄 **Hyperparameters Updated**

### **Federated Learning:**
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `num_clients` | 8 | **10** | +2 clients |
| `num_rounds` | 5 | **13** | +8 rounds |
| `learning_rate` (meta) | 0.007649 | **0.002541** | Lower learning rate |
| `dirichlet_alpha` | 1.026 | **4.035** | Higher heterogeneity |
| `fedprox_mu` | 0.001022 | **0.004425** | Stronger proximal term |

### **Model Architecture:**
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `hidden_dim` | 512 | **512** | Unchanged |
| `embedding_dim` | 512 | **256** | Reduced by 50% |

### **TCN Configuration:**
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `sequence_length` | 31 | **47** | +16 (longer sequences) |
| `sequence_stride` | 13 | **13** | Unchanged |
| `tcn_kernel_sizes` | (5, 3, 5) | **(3, 2, 4)** | Hierarchical pattern |
| `meta_epochs` | 3 | **3** | Unchanged |

### **Meta-Learning:**
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `k_shot` | 169 | **129** | -40 samples |
| `n_query` | 18 | **18** | Unchanged |
| `num_meta_tasks` | 20 | **35** | +15 tasks |
| `enforce_equal_support_composition` | True | **False** | Changed |

### **TTT Configuration:**
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `ttt_base_steps` | 332 | **293** | -39 steps |
| `ttt_batch_size` | 16 | **16** | Unchanged |
| `ttt_adaptation_query_size` | 1990 | **1037** | -953 samples |
| `ttt_lr` | 0.000604 | **0.000111** | Lower learning rate |
| `ttt_temperature` | 1.512 | **1.541** | Slightly higher |

### **TENT + Pseudo-Labels:**
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `use_pseudo_labels` | True | **True** | Unchanged (ENABLED) |
| `pseudo_threshold` | 0.890 | **0.950** | Higher threshold |
| `pseudo_min_threshold` | 0.786 | **0.732** | Lower minimum |
| `pseudo_weight` | 2.062 | **1.754** | Lower weight |
| `entropy_weight` | 0.771 | **1.022** | Higher weight |
| `use_teacher` | True | **True** | Unchanged (ENABLED) |
| `ema_decay` | 0.991 | **0.953** | Lower decay |
| `pseudo_label_temperature` | 0.469 | **0.566** | Higher temperature |

### **Advanced TTT:**
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `use_focal_loss` | True | **False** | DISABLED |
| `focal_gamma` | 2.116 | **2.964** | Higher gamma |
| `focal_alpha` | 0.298 | **0.253** | Lower alpha |

---

## 🎯 **Key Differences from Single-Objective Optimization**

### **1. Pseudo-Labels Enabled:**
- **Single-Objective**: `use_pseudo_labels: false` (pure TENT)
- **Multi-Objective**: `use_pseudo_labels: true` (TENT + Pseudo-Labels) ⭐

**Impact**: Better balance between zero-day detection and overall performance

### **2. Higher Dirichlet Alpha:**
- **Single-Objective**: `dirichlet_alpha: 1.272` (moderate heterogeneity)
- **Multi-Objective**: `dirichlet_alpha: 4.035` (lower heterogeneity)

**Impact**: More balanced data distribution across clients

### **3. More Federated Rounds:**
- **Single-Objective**: `num_rounds: 12`
- **Multi-Objective**: `num_rounds: 13`

**Impact**: Better convergence and model quality

### **4. Different TCN Kernel Sizes:**
- **Single-Objective**: `(6, 3, 3)`
- **Multi-Objective**: `(3, 2, 4)`

**Impact**: Different temporal pattern capture

### **5. Lower TTT Learning Rate:**
- **Single-Objective**: `ttt_lr: 0.000508`
- **Multi-Objective**: `ttt_lr: 0.000111`

**Impact**: More stable adaptation, less aggressive updates

---

## 📈 **Expected Performance**

With these hyperparameters, you should expect:

### **Base Model:**
- Accuracy: ~58%
- F1-Score: ~55%
- ZDR: ~85%

### **TTT Model:**
- Accuracy: **~88%** (+30%)
- F1-Score: **~91%** (+36%)
- ZDR: **100%** (+15%)
- Non-Zero-Day F1: **~87%** (+56%)

---

## ✅ **Next Steps**

1. **Run the system** with the new configuration:
   ```bash
   python main.py
   ```

2. **Compare results** with previous runs to verify the improvements

3. **Monitor**:
   - Zero-day detection rate (should be ~100%)
   - Overall accuracy and F1-score improvements
   - Non-zero-day performance improvements

---

## 📝 **Notes**

- All hyperparameters are from **Trial 6** of the multi-objective optimization
- The configuration balances **zero-day detection** (40% weight) with **overall performance** (60% weight: 30% F1 + 30% Accuracy)
- **Pseudo-labels are enabled** which helps maintain overall performance while achieving 100% ZDR
- The system is now configured for **balanced performance** rather than **zero-day-only optimization**

---

**Applied Date**: 2025-11-28  
**Source**: Multi-Objective Optimization Trial 6  
**Optimization Score**: 0.9371 (93.71%)










