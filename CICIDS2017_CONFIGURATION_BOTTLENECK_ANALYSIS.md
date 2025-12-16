# CICIDS2017 Configuration Bottleneck Analysis

## 🔍 **Critical Issues Found**

### **1. Mismatch Between config_loader.py and Optimized Values**

| Parameter                        | config_loader.py | Optimized (best_hyperparameters_cicids.json) | Status                |
| -------------------------------- | ---------------- | -------------------------------------------- | --------------------- |
| `k_shot`                         | **150**          | **41**                                       | ❌ **MAJOR MISMATCH** |
| `zero_day_attack`                | **"DoS"**        | **"PortScan"**                               | ❌ **WRONG ATTACK**   |
| `n_query`                        | 10               | 10                                           | ✅ Correct            |
| `hidden_dim`                     | 512              | 512                                          | ✅ Correct            |
| `meta_epochs`                    | 22               | 22                                           | ✅ Correct            |
| `tcn_kernel_sizes`               | (3, 4, 4)        | (3, 4, 4)                                    | ✅ Correct            |
| `confidence_rejection_threshold` | 0.568            | 0.568                                        | ✅ Correct            |

**🚨 CRITICAL ISSUE #1**: `k_shot: 150` vs optimized `41` (265% larger!)

- **Impact**: Too many support samples per task
- **Effect**: Model may overfit to support set, poor generalization
- **Base Model**: Reduced few-shot learning capability
- **TTT Model**: Less effective adaptation

**🚨 CRITICAL ISSUE #2**: `zero_day_attack: "DoS"` vs optimized `"PortScan"`

- **Impact**: Wrong zero-day attack type
- **Effect**: Model trained for wrong attack, poor zero-day detection
- **Base Model**: Low ZDR (Zero-Day Detection Rate)
- **TTT Model**: Cannot adapt to correct zero-day pattern

---

### **2. TTT Parameters Not in config_loader.py**

The optimized TTT parameters are in `best_hyperparameters_cicids.json` but **NOT** in `config_loader.py`:

| TTT Parameter       | Optimized Value           | config.py Default | Status                       |
| ------------------- | ------------------------- | ----------------- | ---------------------------- |
| `ttt_lr`            | **0.0001518747922672249** | 0.002             | ❌ **TOO HIGH** (13× larger) |
| `ttt_base_steps`    | **194**                   | 70                | ❌ **TOO LOW** (73% smaller) |
| `ttt_l2_reg_weight` | **0.016409286730647923**  | 0.01              | ⚠️ Close but not exact       |
| `use_pseudo_labels` | **false**                 | True              | ❌ **MISMATCH**              |
| `pseudo_weight`     | 3.1167946962329225        | 1.5               | ❌ **MISMATCH**              |
| `entropy_weight`    | **0.8046137691733707**    | 0.8               | ✅ Close                     |
| `ttt_temperature`   | **1.909320402078782**     | 1.31              | ❌ **MISMATCH**              |

**🚨 CRITICAL ISSUE #3**: TTT learning rate mismatch

- **Optimized**: `0.0001519` (very small)
- **config.py**: `0.002` (13× larger!)
- **Impact**: TTT adaptation too aggressive, may overfit
- **Effect**: Poor TTT performance, potential overfitting

**🚨 CRITICAL ISSUE #4**: TTT steps mismatch

- **Optimized**: `194` steps
- **config.py**: `70` steps (64% fewer)
- **Impact**: Insufficient adaptation time
- **Effect**: TTT cannot fully adapt to test distribution

**🚨 CRITICAL ISSUE #5**: Pseudo-labels disabled in optimization

- **Optimized**: `use_pseudo_labels: false`
- **config.py**: `use_pseudo_labels: True`
- **Impact**: Different TTT method used
- **Effect**: TTT behavior completely different from optimized

---

### **3. Comparison with Other Datasets**

| Parameter                        | CICIDS2017                       | KDD  | UNSW | Issue                              |
| -------------------------------- | -------------------------------- | ---- | ---- | ---------------------------------- |
| `k_shot`                         | **41** (opt) / **150** (current) | 152  | 118  | ⚠️ Current too high                |
| `n_query`                        | **10**                           | 16   | 20   | ⚠️ Very low                        |
| `meta_epochs`                    | 22                               | 21   | 18   | ✅ Reasonable                      |
| `hidden_dim`                     | 512                              | 128  | 256  | ✅ Appropriate for dataset size    |
| `confidence_rejection_threshold` | **0.568**                        | 0.90 | 0.70 | ⚠️ **VERY LOW** (rejects too many) |

**🚨 CRITICAL ISSUE #6**: Very low confidence threshold

- **CICIDS2017**: `0.568` (rejects 43% of samples)
- **KDD**: `0.90` (rejects 10% of samples)
- **UNSW**: `0.70` (rejects 30% of samples)
- **Impact**: Too many samples rejected, evaluation on small subset
- **Effect**: Both base and TTT models evaluated on too few samples

**🚨 CRITICAL ISSUE #7**: Very low n_query

- **CICIDS2017**: `10` (smallest)
- **KDD**: `16`
- **UNSW**: `20`
- **Impact**: Less query samples per meta-task
- **Effect**: Less diverse meta-learning, poor generalization

---

### **4. Zero-Day Attack Configuration**

**Current**: `zero_day_attack: "DoS"` in config_loader.py
**Optimized**: `zero_day_attack: "PortScan"` in best_hyperparameters_cicids.json

**CICIDS2017 Attack Categories**:

- DoS: Multiple types (DoS Hulk, DoS GoldenEye, DoS Slowhttptest, DoS slowloris, Heartbleed)
- PortScan: Single type (PortScan)
- WebAttack: Multiple types (Brute Force, SQL Injection, XSS)
- BruteForce: FTP-Patator, SSH-Patator
- Bot, Infiltration, DDoS

**Impact**: Training excludes wrong attack type, zero-day detection fails

---

## 📊 **Root Cause Analysis**

### **Bottlenecks Affecting BOTH Base and TTT Models:**

1. **❌ Wrong Zero-Day Attack** (`"DoS"` vs `"PortScan"`)

   - Base model: Trained without PortScan, cannot detect it
   - TTT model: Adapts to wrong attack pattern

2. **❌ Too High k_shot** (150 vs 41)

   - Base model: Overfits to support set, poor few-shot learning
   - TTT model: Less effective adaptation

3. **❌ Very Low Confidence Threshold** (0.568)

   - Base model: Evaluated on only 57% of test samples
   - TTT model: Evaluated on only 57% of test samples
   - **Both models**: Performance metrics on small subset

4. **❌ Very Low n_query** (10)
   - Base model: Less diverse meta-tasks, poor generalization
   - TTT model: Less query samples for adaptation

### **Bottlenecks Affecting TTT Model Only:**

5. **❌ Wrong TTT Learning Rate** (0.002 vs 0.0001519)

   - TTT adaptation too aggressive
   - May cause overfitting to test distribution

6. **❌ Too Few TTT Steps** (70 vs 194)

   - Insufficient adaptation time
   - Cannot fully adapt to test distribution

7. **❌ Pseudo-Labels Enabled** (True vs False)
   - Different TTT method than optimized
   - May cause instability or overfitting

---

## 🔧 **Recommended Fixes**

### **Priority 1: Critical Fixes (Immediate Impact)**

1. **Fix Zero-Day Attack**:

   ```python
   'zero_day_attack': "PortScan",  # Change from "DoS"
   ```

2. **Fix k_shot**:

   ```python
   'k_shot': 41,  # Change from 150
   ```

3. **Update TTT Parameters in config.py** (or create dataset-specific TTT config):
   ```python
   ttt_lr: float = 0.0001518747922672249  # Change from 0.002
   ttt_base_steps: int = 194  # Change from 70
   use_pseudo_labels: bool = False  # Change from True
   ttt_temperature: float = 1.909320402078782  # Change from 1.31
   ```

### **Priority 2: Important Fixes (Performance Impact)**

4. **Increase Confidence Threshold** (if too many samples rejected):

   ```python
   'confidence_rejection_threshold': 0.70,  # Increase from 0.568 (test)
   ```

5. **Consider Increasing n_query** (if meta-learning is weak):
   ```python
   'n_query': 15,  # Increase from 10 (test)
   ```

---

## 📋 **Action Items**

1. ✅ **Update config_loader.py**:

   - Change `k_shot: 150` → `41`
   - Change `zero_day_attack: "DoS"` → `"PortScan"`

2. ⚠️ **Update config.py TTT parameters** (or make dataset-specific):

   - Update `ttt_lr`, `ttt_base_steps`, `use_pseudo_labels`, `ttt_temperature`

3. ⚠️ **Test confidence threshold**:

   - Current `0.568` may be too low
   - Consider increasing to `0.70` if too many samples rejected

4. ⚠️ **Monitor n_query**:
   - Current `10` is very low
   - Consider increasing to `15` if meta-learning is weak

---

## 🎯 **Expected Impact After Fixes**

### **Base Model**:

- ✅ Correct zero-day attack → Higher ZDR
- ✅ Correct k_shot → Better few-shot learning
- ✅ Higher confidence threshold → More samples evaluated

### **TTT Model**:

- ✅ Correct TTT parameters → Better adaptation
- ✅ More TTT steps → More adaptation time
- ✅ Correct learning rate → Stable adaptation
- ✅ No pseudo-labels → Pure TENT adaptation

---

## ⚠️ **Note on config.py vs config_loader.py**

**Problem**: TTT parameters are in `config.py` (global), not in `config_loader.py` (dataset-specific).

**Solution Options**:

1. **Option A**: Add TTT parameters to `config_loader.py` for each dataset
2. **Option B**: Update `config.py` with CICIDS2017 optimized values (affects all datasets)
3. **Option C**: Create dataset-specific TTT config override mechanism

**Recommendation**: **Option A** - Add TTT parameters to `config_loader.py` for dataset-specific optimization.



