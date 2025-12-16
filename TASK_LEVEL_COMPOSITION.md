# Task-Level Normal and Attack Sample Composition

## 📊 **Per Meta-Task Composition**

### **Complete Breakdown Per Meta-Task:**

With `k_shot = 169` and `n_query = 18`:

---

## 🎯 **Task-Level Summary**

### **One Complete Meta-Task:**

| Component | Normal | Attack | Total |
|-----------|--------|--------|-------|
| **Support Set** | 169 | 169 | 338 |
| **Query Set** | ~12 | ~24 | 36 |
| **TOTAL PER TASK** | **~181** | **~193** | **374** |

---

## 📋 **Task-Level Percentages**

### **Normal vs Attack Distribution:**

| Component | Normal | Normal % | Attack | Attack % | Total |
|-----------|--------|----------|--------|----------|-------|
| **Support Set** | 169 | 50.0% | 169 | 50.0% | 338 |
| **Query Set** | ~12 | 33.3% | ~24 | 66.7% | 36 |
| **TASK TOTAL** | **~181** | **48.4%** | **~193** | **51.6%** | **374** |

---

## 📊 **Visual Task Composition**

### **Complete Meta-Task (374 samples):**

```
Support Set (338 samples):
  Normal:  ████████████████████ 169 (50.0%)
  Attack:  ████████████████████ 169 (50.0%)

Query Set (36 samples):
  Normal:  ████  ~12 (33.3%)
  Attack:  ████████████  ~24 (66.7%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK-LEVEL TOTAL (374 samples):
  Normal:  ████████████████████▌ ~181 (48.4%)
  Attack:  █████████████████████ ~193 (51.6%)
```

---

## 🔍 **Detailed Task-Level Breakdown**

### **Normal Samples (Total ~181 per task):**

| Source | Count | Percentage of Task |
|--------|-------|-------------------|
| **From Support Set** | 169 | 45.2% |
| **From Query Set** | ~12 | 3.2% |
| **Total Normal** | **~181** | **48.4%** |

### **Attack Samples (Total ~193 per task):**

| Source | Count | Percentage of Task | Distribution |
|--------|-------|-------------------|--------------|
| **From Support Set** | 169 | 45.2% | Uniform across 8 types:<br>• 7 types: 21 samples<br>• 1 type: 22 samples |
| **From Query Set** | ~24 | 6.4% | Natural distribution |
| **Total Attack** | **~193** | **51.6%** | Mixed (all types) |

---

## 📈 **Task-Level Statistics**

### **Per Meta-Task:**

- **Total Samples**: **374**
- **Normal Samples**: **~181** (48.4%)
- **Attack Samples**: **~193** (51.6%)
- **Ratio**: **~48:52** (Normal:Attack)
- **Balance**: **Slightly imbalanced** (more attack samples)

---

## 🎯 **Key Insights**

### **Task-Level Composition:**

1. **Support Set Contribution**: 
   - **338 samples** (90.4% of task)
   - Balanced: 50% Normal, 50% Attack

2. **Query Set Contribution**: 
   - **36 samples** (9.6% of task)
   - Imbalanced: 33% Normal, 67% Attack

3. **Overall Task Balance**:
   - **~48% Normal, ~52% Attack**
   - Slightly more attack samples due to query set imbalance

---

## 💡 **Simple Answer**

### **Per Meta-Task (Task-Level):**

- **Normal Samples**: **~181** (48.4%)
- **Attack Samples**: **~193** (51.6%)
- **Total**: **374 samples**
- **Balance**: Nearly balanced, slightly more attack samples

---

## 📊 **Across Multiple Tasks**

### **20 Meta-Tasks Per Client:**

| Component | Total Samples | Normal | Normal % | Attack | Attack % |
|-----------|--------------|--------|----------|--------|----------|
| **Support Sets** | 6,760 | 3,380 | 50.0% | 3,380 | 50.0% |
| **Query Sets** | 720 | ~240 | 33.3% | ~480 | 66.7% |
| **TOTAL** | **7,480** | **~3,620** | **48.4%** | **~3,860** | **51.6%** |

---

## 🔧 **Why This Composition?**

1. **Support Set (50/50)**: Balanced learning from equal examples
2. **Query Set (33/67)**: Realistic testing with natural imbalance
3. **Task Total (48/52)**: Slight attack bias reflects real-world cybersecurity data










