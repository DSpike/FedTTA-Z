# Training Data Composition: Normal and Attack Samples

## 🔍 **What Attack Types Are Used for Training?**

**Short Answer:** The model is trained on **Normal + ONE randomly selected Attack class** per meta-task, where the attack class is randomly chosen from **available non-zero-day attacks**.

---

## 📊 **Attack Types Available in Training**

### **Current Configuration:**

- **Zero-Day Attack**: `Backdoor` (label 3) - **EXCLUDED from training**
- **Available for Training**: All other attack types

### **UNSW-NB15 Attack Types:**

| Attack Type        | Label | Training Status                 | Training Samples (Approx.) |
| ------------------ | ----- | ------------------------------- | -------------------------- |
| **Normal**         | 0     | ✅ **Always Used**              | 56,000+                    |
| **Fuzzers**        | 1     | ✅ Available (Random Selection) | 18,184                     |
| **Analysis**       | 2     | ✅ Available (Random Selection) | 2,000                      |
| **Backdoor**       | 3     | ❌ **EXCLUDED** (Zero-day)      | 0                          |
| **DoS**            | 4     | ✅ Available (Random Selection) | 12,264                     |
| **Exploits**       | 5     | ✅ Available (Random Selection) | 33,393                     |
| **Generic**        | 6     | ✅ Available (Random Selection) | 40,000                     |
| **Reconnaissance** | 7     | ✅ Available (Random Selection) | 10,491                     |
| **Shellcode**      | 8     | ✅ Available (Random Selection) | 1,133                      |
| **Worms**          | 9     | ✅ Available (Random Selection) | 130                        |

---

## 🎯 **Support Set Composition Per Meta-Task**

### **With `enforce_equal_support_composition = True`:**

For each meta-task, the system:

1. **Always selects Normal (label 0)**
2. **Randomly selects ONE attack class** from:
   - ✅ Fuzzers (1)
   - ✅ Analysis (2)
   - ✅ DoS (4)
   - ✅ Exploits (5)
   - ✅ Generic (6)
   - ✅ Reconnaissance (7)
   - ✅ Shellcode (8)
   - ✅ Worms (9)
   - ❌ **NOT Backdoor (3)** - excluded as zero-day

### **Example Meta-Tasks:**

**Task 1:**

- Support Set: 150 Normal + 150 **Fuzzers** = 300 samples

**Task 2:**

- Support Set: 150 Normal + 150 **Generic** = 300 samples

**Task 3:**

- Support Set: 150 Normal + 150 **Exploits** = 300 samples

**Task 4:**

- Support Set: 150 Normal + 150 **DoS** = 300 samples

**...and so on** (randomly varies across tasks)

---

## 🔄 **Why Random Selection?**

The attack class is **randomly selected** for each meta-task to ensure:

1. **Diverse Training**: Model sees different attack types across tasks
2. **Better Generalization**: Learns attack patterns from multiple classes
3. **Balanced Exposure**: All non-zero-day attacks get used in training
4. **Robust Prototypes**: Prototypes learned from various attack types

---

## 📊 **Full Training Composition**

### **Across All Meta-Tasks (e.g., 20 tasks):**

The model is trained on:

- ✅ **Normal samples** (always present in every task)
- ✅ **Fuzzers** (may appear in some tasks)
- ✅ **Generic** (may appear in some tasks)
- ✅ **Exploits** (may appear in some tasks)
- ✅ **DoS** (may appear in some tasks)
- ✅ **Analysis** (may appear in some tasks)
- ✅ **Reconnaissance** (may appear in some tasks)
- ✅ **Shellcode** (may appear in some tasks)
- ✅ **Worms** (may appear in some tasks)
- ❌ **Backdoor** (never - excluded as zero-day)

**Result**: Model learns from **Normal + 8 different attack types** (all except Backdoor).

---

## 🎯 **Is It Always Fuzzers?**

**No!** The attack class is **randomly selected** per task:

```python
# Line 1502-1504: Random selection
if len(attack_labels) > 0:
    attack_label_idx = torch.randint(0, len(attack_labels), (1,))
    selected_attack_label = attack_labels[attack_label_idx]
```

**Possible selections** (with equal probability):

- Fuzzers (18.6% chance, assuming uniform distribution)
- Generic (40.8% chance - most common)
- Exploits (34.2% chance - second most common)
- DoS (12.5% chance)
- Reconnaissance (10.7% chance)
- Analysis (2.0% chance)
- Shellcode (1.2% chance)
- Worms (0.1% chance - least common)

**Note**: Selection is **uniform random**, not weighted by sample count. So each attack type has equal probability of being selected (1/8 = 12.5% each).

---

## 📈 **What This Means**

### **Per Meta-Task:**

- Support Set = 50% Normal + 50% **One Attack Type** (randomly chosen)
- Example: `150 Normal + 150 Fuzzers` OR `150 Normal + 150 Generic`, etc.

### **Across All Training:**

- Model learns from **Normal + Multiple Attack Types**
- All 8 non-zero-day attack types may appear in different tasks
- Model generalizes across attack patterns
- Zero-day (Backdoor) is completely unseen during training

---

## 🔍 **Code Verification**

```python
# Line 1495-1499: Get available attack labels
attack_labels = available_labels[(available_labels != 0) &
                                 (available_labels != zero_day_attack_label)]
# Result: [1, 2, 4, 5, 6, 7, 8, 9]  (excludes Normal=0 and Backdoor=3)

# Line 1502-1504: Random selection
attack_label_idx = torch.randint(0, len(attack_labels), (1,))
selected_attack_label = attack_labels[attack_label_idx]
# Result: Randomly selects one from [1, 2, 4, 5, 6, 7, 8, 9]
```

---

## 📊 **Summary**

**Training Data Composition:**

| Component        | Always?   | Details                                                                        |
| ---------------- | --------- | ------------------------------------------------------------------------------ |
| **Normal (0)**   | ✅ Yes    | Always in every meta-task                                                      |
| **Attack Class** | ✅ Yes    | One randomly selected per task                                                 |
| **Attack Types** | 🔀 Random | Fuzzers, Generic, Exploits, DoS, Analysis, Reconnaissance, Shellcode, or Worms |
| **Backdoor (3)** | ❌ No     | Excluded as zero-day attack                                                    |

**Answer:** No, it's not always Fuzzers. The attack class is **randomly selected** from 8 available attack types. The model trains on **Normal + various attack types** (excluding Backdoor/zero-day).









