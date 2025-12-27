# Zero-Day Attack Changed to Backdoor

## ✅ **Change Applied**

**Previous**: `DoS` (label 4)  
**Current**: `Backdoor` (label 3)  
**Attack Grouping**: `False` (fine-grained attack types)  
**Dataset**: `UNSW-NB15`

## 📝 **Configuration Updated**

**File**: `config_loader.py`  
**Location**: Line 55 (UNSW section)

```python
'zero_day_attack': "Backdoor",  # Switched from DoS to Backdoor
'use_category_grouping': False,
```

## 🎯 **Expected Behavior**

- **Training/Validation**: Will exclude all "Backdoor" attack samples
- **Test Set**: Will include "Backdoor" samples as zero-day attacks
- **Evaluation**: ZDR and other metrics will be calculated on "Backdoor" samples only

## 📊 **UNSW Attack Types**

| Attack Name | Label | Status |
|------------|-------|--------|
| Normal | 0 | Included in training |
| Fuzzers | 1 | Included in training |
| Analysis | 2 | Included in training |
| **Backdoor** | **3** | **ZERO-DAY (excluded from training)** |
| DoS | 4 | Included in training |
| Exploits | 5 | Included in training |
| Generic | 6 | Included in training |
| Reconnaissance | 7 | Included in training |
| Shellcode | 8 | Included in training |
| Worms | 9 | Included in training |

## 🔍 **Verification**

Run the system and check:
1. Zero-day samples are correctly identified as "Backdoor"
2. Training data excludes "Backdoor" samples
3. Test data includes "Backdoor" samples
4. ZDR is calculated on "Backdoor" samples



