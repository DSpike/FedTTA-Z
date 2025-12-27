# Critical Evaluation Methodology Difference

**Date**: 2025-12-21

---

## VLSTM Paper Approach: Anomaly Detection (One-Class)

### Training:
- **Train on**: Normal samples ONLY
- **Learn**: What "normal" behavior looks like
- **Task**: Detect anything that deviates from normal

### Testing:
- **Test on**: Mix of normal + anomaly samples
- **Goal**: Flag anomalies (anything abnormal)
- **Evaluation**: Can they detect samples that deviate from learned normal?

### Key Characteristic:
- **No attack type information during training**
- Model learns normal distribution, flags outliers
- Easier task: Just detect "different from normal"

---

## Your Approach: Zero-Day Attack Classification (Multi-Class)

### Training:
- **Train on**: Normal + 8 attack types
- **Learn**: What each attack type looks like
- **Task**: Classify into specific categories

### Testing (LOAO):
- **Test on**: 1 unseen attack type (zero-day)
- **Goal**: Correctly classify the new attack type
- **Evaluation**: Can model generalize to unseen attack categories?

### Key Characteristic:
- **Must distinguish between attack types**
- Model learns attack signatures, must generalize
- Harder task: Classify specific unseen attack patterns

---

## Why These Are NOT Directly Comparable

### Different Problem Formulations:

```
VLSTM (Anomaly Detection):
├─ Training: [Normal, Normal, Normal, ...]
├─ Testing: [Normal, Anomaly1, Anomaly2, ...]
└─ Task: Is this normal or anomaly? (Binary)

Your Approach (Zero-Day Classification):
├─ Training: [Normal, Attack1, Attack2, ..., Attack8]
├─ Testing: [Normal, Attack9 (unseen)]
└─ Task: What type is this? (Multi-class)
```

### Difficulty Comparison:

| Aspect | VLSTM (Anomaly Detection) | Your Approach (Zero-Day) |
|--------|---------------------------|--------------------------|
| Training Data | Normal only | Normal + 8 attack types |
| Test Challenge | Detect deviation | Classify unseen attack type |
| Decision | Binary (normal/anomaly) | Multi-class generalization |
| Relative Difficulty | **Easier** | **Harder** |

---

## Why Your Results Look "Worse"

### Your Metrics:
- Recall: 93.99% (detect unseen attack types)
- FAR: 42.53% (misclassify normal as attack)
- F1: 68.69%

### What's happening:
1. Your model learned 8 attack patterns
2. When it sees a 9th unseen attack, it must:
   - Recognize it's an attack (not normal)
   - Generalize from seen attacks to unseen ones
3. This is MUCH harder than just flagging "different from normal"

### VLSTM's Advantage:
1. Only needs to learn normal distribution
2. Anything unusual → flag as anomaly
3. Doesn't need to understand attack types
4. Natural advantage on recall (97.8%)

---

## Are Your Results Actually Competitive?

### NO - if comparing apples to apples
Your task is fundamentally harder, but the metrics don't reflect this.

### Possible Fair Comparison:

**Option 1**: Re-evaluate your model in anomaly detection mode
- Train on: Normal samples only
- Test on: Normal + all 9 attack types (mixed)
- Task: Binary classification (normal vs attack)
- **Prediction**: Your recall would likely be >98%, FAR much lower

**Option 2**: Show VLSTM can't do zero-day classification
- Train VLSTM on: Normal + 8 attack types
- Test on: 9th unseen attack type
- **Prediction**: Their performance would drop significantly

---

## Publication Strategy Implications

### ❌ Cannot Compare Directly
You're solving different problems:
- VLSTM: Anomaly detection (easier, binary)
- Your approach: Zero-day classification (harder, multi-class)

### ✅ What You CAN Do

#### 1. Reframe Your Contribution
**Title**: "Test-Time Training for Zero-Day Attack Type Classification"

**Key Message**:
- Different problem than anomaly detection
- Focus on classifying unseen attack types
- Compare with multi-class baselines (not anomaly detection)

#### 2. Include Anomaly Detection Baseline
Add a simple anomaly detection comparison:
- Train your base model on Normal only
- Test on Normal + all attacks (binary)
- Show you can achieve similar anomaly detection performance
- Then show your multi-class zero-day capability as added value

#### 3. Honest Positioning
"While anomaly detection approaches [VLSTM] achieve high recall (97.8%) by flagging anything abnormal, they cannot distinguish between attack types. Our approach achieves 93.99% recall on **classifying unseen attack types**, a significantly harder task requiring generalization across attack categories."

---

## Recommended Next Steps

### Option A: Re-run in Anomaly Detection Mode (RECOMMENDED)
1. Modify evaluation to binary (Normal vs Attack)
2. Show you can match VLSTM on anomaly detection
3. Then show your additional zero-day classification capability

### Option B: Find Multi-Class Zero-Day Baselines
1. Search for papers doing zero-day attack **classification** (not just detection)
2. Compare with those (fair comparison)
3. Acknowledge VLSTM solves different (easier) problem

### Option C: Combine Both
1. Show anomaly detection results (binary)
2. Show zero-day classification results (multi-class)
3. Position as "unified framework for both tasks"

---

## Bottom Line

**Your results are NOT inferior** - you're solving a **harder problem**.

- VLSTM: "Is this abnormal?" (Binary, easier)
- Your approach: "What type of unseen attack is this?" (Multi-class, harder)

**You need to either:**
1. Compare with multi-class zero-day classification baselines
2. Re-evaluate in binary anomaly detection mode
3. Clearly explain why your task is harder and metrics aren't directly comparable

**The 97.8% vs 93.99% recall difference is NOT a fair comparison** because:
- VLSTM only needs to detect "different from normal"
- You need to classify into specific unseen attack categories
- Your task has higher inherent difficulty
