# Fair Comparison: Your Approach vs VLSTM (Binary Anomaly Detection)

**Date**: 2025-12-21

---

## CRITICAL FINDING: You're Both Doing Binary Classification!

After reviewing your code, I confirmed that your evaluation uses **binary labels**:
- `y_test`: 0 = Normal, 1 = Attack (any attack type)
- Evaluation metrics computed on binary classification
- This is **the same task as VLSTM** - binary anomaly detection

**However**, there's a crucial difference in **training methodology**:

---

## Training Methodology Comparison

### VLSTM Approach:
```
Training: ONLY Normal samples (one-class learning)
Testing: Normal + Attack samples (binary: is it normal or attack?)
```

### Your Approach (LOAO):
```
Training: Normal + 8 attack types (multi-class learning)
Testing: Normal + 1 held-out attack type (binary: is it normal or attack?)
```

**Key Difference**:
- VLSTM trains on normal only (one-class)
- You train on normal + 8 attacks (multi-class), then evaluate binary

---

## Why This Matters: Zero-Day Detection

### VLSTM's Challenge:
- Trains on: Normal samples
- Tests on: **All attack types** (including those seen in training if standard split)
- **If they used LOAO**, they would only train on normal (no attack examples)

### Your Advantage:
- Trains on: Normal + 8 attack types
- Tests on: **1 unseen attack type** (true zero-day)
- Model learns attack patterns, must generalize to new attack type

**This is actually HARDER for you** because:
1. VLSTM learns "anything abnormal is attack" (simple boundary)
2. You learn specific attack patterns, must generalize to unseen attacks
3. Risk of overfitting to seen attack types

---

## The Real Comparison

### VLSTM Results (Binary Anomaly Detection):
| Metric | Value | What it means |
|--------|-------|---------------|
| Precision | 86.00% | 86% of flagged attacks are real attacks |
| Recall | 97.80% | Catches 97.8% of all attacks |
| F1-Score | 90.70% | Balanced performance |
| FAR | 11.70% | 11.7% of normal samples flagged as attacks |
| AUC | 89.50% | Good separability |

**Evaluation**: Likely standard train/test split (not LOAO)
- Trains on: Normal only
- Tests on: Normal + attacks (may include same attack types from training)

### Your Results (Binary Zero-Day Detection with LOAO):
| Metric | Value | What it means |
|--------|-------|---------------|
| Precision | ~65.70% | 65.7% of flagged attacks are real attacks |
| Recall (ZDR) | 93.99% | Catches 93.99% of **unseen** attack type |
| F1-Score | 68.69% | Balanced performance |
| FAR | 42.53% | 42.53% of normal samples flagged as attacks |
| AUC | ??? | Need to calculate |

**Evaluation**: Leave-One-Attack-Out (LOAO)
- Trains on: Normal + 8 attack types
- Tests on: Normal + 1 **completely unseen** attack type

---

## Why Your Results Look Worse (But Might Not Be)

### Evaluation Rigor:
```
VLSTM (Standard Split):
├─ Train: Normal only
├─ Test: Normal + Attacks (possibly same types)
└─ Challenge: Detect known anomaly patterns

Your Approach (LOAO):
├─ Train: Normal + 8 attack types
├─ Test: Normal + 1 UNSEEN attack type
└─ Challenge: Generalize to new attack patterns
```

**LOAO is significantly harder** because:
1. Zero-day attack has never been seen (different signatures)
2. Model must generalize from 8 attacks to 9th attack
3. Risk of rejecting new patterns as normal

### Example Scenario:

**Seen attacks (training)**: DoS, Exploits, Fuzzers, Generic, Shellcode, Worms, Analysis, Backdoor

**Unseen attack (test)**: Reconnaissance

**Your model must**:
- Recognize Reconnaissance as attack (never seen before)
- Not confuse it with normal traffic
- Generalize attack patterns from seen types

**VLSTM approach**:
- Only needs to flag "different from normal"
- Does not need to understand attack types
- Natural advantage on anything abnormal

---

## What Your 93.99% Recall Really Means

### Your Achievement:
**93.99% zero-day detection rate** means:
- Out of 100 samples of a **completely unseen attack type**
- Your model correctly identifies 94 as attacks
- Only 6 are missed (mistaken for normal)

**This is impressive** because:
- The attack type was never seen during training
- Model must generalize from 8 attack types to 9th
- Achieving 93.99% on truly novel attacks is strong

### VLSTM's 97.8% Recall:
If VLSTM used standard split:
- Out of 100 attack samples (possibly same types as training)
- Model correctly identifies 97.8 as attacks

**Easier task** because:
- Attack patterns may have been seen during training (just different samples)
- One-class model flags anything abnormal
- Not true zero-day scenario

---

## The Harsh Reality: Your Metrics Are Still Lower

Even accounting for harder evaluation:

| Metric | VLSTM | Your TTT | Gap |
|--------|-------|----------|-----|
| Recall | 97.80% | 93.99% | -3.81% |
| Precision | 86.00% | 65.70% | -20.30% |
| F1-Score | 90.70% | 68.69% | -22.01% |
| FAR | 11.70% | 42.53% | +30.83% |
| AUC | 89.50% | ??? | ??? |

**Even if LOAO is harder, the gaps are large:**
- FAR is 3.6x higher (42.53% vs 11.70%)
- Precision is 20% lower
- F1-Score is 22% lower

**Honest Assessment**: Your results are **not competitive** with VLSTM, even accounting for evaluation difficulty.

---

## Publication Strategy Options

### Option 1: Re-Evaluate with Standard Split (RECOMMENDED)

**Test if LOAO is the problem**:
1. Re-run evaluation with standard 70/30 train/test split
2. Include all 9 attack types in both train and test
3. See if metrics improve to be competitive with VLSTM

**Expected Results**:
- Recall: 93.99% → 96-98%
- FAR: 42.53% → 25-30%
- F1: 68.69% → 75-85%

**If metrics improve significantly**:
- Shows LOAO was making evaluation much harder
- You can position as "competitive on standard evaluation, superior on zero-day evaluation"

**If metrics DON'T improve**:
- Indicates fundamental model limitation
- Need to improve approach before publication

### Option 2: Focus on Test-Time Training Contribution

**Reframe the paper**:
- Title: "Test-Time Training for Network Intrusion Detection"
- Focus: TTT methodology, not absolute performance
- Contribution: Show TTT improves over base model
- Compare: Base (81.05%) vs TTT (93.99%) = +12.94% improvement

**Acknowledge**:
- VLSTM achieves better absolute performance
- Your contribution is TTT adaptation method
- Future work: Combine TTT with better base model

### Option 3: Find Papers with LOAO Evaluation

**Search for**:
- Papers using LOAO on UNSW-NB15
- Zero-day detection papers with similar evaluation
- Compare with those (fair methodology)

**Expected**:
- LOAO papers likely have lower metrics than standard split
- Your 93.99% may be competitive in that context

---

## Immediate Next Steps

1. **Calculate AUC** from your results to complete metric comparison
2. **Re-run with standard split** to test if LOAO is causing low metrics
3. **Search for LOAO baselines** to find fair comparisons

Would you like me to:
- A) Help calculate AUC from your current results?
- B) Modify evaluation to use standard train/test split?
- C) Search for papers using LOAO evaluation on UNSW-NB15?
