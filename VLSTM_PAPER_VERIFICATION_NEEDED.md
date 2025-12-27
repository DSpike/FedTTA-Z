# VLSTM Paper Verification - Critical Information Needed

**Date**: 2025-12-21

---

## The Metrics You Provided

You stated VLSTM's results are:
- Precision: 0.86 (86%)
- Recall: 0.978 (97.8%)
- F1-Score: 0.907 (90.7%)
- FAR: 0.117 (11.7%)
- AUC: 0.895 (89.5%)

---

## Critical Questions That MUST Be Answered

To determine if comparison is fair, we MUST verify from the VLSTM paper:

### 1. What Did They Train On?
- [ ] **Normal samples only** (one-class learning)?
- [ ] **Normal + all attack types** (multi-class learning)?
- [ ] **Normal + some attack types** (partial multi-class)?

### 2. What Did They Test On?
- [ ] **Standard train/test split** (70/30 or similar, all attack types in both)?
- [ ] **Leave-One-Attack-Out (LOAO)** (1 unseen attack type)?
- [ ] **Cross-validation** (k-fold)?

### 3. Which Metrics Are Reported?
Your values:
- Precision: 86%
- Recall: 97.8%
- F1: 90.7%
- FAR: 11.7%
- AUC: 89.5%

Are these from:
- [ ] **Table in paper** (which table number?)
- [ ] **Figure in paper** (which figure number?)
- [ ] **Text description** (which section?)

### 4. Binary or Multi-Class?
- [ ] **Binary classification** (Normal vs Attack)?
- [ ] **Multi-class classification** (Normal vs Attack Type 1 vs Attack Type 2...)?

---

## Why This Matters

### Scenario 1: VLSTM Used Standard Split with Normal-Only Training

**VLSTM:**
- Train: Normal samples only
- Test: Normal + attacks (standard 70/30 split)

**Your Approach:**
- Train: Normal + 8 attack types
- Test: Normal + 1 **unseen** attack type (LOAO)

**Implication:**
- Your evaluation is **MUCH HARDER**
- VLSTM's 97.8% recall is on **easier task**
- Not a fair comparison
- You can argue evaluation rigor advantage

### Scenario 2: VLSTM Also Used LOAO

**VLSTM:**
- Train: Normal only
- Test: Normal + 1 unseen attack type

**Your Approach:**
- Train: Normal + 8 attack types
- Test: Normal + 1 unseen attack type

**Implication:**
- Both use LOAO (**same difficulty**)
- VLSTM is **clearly superior** (97.8% vs 93.99% recall, 11.7% vs 42.53% FAR)
- Your results are **NOT competitive**
- Cannot claim superiority

### Scenario 3: VLSTM Used Standard Split with All Attack Types in Training

**VLSTM:**
- Train: Normal + all 9 attack types (70%)
- Test: Normal + all 9 attack types (30%)

**Your Approach:**
- Train: Normal + 8 attack types
- Test: Normal + 1 **unseen** attack type (LOAO)

**Implication:**
- Your evaluation is **MUCH HARDER** (unseen attack types)
- VLSTM's 97.8% recall is on **seen attack types** (much easier)
- Not a fair comparison
- You can strongly argue evaluation rigor advantage

---

## What We Know from Search Results

From my web searches, I found:

1. **Paper**: "Variational LSTM Enhanced Anomaly Detection for Industrial Big Data" by Zhou et al., IEEE Trans. Ind. Informat., 2021
   - Source: [ResearchGate](https://www.researchgate.net/publication/344859952_Variational_LSTM_Enhanced_Anomaly_Detection_for_Industrial_Big_Data)
   - Source: [IEEE Xplore](https://ieeexplore.ieee.org/document/9195000/)

2. **Dataset**: UNSW-NB15 (same as yours)
   - Standard partition: 175,341 training records, 82,332 testing records
   - Source: [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
   - Source: [Papers With Code](https://paperswithcode.com/dataset/unsw-nb15)

3. **Evaluation Metrics**: F1 score, AUC, FAR
   - Source: [GitHub - Deep Learning Anomaly Detection](https://github.com/bitzhangcy/Deep-Learning-Based-Anomaly-Detection)

**What I COULD NOT find**:
- ❌ Exact train/test split methodology
- ❌ Whether they used LOAO or standard split
- ❌ Whether they trained on normal only or normal + attacks
- ❌ The exact metrics you provided (86%, 97.8%, 90.7%, 11.7%, 89.5%)

---

## How to Get This Information

### Option 1: Read the Paper Yourself (BEST)
1. Access the paper via your institution: [IEEE Xplore Link](https://ieeexplore.ieee.org/document/9195000/)
2. Look for "Experimental Setup" or "Evaluation Methodology" section
3. Find the exact train/test split description
4. Find the exact metrics table

### Option 2: Check Paper Abstract/Introduction
1. The methodology is usually described in abstract or introduction
2. Check if they mention "one-class", "semi-supervised", or "unsupervised"
3. Check if they mention "Leave-One-Attack-Out" or "LOAO"

### Option 3: Check Figures/Tables in Paper
1. Look for performance comparison tables
2. Check figure captions for evaluation setup
3. Compare metrics with what you provided

---

## Your Current Results vs VLSTM (Unverified)

| Metric | VLSTM (You Provided) | Your TTT | Gap | VLSTM Better? |
|--------|---------------------|----------|-----|---------------|
| Precision | 86.00% | ~65.70% | -20.30% | ✅ YES |
| Recall | 97.80% | 93.99% | -3.81% | ✅ YES |
| F1-Score | 90.70% | 68.69% | -22.01% | ✅ YES |
| FAR | 11.70% | 42.53% | +30.83% | ✅ YES (lower) |
| AUC | 89.50% | ??? | ??? | ??? |

**VLSTM is superior on ALL metrics** (if comparison is fair).

---

## Immediate Action Required

**You MUST**:
1. Access the VLSTM paper and read the evaluation methodology section
2. Verify the exact metrics (86%, 97.8%, 90.7%, 11.7%, 89.5%)
3. Confirm train/test split methodology (LOAO or standard)
4. Confirm training data (normal only or normal + attacks)

**Then**:
- If VLSTM used standard split → Your LOAO is much harder (not fair comparison)
- If VLSTM used LOAO → Same difficulty (VLSTM is clearly superior)

---

## Temporary Conclusion (Pending Verification)

**Most Likely Scenario** (based on typical anomaly detection papers):
- VLSTM used **standard train/test split** (not LOAO)
- VLSTM trained on **normal samples only** (one-class learning)
- VLSTM tested on **normal + attacks** (standard partition)

**If this is true**:
- Your LOAO evaluation is **significantly harder**
- Your 93.99% on LOAO ≈ Their 97.8% on standard split (accounting for difficulty)
- You can publish with emphasis on **rigorous zero-day evaluation**
- BUT metrics are still lower, so honest acknowledgment needed

**Bottom Line**: **VERIFY THE PAPER** before making any publication claims.

---

## Sources

- [ResearchGate - VLSTM Paper](https://www.researchgate.net/publication/344859952_Variational_LSTM_Enhanced_Anomaly_Detection_for_Industrial_Big_Data)
- [IEEE Xplore - VLSTM Paper](https://ieeexplore.ieee.org/document/9195000/)
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [Papers With Code - UNSW-NB15](https://paperswithcode.com/dataset/unsw-nb15)
- [GitHub - Deep Learning Anomaly Detection](https://github.com/bitzhangcy/Deep-Learning-Based-Anomaly-Detection)
