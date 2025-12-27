# Base + TTT Ensemble Implementation - COMPLETE

**Date**: 2025-12-21
**Status**: ✅ Implemented and Ready to Test

---

## What Was Implemented

Implemented a **Base + TTT Ensemble** approach as an alternative strategy to reduce False Alarm Rate (FAR) while maintaining high Zero-Day Detection Rate (ZDR).

### Key Concept

Instead of trying to fix TTT's overconfidence problem directly, we leverage the complementary strengths of both models:

- **Base Model**: Conservative, low FAR (25.94%), moderate ZDR (81.05%)
- **TTT Model**: Aggressive, high ZDR (93.99%), high FAR (42.53%)
- **Ensemble**: Combines both to achieve balanced performance

---

## Files Modified

### 1. [base_ttt_ensemble.py](base_ttt_ensemble.py) - NEW FILE

**Complete ensemble predictor with three methods**:

#### Method 1: Weighted Probability Ensemble
```python
ensemble_probs = alpha * base_probs + (1-alpha) * ttt_probs
```
Simple linear combination of probabilities.

#### Method 2: Confidence-Weighted Ensemble (RECOMMENDED)
```python
Zone 1: Base VERY confident about normal (>0.85) → Use base (reduces FAR)
Zone 2: TTT confident about attack (>0.70) → Use TTT (maintains ZDR)
Zone 3: Uncertain region → Weighted average
```
Zone-based decision leveraging each model's strengths.

#### Method 3: Voting Ensemble
```python
Predict attack if EITHER model predicts attack
```
Maximizes recall but may have highest FAR.

### 2. [config.py](config.py:644-676)

**Added ensemble configuration parameters**:

```python
# Enable/disable ensemble
use_ensemble: bool = True  # ✅ ENABLED

# Ensemble method selection
ensemble_method: str = 'confidence_weighted'  # RECOMMENDED

# Weighted probability parameters
ensemble_base_weight: float = 0.4  # 40% base, 60% TTT

# Confidence-weighted parameters (RECOMMENDED)
ensemble_base_conf_threshold: float = 0.85  # Base must be >0.85 to override
ensemble_ttt_conf_threshold: float = 0.70   # TTT must be >0.70 to override

# Decision threshold
ensemble_decision_threshold: float = 0.5
```

### 3. [multi_episode_evaluation.py](multi_episode_evaluation.py:114-249)

**Integrated ensemble into evaluation pipeline**:

- Line 114-185: Ensemble evaluation logic
- Line 232-249: Added ensemble results to episode metrics
- Line 349-357: Extracted ensemble metrics from episodes
- Line 403-415: Added ensemble statistics to aggregated results
- Line 453-497: Added ensemble to summary printing

**Key changes**:
- Automatically runs ensemble evaluation when `config.use_ensemble = True`
- Extracts probabilities from both base and TTT models
- Computes ensemble predictions using selected method
- Calculates all metrics (Accuracy, ZDR, FAR, F1-Score) for ensemble
- Tracks ensemble statistics across episodes

### 4. [run_comprehensive_multi_episode_evaluation.py](run_comprehensive_multi_episode_evaluation.py:122-176)

**Updated comprehensive report generation**:

- Line 122-129: Check for ensemble results and extract metrics
- Line 164-176: Added ensemble to overall statistics

**Report now includes**:
- Ensemble model performance (mean ± CI)
- Ensemble improvement vs Base model
- Comparison table: Base vs TTT vs Ensemble

---

## How It Works

### Evaluation Flow

1. **Base Model Evaluation** → Get base_probabilities
2. **TTT Adaptation** → Get ttt_probabilities
3. **TTT Model Evaluation** → Get TTT metrics
4. **Ensemble Evaluation** (NEW):
   - Extract probabilities from both models
   - Apply ensemble method (confidence_weighted)
   - Compute ensemble predictions
   - Calculate ensemble metrics (ZDR, FAR, Accuracy, F1)
5. **Aggregate Results** → Mean ± 95% CI across episodes
6. **Generate Report** → Compare all three models

### Confidence-Weighted Logic (Recommended Method)

```python
for each sample:
    base_confidence = max(base_probs)
    ttt_confidence = max(ttt_probs)

    if base_confidence > 0.85 and base predicts Normal:
        # Zone 1: Trust base model (reduces FAR)
        ensemble_prediction = base_prediction

    elif ttt_confidence > 0.70 and ttt predicts Attack:
        # Zone 2: Trust TTT model (maintains ZDR)
        ensemble_prediction = ttt_prediction

    else:
        # Zone 3: Uncertain - use weighted average
        ensemble_probs = 0.4 * base_probs + 0.6 * ttt_probs
        ensemble_prediction = threshold(ensemble_probs, 0.5)
```

---

## Expected Performance

### Conservative Estimate
- **FAR**: 30-35% (down from 42.53%, ~20% relative reduction)
- **ZDR**: 88-92% (slight drop from 93.99%)
- **Accuracy**: 75-80%
- **F1-Score**: 75-80%

### Realistic Estimate
- **FAR**: 25-30% (down from 42.53%, ~35% relative reduction)
- **ZDR**: 90-93% (maintained)
- **Accuracy**: 80-85%
- **F1-Score**: 80-85%

### Optimistic Estimate
- **FAR**: 20-25% (down from 42.53%, ~45% relative reduction)
- **ZDR**: 91-94% (maintained or slightly improved)
- **Accuracy**: 82-88%
- **F1-Score**: 82-88%

**Note**: Unlikely to achieve target FAR <10%, but significant improvement expected.

---

## How to Run

### Test on Single Attack (Quick Test ~20 min)

```bash
cd "c:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn"

# Run on DoS attack with 3 episodes
python multi_episode_evaluation.py --attack DoS --episodes 3 --episode-size 800
```

**Expected output**:
- Base model results
- TTT model results
- **Ensemble model results** (NEW)
- Comparison table showing all three

### Comprehensive Evaluation (All 9 Attacks, ~3 hours)

```bash
cd "c:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn"

# Run all 9 attacks with 10 episodes each
python run_comprehensive_multi_episode_evaluation.py --episodes 10 --episode-size 800
```

**When prompted, type `yes` to confirm**.

**Output files**:
- `multi_episode_results/comprehensive_multi_episode_results.json`
- `multi_episode_results/comprehensive_multi_episode_results.md`
- `multi_episode_results/multi_episode_{attack}.json` (per attack)

---

## What to Look For in Results

### Success Indicators

✅ **Target Met** (FAR 20-30%, ZDR >90%):
- FAR reduced by ~30-40% compared to TTT alone
- ZDR maintained at >90%
- Accuracy improved compared to Base
- F1-Score >80%

✅ **Strong Success** (FAR 15-25%, ZDR >90%):
- FAR reduced by ~40-50% compared to TTT alone
- ZDR maintained at >90%
- Approaching mid-tier publication threshold

⚠️ **Needs Tuning** (FAR 30-35%, ZDR >88%):
- Some improvement but not enough
- Adjust ensemble parameters (see Tuning Guide below)

### Comparison Table

The results will show:

| Model | ZDR | FAR | Accuracy | F1-Score |
|-------|-----|-----|----------|----------|
| Base | 81.05% | 25.94% | 72.51% | 63.81% |
| TTT | 93.99% | 42.53% | 69.97% | 68.69% |
| **Ensemble** | **~90%** | **~28%** | **~78%** | **~75%** |

**Goal**: Ensemble achieves middle ground - ZDR closer to TTT, FAR closer to Base.

---

## Tuning Guide

If ensemble results don't meet expectations, tune these parameters in [config.py](config.py:650-671):

### To Reduce FAR Further

```python
# Make base model more influential
ensemble_base_weight: float = 0.5  # Increase from 0.4
ensemble_base_conf_threshold: float = 0.80  # Decrease from 0.85 (easier to trust base)
ensemble_ttt_conf_threshold: float = 0.75  # Increase from 0.70 (harder to trust TTT)
```

### To Maintain/Improve ZDR

```python
# Make TTT model more influential
ensemble_base_weight: float = 0.3  # Decrease from 0.4
ensemble_base_conf_threshold: float = 0.90  # Increase from 0.85 (harder to trust base)
ensemble_ttt_conf_threshold: float = 0.65  # Decrease from 0.70 (easier to trust TTT)
```

### Try Different Ensemble Methods

```python
# Method 1: Simple weighted average (fastest, most stable)
ensemble_method: str = 'weighted_prob'
ensemble_base_weight: float = 0.4  # Tune this (0.3-0.5)

# Method 2: Confidence-weighted (current, recommended)
ensemble_method: str = 'confidence_weighted'
# Tune base_conf_threshold and ttt_conf_threshold

# Method 3: Voting (highest ZDR, highest FAR)
ensemble_method: str = 'voting'
# No parameters to tune
```

---

## Next Steps

### Option 1: Quick Test First (Recommended)

1. Run single attack test (DoS, 3 episodes)
2. Check ensemble FAR and ZDR
3. If promising, run comprehensive evaluation
4. If not, tune parameters and re-test

### Option 2: Full Comprehensive Evaluation

1. Run all 9 attacks with 10 episodes each (~3 hours)
2. Analyze comprehensive results
3. Compare Base vs TTT vs Ensemble
4. Decision on publication strategy

---

## Implementation Status

✅ **Completed**:
1. Created `base_ttt_ensemble.py` with three ensemble methods
2. Added ensemble configuration to `config.py`
3. Integrated ensemble into `multi_episode_evaluation.py`
4. Updated comprehensive report generation
5. Enabled ensemble (`use_ensemble = True`)

⏳ **Pending**:
1. Run evaluation to test ensemble performance
2. Analyze results and compare with expectations
3. Tune parameters if needed
4. Final comprehensive evaluation

---

## Why This Might Work

### Theoretical Justification

1. **Complementary Strengths**: Base is conservative (low FAR), TTT is aggressive (high ZDR)
2. **Zone-Based Decision**: Use each model where it's most confident
3. **Calibration**: Base model is better calibrated, ensemble inherits this
4. **Proven Approach**: Ensemble methods are standard in ML for bias-variance tradeoff

### Comparison to Confidence Regularization

- **Confidence Reg**: Tried to fix TTT directly → **Failed** (FAR 42.53%)
- **Ensemble**: Leverage Base model's conservative predictions → **Testing**

### Expected Outcome

**Realistic expectation**: FAR 25-30% (improvement but not target <10%)
**Best case**: FAR 20-25%
**Worst case**: FAR 30-35%

Even worst case is better than current 42.53% and may be publishable in mid-tier venues with honest analysis.

---

## Ready to Test!

The implementation is complete. Run the quick test or comprehensive evaluation to see results.

```bash
# Quick test (20 min)
python multi_episode_evaluation.py --attack DoS --episodes 3 --episode-size 800

# OR comprehensive evaluation (3 hours)
echo "yes" | python run_comprehensive_multi_episode_evaluation.py --episodes 10 --episode-size 800
```
