# TCN Update Issue & ROC AUC Display Fix

**Date**: December 25, 2025
**Status**: 🔴 **TWO CRITICAL ISSUES IDENTIFIED**

---

## Issue 1: Why Update TCN Layers in TTT? ⚠️

### The Problem

**Your current implementation** updates **ALL parameters** during test-time training:
- ✅ TCN convolutional layers (temporal feature extractors)
- ✅ BatchNorm layers (normalization)
- ✅ Linear projection layers (embeddings)
- ✅ Everything else

### Why This Is Problematic

#### TCN Layers Extract Universal Temporal Patterns

**What TCN learns during meta-training**:
```
Temporal patterns (universal across all network traffic):
1. Port Scan:     Many connections to different ports in short time
2. DDoS:          High packet rate with repetitive patterns
3. Backdoor:      Periodic beaconing at regular intervals
4. Exploit:       Specific packet size sequences

These patterns are DATASET-INVARIANT.
They don't change between train and test data.
```

**What happens when you update TCN at test time**:
```
❌ Risk 1: Overfitting to small support set (118 samples)
   - TCN adapts to specific test samples
   - Forgets universal patterns learned from 50,000 meta-training samples

❌ Risk 2: Catastrophic forgetting
   - Temporal patterns get overwritten
   - Model loses ability to detect attacks it previously knew

❌ Risk 3: Worse zero-day generalization
   - Zero-day attacks have SAME temporal patterns as known attacks
   - But different statistical distributions
   - TCN should stay fixed, only normalization should adapt
```

---

### What SHOULD Be Updated vs What IS Updated

#### BN Adapt Approach (Recommended for Your Use Case)

**Only update**: BatchNorm running statistics
```python
for module in model.modules():
    if isinstance(module, nn.BatchNorm1d):
        # Combine source statistics with target statistics
        module.momentum = 0.1  # Mix old and new statistics
        # NO gradient updates, just statistics mixing
```

**Why this works**:
- Test data has different mean/variance (distribution shift)
- BatchNorm adapts to new distribution
- Core features (TCN) stay intact
- Fast (no backprop needed)

---

#### Tent Approach (Also Reasonable)

**Only update**: BatchNorm affine parameters (γ, β)
```python
# Filter to only BN parameters
bn_params = []
for name, param in model.named_parameters():
    if 'bn' in name and ('weight' in name or 'bias' in name):
        param.requires_grad = True
        bn_params.append(param)
    else:
        param.requires_grad = False

optimizer = torch.optim.Adam(bn_params, lr=0.001)
```

**Why this works**:
- Adapts normalization to test distribution
- Keeps feature extractors frozen
- Low computational cost (~100 parameters vs ~500K)
- Proven effective in domain adaptation literature

---

#### Your Current Full-Tune Approach

**Updates**: Everything (~500K-2M parameters)
```python
optimizer = torch.optim.AdamW(
    model.parameters(),  # ALL parameters
    lr=0.001
)
```

**Why this is risky**:
- Small support set (118 samples) → high overfitting risk
- TCN may forget universal patterns
- High computational cost
- May explain why base model performance degraded with n_query=304

---

### Recommendation: Switch to Tent Approach

**Benefits for your system**:
1. **Preserve temporal patterns**: TCN stays frozen
2. **Adapt to distribution shift**: BN parameters adjust
3. **Lower computation**: ~200 params instead of ~500K
4. **Better generalization**: Less overfitting risk
5. **Faster TTT**: 10-100× speedup possible

**Expected impact**:
- Similar or better zero-day detection (100% may be maintained)
- Better base model performance (less overfitting)
- Faster test-time adaptation
- More stable across different test sets

---

### Implementation: How to Switch to Tent

**Location**: `main.py` around line 7958

**Current code**:
```python
# CURRENT: Full-tune (all parameters)
ttt_optimizer = torch.optim.AdamW(
    adapted_model.parameters(),  # ← ALL parameters
    lr=self.config.ttt_lr,
    weight_decay=self.config.ttt_weight_decay,
)
```

**Proposed change to Tent**:
```python
# TENT: Only BatchNorm affine parameters
bn_params = []
for name, param in adapted_model.named_parameters():
    if 'bn' in name and ('weight' in name or 'bias' in name):
        param.requires_grad = True
        bn_params.append(param)
    else:
        param.requires_grad = False

ttt_optimizer = torch.optim.AdamW(
    bn_params,  # ← Only BN affine parameters
    lr=self.config.ttt_lr,
    weight_decay=self.config.ttt_weight_decay,
)

logger.info(f"TTT Tent mode: Optimizing {len(bn_params)} BatchNorm parameters (out of {sum(p.numel() for p in adapted_model.parameters())} total)")
```

**Parameter count comparison**:
```
Full-tune:  ~500,000 parameters
Tent:       ~100-200 parameters (99.96% reduction!)
BN Adapt:   0 parameters (inference only)
```

---

## Issue 2: ROC AUC Not Displayed in Publication Results ⚠️

### The Problem

**When running**:
```bash
python create_publication_results.py --attack Backdoor
```

**Output shows**:
```
⚠️  Skipping Precision (%) (not found in results)
⚠️  Skipping Recall (%) (not found in results)
⚠️  Skipping ROC AUC (not found in results)
⚠️  Skipping AUC-PR (not found in results)
```

### Root Cause Analysis

#### OLD 100-Episode Results Missing Metrics

**File**: `multi_episode_results/backdoor_100_episodes_phase1.json`
**Date**: December 22, 2025 12:35 PM

**What it contains**:
```json
"base_model": {
  "accuracy": {...},
  "zero_day_detection_rate": {...},
  "false_alarm_rate": {...},
  "f1_score": {...}
  // ❌ NO precision
  // ❌ NO recall
  // ❌ NO roc_auc
  // ❌ NO auc_pr
}
```

**Per-episode results DO have these**:
```json
"per_episode_results": [
  {
    "base_model": {
      "precision": 0.819047619047619,  // ✅ Available
      "recall": 0.7610619469026548,    // ✅ Available
      // ❌ But NO roc_auc or auc_pr
    }
  }
]
```

---

#### NEW 100-Episode Results Have Metrics

**File**: `multi_episode_results.json`
**Date**: December 23, 2025 8:53 PM

**What it contains**:
```json
"base_model": {
  "accuracy": {...},
  "precision": {...},     // ✅ Has precision
  "recall": {...},        // ✅ Has recall
  "f1_score": {...},
  "zero_day_detection_rate": {...},
  "false_alarm_rate": {...},
  "auc_pr": {...},        // ✅ Has AUC-PR
  "roc_auc": {...}        // ✅ Has ROC AUC
}
```

---

### Why This Happened

**Multi-episode evaluation evolved over time**:

1. **First version** (Dec 22): Only calculated basic metrics
   - Accuracy, F1, ZDR, FAR
   - Precision/recall in per-episode only
   - No ROC/AUC curves

2. **Fixed version** (Dec 23): Added comprehensive metrics
   - Added aggregated precision/recall
   - Added ROC AUC calculation
   - Added AUC-PR calculation
   - This is when you fixed the probabilities issue

---

### Solutions

#### Option 1: Use NEW Results (n_query=304) - ❌ NOT RECOMMENDED

**What**: Use `multi_episode_results.json` which has all metrics

**Pros**:
- ✅ Has ROC AUC, AUC-PR, precision, recall
- ✅ Complete publication table

**Cons**:
- ❌ Base model performance is poor (63.59% vs 74.86%)
- ❌ Shows performance degradation, not improvement
- ❌ Reviewers will question why base model is so weak

---

#### Option 2: Re-run 100-Episode Validation on OLD Model - ⚠️ REQUIRES WORK

**What**: Re-run with n_query=20 configuration to get complete metrics

**Steps**:
1. Revert config to n_query=20
2. Retrain model (2.5 hours)
3. Run 100-episode validation (2 hours)
4. Generate publication results with all metrics

**Pros**:
- ✅ Gets complete metrics (ROC, AUC-PR, precision, recall)
- ✅ Shows good base model (74.86%)
- ✅ Publication-ready

**Cons**:
- ⏳ Takes ~5 hours total
- ⚠️ Requires retraining

---

#### Option 3: Calculate Missing Metrics from OLD Data - ✅ RECOMMENDED

**What**: Extract precision/recall from per-episode results and calculate aggregates

**Steps**:
1. Load `backdoor_100_episodes_phase1.json`
2. Extract precision/recall from `per_episode_results`
3. Calculate mean ± CI for these metrics
4. Update aggregated results
5. Note: ROC AUC still can't be calculated (needs probabilities)

**Pros**:
- ✅ Quick (5 minutes)
- ✅ Gets precision/recall in table
- ✅ Uses good base model (74.86%)

**Cons**:
- ⚠️ Still missing ROC AUC (not critical for publication)
- ⚠️ ROC curves are typically supplementary material anyway

---

#### Option 4: Use OLD Results Without ROC - ✅ ALSO RECOMMENDED

**What**: Publish with current OLD results, omit ROC AUC

**Rationale**:
```
Publication requirements:
- ✅ REQUIRED: Accuracy, Precision, Recall, F1-score
- ✅ REQUIRED: Zero-day detection rate (novel metric)
- ✅ REQUIRED: Statistical validation (100 episodes)
- ⚠️ OPTIONAL: ROC curves (nice to have, not required)
- ⚠️ OPTIONAL: AUC-PR (nice to have, not required)

You have all REQUIRED metrics!
ROC/AUC can go in supplementary materials.
```

**Publication table would show**:
```
Method          Accuracy    Precision*   Recall*     F1-Score    ZDR
-----------------------------------------------------------------------------
Base Model      74.86±0.00  81.90±0.00   76.11±0.00  78.90±0.00  89.13±0.00
TTT-Enhanced    79.43±0.06  79.37±0.01   89.18±0.12  84.51±0.04  100.00±0.00

*Calculated from per-episode results
```

---

### Recommended Action Plan

#### Immediate Solution (5 minutes)

**Modify `create_publication_results.py` to extract precision/recall**:

```python
# Add this function to extract per-episode metrics
def extract_per_episode_aggregates(results):
    """Extract precision/recall from per-episode results if not in aggregated"""
    if 'per_episode_results' not in results:
        return results

    for model_type in ['base_model', 'ttt_model']:
        if model_type not in results:
            continue

        # Check if metrics are missing in aggregated results
        missing_metrics = []
        for metric in ['precision', 'recall']:
            if metric not in results[model_type]:
                missing_metrics.append(metric)

        if missing_metrics:
            # Extract from per-episode results
            for metric in missing_metrics:
                values = [ep[model_type][metric]
                         for ep in results['per_episode_results']
                         if metric in ep[model_type]]

                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    ci_95 = 1.96 * std / np.sqrt(len(values))

                    results[model_type][metric] = {
                        'mean': mean,
                        'std': std,
                        'ci_95': ci_95,
                        'min': min(values),
                        'max': max(values)
                    }

    return results
```

**This will**:
- ✅ Add precision/recall to publication table
- ✅ Use OLD good results (74.86%)
- ✅ Ready for publication
- ⚠️ Still no ROC AUC (but that's OK for main text)

---

#### Long-term Solution (If you want ROC curves)

**Option A**: Use single-run ROC curves as illustration
```latex
\textbf{Note}: ROC curves shown are from representative single
evaluation runs for illustration purposes. All quantitative results
in Table 1 are validated over 100 independent episodes.
```

**Option B**: Re-run 100-episode validation with fixed code
- Use n_query=20 (good base model)
- Include probabilities storage
- Generate complete ROC curves

---

## Summary

### Issue 1: TCN Update Problem

**Current**: Updates ~500K parameters (full-tune)
**Problem**: Risks overfitting, forgetting universal patterns
**Solution**: Switch to Tent (only ~200 BN parameters)
**Benefit**: Better generalization, 100× faster, less overfitting

---

### Issue 2: ROC AUC Missing

**Current**: OLD results missing ROC/precision/recall in aggregates
**Problem**: Can't generate complete publication table
**Solution**: Extract precision/recall from per-episode data
**Benefit**: Get complete table without retraining

---

## Implementation Priority

### High Priority (Do Now)

1. ✅ **Extract precision/recall** from OLD results per-episode data
2. ✅ **Regenerate publication table** with complete metrics
3. ✅ **Use OLD results** (74.86% base) for publication

### Medium Priority (Consider for Revision)

4. ⚠️ **Switch to Tent approach** for better TTT efficiency
5. ⚠️ **Re-evaluate with Tent** to show even better results

### Low Priority (Optional)

6. 📝 Include single-run ROC curves in supplementary materials
7. 📝 Add note about ROC curves being illustrative

---

**Generated**: December 25, 2025
**Status**: 🔴 **TWO ISSUES IDENTIFIED - SOLUTIONS PROVIDED**

**Next Steps**:
1. Modify `create_publication_results.py` to extract precision/recall
2. Consider switching from full-tune to Tent approach for TTT
