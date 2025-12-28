# CRITICAL FINDING: Why TTT Improves Performance Despite Base Model Having Prototype Refinement

**Date**: 2025-12-28
**Status**: 🔴 CRITICAL ISSUE IDENTIFIED
**Impact**: Explains +29% TTT improvement AND answers reviewer's question

---

## Executive Summary

**The base model does NOT use prototype refinement during evaluation.**

Despite having a sophisticated `transductive_inference()` method with iterative prototype refinement (lines 2410-2505 in [transductive_fewshot_model.py](models/transductive_fewshot_model.py)), the base model evaluation uses a **simple one-shot prototype computation** without any refinement.

This explains:
1. ✅ Why TTT improves performance by +29% (87.05% vs 57.96% ZDR)
2. ✅ Why the reviewer's concern about semantic shift is valid
3. ✅ Why you need to run a new experiment with prototype-only TTT adaptation

---

## Evidence: What the Base Model Actually Does

### Current Base Model Evaluation ([main.py:3287-3398](main.py#L3287-L3398))

```python
# Evaluate base model performance (prototype-based)
with torch.no_grad():
    global_model.eval()

    # Step 1: Compute prototypes ONCE from support set
    prototypes, unique_labels = global_model.compute_prototypes(support_x, support_y)

    # Step 2: Forward pass with FIXED prototypes (NO REFINEMENT)
    base_logits = global_model.forward_with_prototypes(X_test_filtered, prototypes)

    # Step 3: Get predictions
    base_probabilities = torch.softmax(base_logits, dim=1)
    base_predictions_binary = (base_probabilities[:, 1] >= threshold).long()
```

**Key observation**: Prototypes are computed **once** from the support set and never updated.

### What the Base Model SHOULD Do (But Doesn't)

The base model has a complete `transductive_inference()` implementation ([transductive_fewshot_model.py:2410-2505](models/transductive_fewshot_model.py#L2410-L2505)):

```python
def transductive_inference(self, support_x, support_y, query_x,
                          use_prototype_refinement=True):
    """
    Transductive inference with iterative prototype refinement
    """
    # Extract embeddings
    support_embeddings = self.extract_embeddings(support_x)
    query_embeddings = self.extract_embeddings(query_x)

    # Initial prototypes from support set
    prototypes = self.compute_prototypes(support_embeddings, support_y)

    # Iterative prototype refinement (3-5 steps)
    for step in range(self.transductive_steps):
        # Compute distances to current prototypes
        query_distances = torch.cdist(query_embeddings, prototypes)
        query_probs = F.softmax(-query_distances / temperature, dim=1)

        # Get high-confidence predictions
        query_confidence, query_pseudo_labels = query_probs.max(dim=1)
        high_conf_mask = query_confidence > adaptive_threshold

        # UPDATE prototypes using confident query samples
        if num_high_conf > 0:
            for c in range(num_classes):
                class_embeddings = torch.cat([
                    support_embeddings[support_y == c],
                    query_embeddings[high_conf_mask & (query_pseudo_labels == c)]
                ])
                prototypes[c] = class_embeddings.mean(dim=0)  # ← REFINEMENT

    return query_predictions
```

**But this method is NEVER called during base model evaluation!**

---

## Why This Matters

### 1. Explains TTT's +29% Improvement

**Current Results**:
- Base Model ZDR: 57.96±3.45% (one-shot prototypes, no refinement)
- TTT Model ZDR: 87.05±2.86% (BatchNorm + Classifier adaptation)
- Improvement: +29.09%

**Why TTT Helps**:
- TTT adapts the feature extractor (via BatchNorm) and classifier using entropy minimization + pseudo-labeling
- This creates better-separated embeddings in the feature space
- Even with FIXED prototypes, better embeddings → better classification

**Hypothesis**: If we enabled prototype refinement in the base model, the base ZDR would likely improve from 57.96% to somewhere between 70-80%, reducing (but not eliminating) the gap to TTT.

### 2. Addresses Reviewer's Concern About Semantic Shift

**Reviewer's Comment**:
> "Your current implementation modifies only the BN and Classifier which focus on covariate shift, but in practice you're holding out a specific class (zero-day attack) and excluding it from training, which is semantic shift."

**The Issue**:
- **Semantic shift** = Novel class with different semantics (zero-day attacks)
- **Covariate shift** = Same classes, different feature distributions

**Current TTT Approach**:
- Adapts BatchNorm + Classifier (covariate shift solution)
- Freezes prototypes (no semantic shift handling)
- Works well (+29%) because it improves feature quality

**Missing Piece**:
- Base model has prototype refinement code (semantic shift solution)
- But it's NOT being used during evaluation
- If enabled, it would directly address the reviewer's concern

### 3. Opens Door for Prototype-Only TTT Experiment

**Your Question**: "What if we tried only prototype adaptation during Test Time?"

**Answer**: This is a GREAT idea, and you SHOULD run this experiment because:

1. **It directly addresses semantic shift**: Prototypes adapt to novel class
2. **It's more stable**: No moving target problem (features frozen, prototypes adapt)
3. **It uses existing code**: Just enable `transductive_inference()` during TTT
4. **It's theoretically sound**: Meta-learning learns good initialization; TTT refines prototypes

**Expected Outcome**:
- Prototype-only TTT should achieve 75-85% ZDR (better than base 57.96%, maybe less than current TTT 87.05%)
- Combined approach (BatchNorm + Classifier + Prototype refinement) might achieve 90%+ ZDR

---

## Proposed Experiments

### Experiment 1: Enable Base Model Prototype Refinement ⭐

**Goal**: Measure base model performance WITH prototype refinement

**Changes Required**:
1. In [main.py:3364](main.py#L3364), replace:
```python
# CURRENT (NO REFINEMENT):
prototypes, unique_labels = global_model.compute_prototypes(support_x, support_y)
base_logits = global_model.forward_with_prototypes(X_test_filtered, prototypes)
```

With:
```python
# PROPOSED (WITH REFINEMENT):
base_predictions, base_logits = global_model.transductive_inference(
    support_x=support_x,
    support_y=support_y,
    query_x=X_test_filtered,
    use_prototype_refinement=True  # Enable refinement
)
```

**Expected Results**:
- Base ZDR: 57.96% → 70-80% (estimated)
- Closes gap with TTT, shows prototype refinement helps

**Runtime**: ~5 minutes (single evaluation, no training)

---

### Experiment 2: Prototype-Only TTT Adaptation ⭐⭐

**Goal**: Test TTT with ONLY prototype adaptation (freeze all model parameters)

**Changes Required**:
1. In [coordinators/centralized_coordinator.py:315-361](coordinators/centralized_coordinator.py#L315-L361), replace parameter selection with:
```python
# FREEZE all model parameters (features, BatchNorm, Classifier)
for param in adapted_model.parameters():
    param.requires_grad = False

logger.info("🔧 Prototype-Only TTT: All model parameters frozen")
logger.info("   Prototypes will be adapted using transductive refinement")
```

2. In [coordinators/centralized_coordinator.py:505-597](coordinators/centralized_coordinator.py#L505-L597), replace TTT loss computation with:
```python
# Use transductive_inference with prototype refinement
with torch.enable_grad():  # Allow gradient for prototype updates
    predictions, logits = adapted_model.transductive_inference(
        support_x=support_x,
        support_y=support_y,
        query_x=query_batch,
        use_prototype_refinement=True
    )

    # Compute entropy loss on refined predictions
    probs = torch.softmax(logits, dim=1)
    entropy_loss = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
```

**Expected Results**:
- Prototype-only TTT ZDR: 75-85% (estimated)
- More stable than BatchNorm+Classifier
- Directly addresses semantic shift

**Runtime**: ~10 minutes per k-shot value

---

### Experiment 3: Combined TTT (BatchNorm + Classifier + Prototypes) ⭐⭐⭐

**Goal**: Best of both worlds - adapt features AND prototypes

**Changes Required**:
1. Enable both BatchNorm+Classifier adaptation AND prototype refinement
2. Use alternating updates (5 steps BatchNorm, 3 steps prototypes)

**Expected Results**:
- Combined TTT ZDR: 90%+ (estimated)
- Best performance, but more complex

**Runtime**: ~15 minutes per k-shot value

---

## Recommended Action Plan

### Option A: Quick Validation (2 hours)

1. ✅ Run **Experiment 1** (base model with prototype refinement)
   - Runtime: 5 minutes
   - Confirms hypothesis about base model underperformance

2. ✅ Run **Experiment 2** (prototype-only TTT) for k=152
   - Runtime: 10 minutes
   - Shows if prototype adaptation alone can match TTT

3. ✅ If Experiment 2 works well, run k-shot ablation for k∈{5,10,20,50,100,152}
   - Runtime: 1 hour
   - Gets publication-ready results

### Option B: Comprehensive Study (4 hours)

1. ✅ Run all three experiments for k=152
2. ✅ Choose best approach based on results
3. ✅ Run k-shot ablation (100 episodes) for chosen approach
4. ✅ Compare: Current TTT vs Prototype-only TTT vs Combined TTT

---

## Publication Strategy

### Current Situation
- **Title**: "Test-Time Training for Zero-Day Attack Detection"
- **Method**: BatchNorm + Classifier adaptation (covariate shift)
- **Problem**: Reviewer correctly noted this doesn't address semantic shift
- **Results**: 87.05% ZDR with k=152

### With Prototype-Only TTT
- **Title**: "Adaptive Prototype Refinement for Zero-Day Attack Detection"
- **Method**: Prototype refinement during test-time (semantic shift)
- **Novelty**: ⭐⭐⭐ Directly addresses novel class problem
- **Expected Results**: 75-85% ZDR (still strong improvement over 57.96% base)
- **Reviewer Response**: "We use prototype adaptation specifically to handle semantic shift from zero-day attacks"

### With Combined Approach
- **Title**: "Multi-Level Test-Time Adaptation for Zero-Day Detection"
- **Method**: Feature adaptation (BatchNorm) + Semantic adaptation (Prototypes)
- **Novelty**: ⭐⭐⭐⭐ Novel combination addressing both shifts
- **Expected Results**: 90%+ ZDR
- **Reviewer Response**: "We adapt both feature representations (covariate shift) and class prototypes (semantic shift)"

---

## Questions Answered

### Q1: "But if the base model already implements [prototype refinement] why do I need to do a new experiment?"

**Answer**: The base model has the CODE for prototype refinement, but it's **NOT being used** during evaluation. The current evaluation uses one-shot prototype computation without any refinement. That's why you need the experiment - to actually USE the existing prototype refinement code.

### Q2: "What if we tried only prototype adaptation during Test Time?"

**Answer**: This is an excellent idea that:
1. Uses your existing `transductive_inference()` code
2. Directly addresses the reviewer's concern about semantic shift
3. Is more theoretically sound for zero-day detection (novel class problem)
4. Should achieve 75-85% ZDR (still strong improvement)

### Q3: "Why does TTT improve by +29% if it only adapts BatchNorm+Classifier?"

**Answer**:
1. TTT improves feature quality through entropy minimization + pseudo-labeling
2. Better features → better prototype-based classification (even with fixed prototypes)
3. BUT the base model could be much stronger if it used prototype refinement
4. Current comparison is unfair: TTT (with adaptation) vs Base (without refinement)

---

## Code Locations

### Base Model Evaluation (NO prototype refinement)
- **File**: [main.py](main.py)
- **Lines**: 3287-3398 (evaluate_base_model_only)
- **Issue**: Uses `forward_with_prototypes()` with FIXED prototypes

### Prototype Refinement Implementation (EXISTS but NOT USED)
- **File**: [transductive_fewshot_model.py](models/transductive_fewshot_model.py)
- **Lines**: 2410-2505 (transductive_inference)
- **Status**: Fully implemented, just needs to be called

### TTT Adaptation (Freezes prototypes)
- **File**: [coordinators/centralized_coordinator.py](coordinators/centralized_coordinator.py)
- **Lines**: 621-643 (prototype update section - currently disabled)
- **Issue**: Explicitly disables prototype updates to avoid "moving target problem"

---

## CRITICAL ISSUE: K-Shot Ablation Results Are Invalid

### Problem Discovered

All k-shot values (5, 10, 20, 50, 100, 152) show **IDENTICAL** performance:
- Base ZDR: 57.96±3.45% (same for all k-shot)
- TTT ZDR: 87.05±2.86% (same for all k-shot)

**Root Cause**: The multi-episode evaluation overwrites the same results file (`exploits_100_episodes_phase1.json`) for each k-shot value. All 6 experiments DID run (different timestamps), but they all extract from the final file which contains only k=152 results.

**Impact**:
- ❌ K-shot ablation table is invalid (all rows are identical)
- ❌ Cannot claim performance saturation at k=10
- ❌ Cannot prove few-shot capability (k=5, 10)

**Fix Required**: Modify `run_kshot_ablation_multiepisode.py` to save results to k-shot-specific files:
- `multi_episode_results/exploits_100_episodes_k5.json`
- `multi_episode_results/exploits_100_episodes_k10.json`
- etc.

**Status**: 🔴 Must re-run k-shot ablation with fixed script

---

## Next Steps

### Step 1: Fix K-Shot Ablation Script (URGENT) ⚠️

The k-shot ablation must be re-run with proper result isolation per k-shot value.

**Required changes**:
1. Modify multi-episode evaluation to save to k-shot-specific files
2. Re-run all 6 k-shot experiments with fixed script
3. Verify results differ across k-shot values

### Step 2: Validate Base Model Prototype Refinement Hypothesis

**Immediate Action**: Run Experiment 1 to validate hypothesis

```bash
# Quick test (5 minutes)
# 1. Modify main.py line 3364 to use transductive_inference()
# 2. Run single evaluation
python main.py --mode evaluate_base_only

# Expected output:
# Base ZDR (with refinement): 70-80% (currently 57.96%)
```

If this confirms the hypothesis, proceed with Experiment 2 (prototype-only TTT).

### Step 3: Run Prototype-Only TTT Experiment

After validating Step 2, implement and test prototype-only TTT adaptation.

---

**Status**: 🔴 Critical issues identified | **Action Required**: Fix k-shot ablation + Run validation experiments
**Impact**: High - affects publication narrative and addresses reviewer concerns
**Effort**: Medium - requires re-running k-shot ablation (~40 min) + validation experiments (~2 hours)
