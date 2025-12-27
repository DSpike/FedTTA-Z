# Low-Confidence-Only TTT: Radical Change Explained

## 🎯 **What Is "Low-Confidence-Only Adaptation"?**

### **Current TTT Approach (What You're Doing Now):**

```
Test Set (100 samples):
├─ 70 samples: Non-zero-day (known attacks, normal) ← Base model is CONFIDENT
└─ 30 samples: Zero-day (unseen attacks) ← Base model is UNCERTAIN

Current TTT:
→ Adapts on ALL 100 samples
→ 70% of gradient comes from confident samples
→ Only 30% from uncertain (zero-day) samples
→ Result: TTT optimizes for majority, not zero-day
```

**Problem:** TTT spends most effort on samples the base model already handles well!

---

### **Low-Confidence-Only TTT (Radical Change):**

```
Test Set (100 samples):
├─ 70 samples: Non-zero-day (known attacks, normal) ← Base model is CONFIDENT
└─ 30 samples: Zero-day (unseen attacks) ← Base model is UNCERTAIN

Low-Confidence-Only TTT:
→ Step 1: Run base model on ALL samples
→ Step 2: Identify LOW-CONFIDENCE samples (uncertain predictions)
→ Step 3: Adapt ONLY on low-confidence samples (maybe 20-40 samples)
→ Result: TTT focuses 100% on samples that need help (zero-day)
```

**Key Insight:** Zero-day samples are likely to have **low confidence** because the model hasn't seen them before!

---

## 🔍 **How to Identify Low-Confidence Samples**

### **Method 1: Entropy-Based (Recommended)**

**Concept:** High entropy = Low confidence = Uncertain prediction

```python
# Step 1: Get base model predictions on test set
with torch.no_grad():
    logits = base_model(test_samples)
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

# Step 2: Identify low-confidence samples (high entropy)
# High entropy = model is uncertain = likely zero-day
entropy_threshold = entropy.quantile(0.7)  # Top 30% most uncertain
low_confidence_mask = entropy > entropy_threshold

# Step 3: Adapt ONLY on low-confidence samples
low_confidence_samples = test_samples[low_confidence_mask]
ttt_model = adapt_to_test_data(low_confidence_samples)  # Only these!
```

**Why This Works:**
- Zero-day samples → High entropy (model uncertain)
- Known samples → Low entropy (model confident)
- **Focus adaptation on uncertain samples = focus on zero-day**

---

### **Method 2: Probability-Based**

**Concept:** Low max probability = Low confidence

```python
# Step 1: Get base model predictions
with torch.no_grad():
    logits = base_model(test_samples)
    probs = F.softmax(logits, dim=1)
    max_probs = probs.max(dim=1)[0]  # Highest class probability

# Step 2: Identify low-confidence samples (low max probability)
# Low max_prob = model is uncertain = likely zero-day
confidence_threshold = max_probs.quantile(0.3)  # Bottom 30% least confident
low_confidence_mask = max_probs < confidence_threshold

# Step 3: Adapt ONLY on low-confidence samples
low_confidence_samples = test_samples[low_confidence_mask]
ttt_model = adapt_to_test_data(low_confidence_samples)
```

---

### **Method 3: Distance-Based (For Prototype Models)**

**Concept:** Far from prototypes = Low confidence = Likely zero-day

```python
# Step 1: Get embeddings and compute distances to prototypes
with torch.no_grad():
    embeddings = base_model.get_embeddings(test_samples)
    distances = compute_distances_to_prototypes(embeddings, prototypes)
    min_distances = distances.min(dim=1)[0]  # Distance to nearest prototype

# Step 2: Identify low-confidence samples (far from prototypes)
# Far from prototypes = model is uncertain = likely zero-day
distance_threshold = min_distances.quantile(0.7)  # Top 30% farthest
low_confidence_mask = min_distances > distance_threshold

# Step 3: Adapt ONLY on low-confidence samples
low_confidence_samples = test_samples[low_confidence_mask]
ttt_model = adapt_to_test_data(low_confidence_samples)
```

---

## 🎯 **Why This Might Work**

### **1. Focuses on What Needs Help**

**Current TTT:**
- Adapts on ALL samples
- 70% effort on samples base model already handles well
- 30% effort on samples that need help

**Low-Confidence-Only:**
- Adapts ONLY on uncertain samples
- 100% effort on samples that need help
- **Zero-day samples are likely in this set!**

---

### **2. Zero-Day Samples Are Naturally Low-Confidence**

**Why:**
- Model hasn't seen zero-day attacks during training
- Model is **uncertain** about zero-day samples
- High entropy = Low confidence = Likely zero-day

**Result:** Low-confidence set will contain most zero-day samples!

---

### **3. Avoids Overfitting to Known Samples**

**Current TTT:**
- Adapts on known samples (70%)
- May overfit to known patterns
- **Hurt zero-day detection**

**Low-Confidence-Only:**
- Doesn't adapt on known samples
- Focuses on uncertain/zero-day samples
- **Better zero-day detection**

---

## 🔧 **Implementation Plan**

### **Step 1: Identify Low-Confidence Samples**

Add this function to `main.py` or `coordinators/centralized_coordinator.py`:

```python
def identify_low_confidence_samples(self, model, test_samples, threshold_percentile=0.7):
    """
    Identify low-confidence samples (high entropy = uncertain predictions)
    
    Args:
        model: Base model
        test_samples: Test data tensor
        threshold_percentile: Percentile for threshold (0.7 = top 30% most uncertain)
    
    Returns:
        low_confidence_mask: Boolean mask for low-confidence samples
        low_confidence_indices: Indices of low-confidence samples
    """
    model.eval()
    with torch.no_grad():
        # Get predictions
        if hasattr(model, 'forward_with_prototypes'):
            # Prototype-based model
            logits = model.forward_with_prototypes(test_samples, prototypes)
        else:
            outputs = model(test_samples)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Compute entropy (high entropy = low confidence)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        
        # Identify low-confidence samples (high entropy)
        threshold = torch.quantile(entropy, threshold_percentile)
        low_confidence_mask = entropy > threshold
        
        logger.info(f"🔍 Low-confidence identification:")
        logger.info(f"   Total samples: {len(test_samples)}")
        logger.info(f"   Low-confidence samples: {low_confidence_mask.sum().item()} ({100*low_confidence_mask.sum().item()/len(test_samples):.1f}%)")
        logger.info(f"   Entropy threshold: {threshold.item():.4f}")
        logger.info(f"   Mean entropy (low-conf): {entropy[low_confidence_mask].mean().item():.4f}")
        logger.info(f"   Mean entropy (high-conf): {entropy[~low_confidence_mask].mean().item():.4f}")
        
        return low_confidence_mask, torch.where(low_confidence_mask)[0]
```

---

### **Step 2: Modify TTT Adaptation**

Modify `perform_coordinator_side_ttt_adaptation()` in `main.py`:

```python
def perform_coordinator_side_ttt_adaptation(self) -> torch.nn.Module:
    # ... existing code ...
    
    # NEW: Identify low-confidence samples BEFORE adaptation
    use_low_confidence_only = getattr(self.config, 'ttt_low_confidence_only', False)
    
    if use_low_confidence_only:
        logger.info("🎯 LOW-CONFIDENCE-ONLY MODE: Identifying uncertain samples...")
        
        # Get base model predictions
        low_confidence_mask, low_confidence_indices = self.identify_low_confidence_samples(
            self.coordinator.model,
            X_test,
            threshold_percentile=0.7  # Top 30% most uncertain
        )
        
        # Filter to ONLY low-confidence samples
        X_test_low_conf = X_test[low_confidence_mask]
        logger.info(f"   Adapting on {len(X_test_low_conf)} low-confidence samples (was {len(X_test)} total)")
        
        # Verify zero-day samples are in low-confidence set
        if 'y_test_multiclass' in self.preprocessed_data:
            y_test_multiclass = self.preprocessed_data['y_test_multiclass']
            zero_day_mask = (y_test_multiclass == self.config.zero_day_attack_label)
            zero_day_in_low_conf = (zero_day_mask & low_confidence_mask).sum().item()
            zero_day_total = zero_day_mask.sum().item()
            logger.info(f"   Zero-day samples in low-confidence set: {zero_day_in_low_conf}/{zero_day_total} ({100*zero_day_in_low_conf/zero_day_total:.1f}%)")
        
        # Use low-confidence samples for adaptation
        query_x = X_test_low_conf
    else:
        # Standard: Use all samples
        query_x = X_test
    
    # Continue with TTT adaptation on query_x
    adapted_model = self.coordinator.adapt_to_test_data(
        query_x=query_x,
        # ... rest of parameters ...
    )
```

---

### **Step 3: Add Configuration**

Add to `config_loader.py`:

```python
'ttt_low_confidence_only': True,  # NEW: Adapt only on low-confidence samples (radical change)
'ttt_low_confidence_percentile': 0.7,  # Top 30% most uncertain samples
```

---

## 📊 **Expected Results**

### **Current TTT:**
- Adapts on: 100 samples (70% non-zero-day, 30% zero-day)
- Zero-day ZDR: 55-60%
- **Problem:** Optimizes for majority

### **Low-Confidence-Only TTT:**
- Adapts on: 30-40 samples (mostly zero-day!)
- Zero-day ZDR: **70-80%** (expected improvement)
- **Benefit:** Focuses on what needs help

---

## ⚠️ **Potential Issues**

### **1. May Miss Some Zero-Day Samples**
- If some zero-day samples have high confidence (rare)
- **Solution:** Use higher percentile (0.8 or 0.9)

### **2. May Include Some Non-Zero-Day Samples**
- Some known samples might be low-confidence (hard cases)
- **Solution:** This is OK - adapting on hard cases helps too

### **3. Smaller Adaptation Set**
- Only 30-40 samples instead of 100
- **Solution:** May need more TTT steps or higher learning rate

---

## 🎯 **Why This Is "Radical"**

**Current Approach:**
- Adapt on ALL samples
- Optimize for overall distribution
- **Conservative, standard TTT**

**Low-Confidence-Only:**
- Adapt ONLY on uncertain samples
- Optimize for outliers (zero-day)
- **Aggressive, targeted TTT**

**This is fundamentally different** - it's not just a hyperparameter change, it's a **different adaptation strategy**!

---

## 💡 **Bottom Line**

**Low-Confidence-Only TTT:**
1. Identifies samples where base model is **uncertain** (high entropy)
2. Adapts **ONLY** on those uncertain samples
3. **Focuses 100% on samples that need help** (likely zero-day)
4. **Avoids wasting effort** on samples base model already handles well

**Expected Improvement:**
- Current TTT ZDR: 55-60%
- Low-Confidence-Only TTT ZDR: **70-80%** (if it works)

**Time to Implement:** 2-3 days

**Risk:** Medium (might work, might not - but worth trying as last attempt)

---

**This is the "radical change" - completely different adaptation strategy that focuses on zero-day samples instead of all samples!**



