# L2 Regularization Effects on Base Model and TTT Model Performance

## 🎯 **Overview**

The `ttt_l2_reg_weight` parameter controls how much the TTT-adapted model is constrained to stay close to the original (base) model parameters. This analysis explains how different values affect performance.

---

## 📊 **Key Point: Base Model is NOT Affected**

**Important**: The L2 regularization only applies during **TTT adaptation**. The base model performance remains **unchanged** regardless of `ttt_l2_reg_weight` value.

- Base model is trained before TTT
- Base model evaluation doesn't use TTT
- L2 regularization only affects the **TTT adaptation process**

---

## 🔬 **Effects of Different Regularization Values**

### **1. Very Low Regularization (0.0 - 0.001)**

#### **Behavior**:

- Almost no constraint on parameter changes
- Model can adapt freely to test data
- Parameters can drift significantly from original

#### **TTT Model Performance**:

- ✅ **High adaptation flexibility** - can fit test distribution well
- ❌ **Risk of overfitting** - may over-adapt to test data
- ❌ **Catastrophic forgetting** - may forget original learned features
- ❌ **Unstable** - performance can vary wildly across different test sets
- ❌ **Poor generalization** - works on current test set, fails on new data

#### **Expected Performance**:

- On current test set: **Potentially high** (if test set matches adaptation data)
- On new test sets: **Potentially low** (overfitting)
- Zero-day detection: **Unpredictable** (may help or hurt)

#### **When to Use**:

- Test set distribution is very different from training
- Need maximum adaptation flexibility
- Willing to risk overfitting

---

### **2. Low Regularization (0.001 - 0.005)**

#### **Behavior**:

- Minimal constraint on parameter changes
- Model can still adapt significantly
- Some protection against extreme drift

#### **TTT Model Performance**:

- ✅ Good adaptation flexibility
- ✅ Moderate protection against overfitting
- ⚠️ Still some risk of overfitting
- ⚠️ Performance can be inconsistent

#### **Expected Performance**:

- On current test set: **Good to High**
- On new test sets: **Moderate**
- Zero-day detection: **Variable**

#### **When to Use**:

- Test set is somewhat different from training
- Want adaptation but with some stability
- Balanced approach

---

### **3. Default/Moderate Regularization (0.01)**

#### **Behavior**:

- Balanced constraint on parameter changes
- Model can adapt but stays close to original
- Good protection against overfitting

#### **TTT Model Performance**:

- ✅ Good adaptation capability
- ✅ Strong protection against overfitting
- ✅ Stable performance across test sets
- ✅ Better generalization
- ✅ Expected **+2-4% improvement** over unregularized

#### **Expected Performance**:

- On current test set: **Good** (may be slightly lower than unregularized)
- On new test sets: **Better** (more consistent)
- Zero-day detection: **Better** (more reliable)

#### **When to Use**:

- **Recommended default**
- General use case
- Want stable, generalizable improvements

---

### **4. High Regularization (0.05 - 0.1)**

#### **Behavior**:

- Strong constraint on parameter changes
- Model adapts very conservatively
- Stays very close to original parameters

#### **TTT Model Performance**:

- ✅ Very stable
- ✅ Strong protection against overfitting
- ✅ Excellent generalization
- ❌ Limited adaptation capability
- ❌ May not benefit much from TTT
- ❌ TTT improvements may be minimal

#### **Expected Performance**:

- On current test set: **Moderate** (similar to base model)
- On new test sets: **Good** (consistent with base)
- Zero-day detection: **Similar to base** (little improvement)

#### **When to Use**:

- Very similar test/train distributions
- Prioritize stability over adaptation
- Base model is already very good

---

### **5. Very High Regularization (0.1 - 1.0)**

#### **Behavior**:

- Extremely strong constraint
- Parameters barely change
- Model stays almost identical to base

#### **TTT Model Performance**:

- ✅ Maximum stability
- ✅ No overfitting risk
- ❌ Essentially no adaptation
- ❌ TTT becomes ineffective
- ❌ Performance ≈ base model

#### **Expected Performance**:

- On current test set: **Similar to base**
- On new test sets: **Similar to base**
- Zero-day detection: **No improvement**

#### **When to Use**:

- Don't want any adaptation
- Only using TTT for stability
- Base model is optimal

---

## 📈 **Performance Trade-off Curve**

```
TTT Performance vs Regularization Weight:

High Performance
    │
    │     ┌─────────────────────┐
    │    ╱                      │
    │   ╱        Optimal        │
    │  ╱        Range           │
    │ ╱                         │
    │╱                          │
    ├────────────────────────────→
   0.0     0.01    0.05    0.1+  Regularization Weight

Too Low          Default    Too High
(Overfitting)    (Balanced)  (No Adaptation)
```

---

## 🎯 **Specific Effects on Metrics**

### **Accuracy**:

- **Low reg**: High on current test, low on new tests (overfitting)
- **Default (0.01)**: Good on current, better on new (balanced)
- **High reg**: Moderate on all (stable, less adaptation)

### **F1 Score**:

- **Low reg**: Can be high if test matches adaptation data
- **Default (0.01)**: Balanced, reliable improvements
- **High reg**: Similar to base model

### **Zero-Day Detection Rate (ZDR)**:

- **Low reg**: Unpredictable (may overfit to non-zero-day patterns)
- **Default (0.01)**: Better generalization, more reliable ZDR
- **High reg**: Similar to base (no adaptation benefit)

### **False Alarm Rate (FAR)**:

- **Low reg**: Can be high (overfitting to test distribution)
- **Default (0.01)**: Better controlled (regularized adaptation)
- **High reg**: Similar to base

---

## 💡 **Recommendations**

### **For Different Scenarios**:

#### **Scenario 1: Test Set Similar to Training**

```
Regularization: 0.05 - 0.1
Reason: Less adaptation needed, prioritize stability
```

#### **Scenario 2: Test Set Different from Training** (Typical)

```
Regularization: 0.01 (Default)
Reason: Balanced adaptation and stability
```

#### **Scenario 3: Zero-Day Detection Focus**

```
Regularization: 0.01 - 0.02
Reason: Need adaptation for unseen attacks, but avoid overfitting
```

#### **Scenario 4: Maximum Stability Required**

```
Regularization: 0.05 - 0.1
Reason: Production environment, consistency critical
```

---

## 🔍 **How to Choose the Right Value**

### **Step 1: Start with Default (0.01)**

- Test performance
- Monitor overfitting (compare train vs test)

### **Step 2: If Overfitting Detected**

- **Increase** regularization (0.02 → 0.05)
- Monitor: Does performance become more stable?
- Check: Does FAR increase too much?

### **Step 3: If Adaptation Too Weak**

- **Decrease** regularization (0.01 → 0.005)
- Monitor: Does performance improve?
- Check: Does it become unstable across test sets?

### **Step 4: Monitor Metrics**

- **Good signs**: Consistent improvements, stable FAR, better ZDR
- **Bad signs**: High variance, increasing FAR, inconsistent ZDR

---

## 📊 **Expected Performance Changes**

### **Current Default (0.01)**:

- Base Model: **Unaffected** (same performance)
- TTT Model: **+2-4% improvement** over base
- Stability: **Good** (consistent across test sets)

### **If Increased to 0.05**:

- Base Model: **Unaffected**
- TTT Model: **+1-2% improvement** (more conservative)
- Stability: **Very Good** (more consistent)

### **If Decreased to 0.005**:

- Base Model: **Unaffected**
- TTT Model: **+3-5% improvement** (more aggressive)
- Stability: **Moderate** (may vary more)

### **If Set to 0.0 (No Regularization)**:

- Base Model: **Unaffected**
- TTT Model: **+4-6% OR -2-5%** (unpredictable)
- Stability: **Poor** (high variance)

---

## ⚠️ **Warning Signs**

### **Regularization Too Low**:

- ✅ Large performance gains on current test set
- ❌ Large performance drops on different test sets
- ❌ High variance in results
- ❌ Increasing FAR over time
- ❌ TTT model performs worse than base on some metrics

### **Regularization Too High**:

- ✅ Very stable results
- ❌ TTT model performance ≈ base model
- ❌ No improvement from TTT
- ❌ Wasted computation (TTT not helping)

---

## 🎯 **Summary**

| Regularization     | Adaptation | Overfitting Risk | Stability | Best For             |
| ------------------ | ---------- | ---------------- | --------- | -------------------- |
| **0.0 - 0.001**    | Maximum    | Very High        | Poor      | Research only        |
| **0.001 - 0.005**  | High       | Moderate         | Moderate  | Specific domains     |
| **0.01 (Default)** | Balanced   | Low              | Good      | **General use** ⭐   |
| **0.05 - 0.1**     | Low        | Very Low         | Excellent | Production stability |
| **0.1+**           | Minimal    | None             | Maximum   | Stability only       |

---

## 💡 **Bottom Line**

1. **Base Model**: **Not affected** by regularization weight
2. **TTT Model**:
   - Lower reg → More adaptation, higher overfitting risk
   - Higher reg → Less adaptation, more stability
3. **Default (0.01)**: Good balance for most use cases
4. **Tune based on**: Test set similarity, overfitting signs, stability needs

**Recommended**: Start with `0.01`, adjust based on your specific dataset and requirements!



