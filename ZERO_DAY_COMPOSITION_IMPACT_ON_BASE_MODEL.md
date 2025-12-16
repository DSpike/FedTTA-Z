# Impact of Lowering Zero-Day Composition on Base Model Performance

## ⚠️ **CRITICAL CLARIFICATION**

**Yes, lowering zero-day composition will INCREASE the reported accuracy, BUT:**

- ❌ It does **NOT improve the model's actual performance**
- ❌ It only makes the **test set easier** (fewer hard samples)
- ❌ This is **methodologically unsound** for zero-day detection research

---

## 📊 **What Will Happen**

### **Current Test Set (25% Zero-Day):**

```
Test Set Composition:
├── 40% Normal (easy - model seen these)
├── 35% Known attacks (moderate - model seen these)
└── 25% Zero-day (hard - model NEVER seen these)

Base Model Performance: 60.11% accuracy
```

### **If You Lower to 10% Zero-Day:**

```
Test Set Composition:
├── 40% Normal (easy - model seen these)
├── 50% Known attacks (moderate - model seen these)
└── 10% Zero-day (hard - model NEVER seen these)

Expected Base Model Performance: ~70-75% accuracy ⬆️
```

**Yes, the number will go up!** But why?

---

## 🔍 **Why Accuracy Will Increase**

### **The Math:**

**Current (25% zero-day):**

```
Performance breakdown:
- Normal (40%): 90% accuracy → contributes 0.40 × 0.90 = 0.36
- Known Attacks (35%): 85% accuracy → contributes 0.35 × 0.85 = 0.2975
- Zero-Day (25%): 50% accuracy → contributes 0.25 × 0.50 = 0.125

Overall = 0.36 + 0.2975 + 0.125 = 0.7825 ≈ 78% theoretical
Actual: 60.11% (prototype-based evaluation is harder)
```

**After lowering to 10% zero-day:**

```
Performance breakdown:
- Normal (40%): 90% accuracy → contributes 0.40 × 0.90 = 0.36
- Known Attacks (50%): 85% accuracy → contributes 0.50 × 0.85 = 0.425
- Zero-Day (10%): 50% accuracy → contributes 0.10 × 0.50 = 0.05

Overall = 0.36 + 0.425 + 0.05 = 0.835 ≈ 83% theoretical
Expected Actual: ~70-75% (easier test set)
```

**Accuracy increases because:**

- ✅ Fewer "hard" samples (zero-day)
- ✅ More "easy" samples (known attacks)
- ✅ Test set becomes **easier**, not because model improved

---

## ⚠️ **Why This Is Problematic**

### **1. It's Not Actually Improving Your Model**

**What's happening:**

- ✅ Reported accuracy: 60% → 75% (looks better!)
- ❌ Model performance: **Stays exactly the same**
- ❌ You're just testing on an **easier dataset**

**Real-world analogy:**

```
Student takes exam:
- Hard exam (25% hard questions): Score = 60%
- Easy exam (10% hard questions): Score = 75%

Is the student smarter? NO - the exam is just easier!
```

---

### **2. Misleading for Zero-Day Detection Research**

**Your research goal:**

- Detect **zero-day attacks** (unseen attacks)
- Test model's ability to handle **novel threats**

**Problem with lowering zero-day:**

- ❌ You're testing **less** on zero-day attacks
- ❌ Can't properly evaluate zero-day detection capability
- ❌ Results become **misleading** and **not comparable** to other papers

**Academic standard:**

- Zero-day detection papers typically use **20-35% zero-day** in test set
- Lower proportions are considered **insufficient** for evaluation
- Your current 25% is **within standard range** ✅

---

### **3. Statistical Reliability Decreases**

**With 25% zero-day:**

- Zero-day samples: ~188 out of 752 sequences
- Statistically reliable metrics
- Can calculate meaningful zero-day detection rate

**With 10% zero-day:**

- Zero-day samples: ~75 out of 752 sequences
- Less reliable metrics (higher variance)
- Zero-day detection rate becomes noisy
- Not statistically meaningful

---

## 📊 **Detailed Comparison**

### **Scenario A: Keep 25% Zero-Day (Current)**

| Aspect                       | Impact                       | Rating                      |
| ---------------------------- | ---------------------------- | --------------------------- |
| **Overall Accuracy**         | 60.11% (realistic)           | ✅ Good                     |
| **Zero-Day Detection Rate**  | Reliable with 188 samples    | ✅ Good                     |
| **Statistical Significance** | Sufficient samples           | ✅ Good                     |
| **Research Validity**        | Standard practice            | ✅ Good                     |
| **Model Improvement**        | Requires actual improvements | ✅ Encourages better models |

### **Scenario B: Lower to 10% Zero-Day**

| Aspect                       | Impact                     | Rating                   |
| ---------------------------- | -------------------------- | ------------------------ |
| **Overall Accuracy**         | ~70-75% (misleading)       | ⚠️ Looks better but fake |
| **Zero-Day Detection Rate**  | Less reliable (75 samples) | ❌ Poor                  |
| **Statistical Significance** | Insufficient samples       | ❌ Poor                  |
| **Research Validity**        | Non-standard               | ❌ Poor                  |
| **Model Improvement**        | None (just easier test)    | ❌ No improvement        |

---

## 🎯 **What Actually Improves Base Model Performance**

### **Things That WILL Improve Base Model (60% → higher):**

1. **Better Hyperparameters** ✅

   - Your optimized config should help
   - More rounds/clients improve aggregation

2. **More Training Data** ✅

   - More diverse attack samples
   - Better feature learning

3. **Better Architecture** ✅

   - Deeper networks
   - Better feature extraction

4. **Better Training Strategy** ✅

   - More federated rounds (15 is good)
   - Better aggregation (FedProx is good)

5. **Better Meta-Learning** ✅
   - More meta-tasks
   - Better support set composition

### **Things That DON'T Improve Base Model:**

1. **Changing Test Set Composition** ❌

   - Just makes test easier
   - Doesn't improve model

2. **Removing Hard Samples** ❌
   - Misleading results
   - Not scientifically valid

---

## 📈 **Expected Performance with Different Zero-Day Proportions**

### **Estimated Base Model Performance:**

| Zero-Day %        | Overall Accuracy | Zero-Day Detection Rate | Notes                      |
| ----------------- | ---------------- | ----------------------- | -------------------------- |
| **25% (Current)** | **60.11%**       | ~50-60%                 | ✅ Standard, realistic     |
| **20%**           | ~65-70%          | ~50-60%                 | ⚠️ Slightly easier         |
| **15%**           | ~68-73%          | ~50-60%                 | ⚠️ Easier test             |
| **10%**           | ~70-75%          | ~50-60%                 | ❌ Too easy, less reliable |
| **5%**            | ~73-78%          | ~50-60%                 | ❌ Insufficient zero-day   |

**Key Point:** As you lower zero-day %, overall accuracy increases, but:

- ❌ Model capability stays the same
- ❌ Zero-day detection rate stays similar (~50-60%)
- ❌ Statistical reliability decreases

---

## ✅ **Recommendation: Keep 25% Zero-Day**

### **Why 25% is Good:**

1. **✅ Realistic Scenario**

   - Represents real-world zero-day threat level
   - Tests actual zero-day detection capability

2. **✅ Statistically Reliable**

   - Enough samples for reliable metrics
   - Can calculate meaningful detection rates

3. **✅ Standard Practice**

   - Aligns with academic zero-day detection papers
   - Makes results comparable to other research

4. **✅ Encourages Real Improvements**

   - Can't fake good performance by manipulating test set
   - Must actually improve the model to get better results

5. **✅ Research Integrity**
   - Scientifically valid evaluation
   - Honest assessment of model capability

---

## 🎯 **If You Want Better Performance, Improve the Model**

### **Ways to Actually Improve Base Model from 60% → Higher:**

1. **✅ More Training Rounds**

   - Current: 15 rounds (optimized)
   - Could try: 20 rounds (but diminishing returns)

2. **✅ More Clients**

   - Current: 5 clients (optimized)
   - Could try: 7-10 clients (but more computation)

3. **✅ Better Support Set Composition**

   - Already optimized with equal distribution
   - Could experiment with different ratios

4. **✅ Better Hyperparameters**

   - Already using optimized values
   - Could run more optimization trials

5. **✅ More Training Data**

   - Add more diverse attack samples
   - Better feature representation

6. **✅ Better Architecture**
   - Deeper networks
   - Better feature extraction layers

**Don't manipulate the test set - improve the model!** ✅

---

## 📋 **Summary**

### **Q: If I lower zero-day composition in test set, will it increase base model performance?**

### **A: YES, but it's MISLEADING**

**What Will Happen:**

- ✅ Overall accuracy will increase (60% → ~70-75%)
- ❌ Model performance stays the same (no actual improvement)
- ❌ Zero-day detection becomes less reliable (fewer samples)
- ❌ Results become misleading (easier test, not better model)

**What You Should Do:**

- ✅ **Keep 25% zero-day** (realistic and reliable)
- ✅ **Focus on improving the model** (hyperparameters, architecture, training)
- ✅ **Use TTT adaptation** (already improves 60% → 77%)
- ❌ **Don't manipulate test set** (unethical and misleading)

**Bottom Line:** Lowering zero-day will make the numbers look better, but it's not actually improving your model. Keep 25% and focus on real improvements! ✅



