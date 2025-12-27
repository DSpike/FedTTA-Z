# Zero-Day Entropy Scatter Plot: Interpretation Guide

## 📊 **What is This Plot?**

The **Zero-Day Entropy Scatter Plot** visualizes the **prediction uncertainty (entropy)** for each zero-day attack sample after TTT adaptation. It helps you understand:

1. **Which zero-day samples are easy/hard to detect**
2. **How confident the model is for each zero-day sample**
3. **Distribution of uncertainty across zero-day attacks**
4. **Effectiveness of TTT adaptation on zero-day samples**

---

## 🎯 **Plot Components**

### **1. X-Axis: Zero-Day Sample Index**
- **What it shows**: Each point represents one zero-day attack sample
- **Range**: 0 to N (where N = number of zero-day samples in test set)
- **Example**: If you have 50 zero-day samples, x-axis goes from 0 to 49

**Interpretation**:
- Each point is a **different zero-day sample**
- Samples are ordered by their **index in the test set** (not by difficulty)

---

### **2. Y-Axis: Entropy Loss**
- **What it shows**: Prediction uncertainty for each sample
- **Range**: Typically 0.0 to ~2.0 (depends on number of classes)
- **Units**: Entropy (bits of uncertainty)

**Mathematical Definition**:
```
Entropy = -Σ(p_i * log(p_i))
```
Where `p_i` is the probability of class `i`

**Interpretation**:
- **Low Entropy (0.0 - 0.5)**: Model is **confident** (high certainty)
  - ✅ Good: Model knows what it's predicting
  - ✅ Likely correct classification
- **Medium Entropy (0.5 - 1.5)**: Model is **moderately uncertain**
  - ⚠️ Moderate: Some uncertainty, but still reasonable
  - ⚠️ May be correct or incorrect
- **High Entropy (1.5 - 2.0+)**: Model is **very uncertain**
  - ❌ Bad: Model doesn't know what to predict
  - ❌ Likely incorrect classification or confusion

---

### **3. Color Coding (Viridis Colormap)**
- **What it shows**: Same as Y-axis (entropy value), but visually encoded
- **Color Scale**: 
  - **Dark Purple/Blue**: Low entropy (confident predictions)
  - **Green/Yellow**: Medium entropy (moderate uncertainty)
  - **Bright Yellow**: High entropy (very uncertain)

**Interpretation**:
- **Quick visual scan**: Darker points = confident, Brighter points = uncertain
- **Pattern recognition**: Clusters of similar colors indicate groups of samples with similar uncertainty

---

### **4. Statistics Box**
Shows key statistics for all zero-day samples:

| Statistic | Meaning | Interpretation |
|-----------|---------|----------------|
| **Mean** | Average entropy across all zero-day samples | Lower = better overall confidence |
| **Median** | Middle value (50th percentile) | Less affected by outliers than mean |
| **Std** | Standard deviation (spread) | Higher = more variation in confidence |
| **Min** | Lowest entropy (most confident sample) | Best-case scenario |
| **Max** | Highest entropy (least confident sample) | Worst-case scenario |
| **Samples** | Total number of zero-day samples | Sample size for evaluation |

**Interpretation**:
- **Mean < 0.5**: ✅ Good - Model is generally confident on zero-day samples
- **Mean > 1.0**: ❌ Bad - Model is uncertain on zero-day samples
- **Std > 0.5**: ⚠️ High variation - Some samples easy, some hard
- **Std < 0.2**: ✅ Low variation - Consistent confidence across samples

---

### **5. Mean Line (Red Dashed)**
- **What it shows**: Horizontal line at the mean entropy value
- **Purpose**: Visual reference for average confidence

**Interpretation**:
- **Points above line**: Above-average uncertainty (harder samples)
- **Points below line**: Below-average uncertainty (easier samples)
- **Distance from line**: How much each sample deviates from average

---

## 📈 **How to Interpret the Plot**

### **Scenario 1: Good Zero-Day Detection** ✅

```
Y-Axis (Entropy)
2.0 |                    ●
    |              ●
1.5 |        ●  ●
    |    ●  ●  ●  ●
1.0 |  ●  ●  ●  ●  ●
    |●  ●  ●  ●  ●  ●
0.5 |●  ●  ●  ●  ●  ●  ●
    |●  ●  ●  ●  ●  ●  ●  ●
0.0 |________________________
    0  10  20  30  40  50  X (Sample Index)
```

**Characteristics**:
- **Most points clustered at bottom** (low entropy)
- **Mean < 0.5**
- **Few outliers** (high entropy points)
- **Low standard deviation**

**Interpretation**:
- ✅ Model is **confident** on most zero-day samples
- ✅ TTT adaptation **worked well** for zero-day detection
- ✅ Zero-day attacks are **detectable** with high confidence

---

### **Scenario 2: Poor Zero-Day Detection** ❌

```
Y-Axis (Entropy)
2.0 |●  ●  ●  ●  ●  ●  ●  ●
    |●  ●  ●  ●  ●  ●  ●  ●
1.5 |●  ●  ●  ●  ●  ●  ●  ●
    |●  ●  ●  ●  ●  ●  ●  ●
1.0 |●  ●  ●  ●  ●  ●  ●  ●
    |●  ●  ●  ●  ●  ●  ●  ●
0.5 |●  ●  ●  ●  ●  ●  ●  ●
    |●  ●  ●  ●  ●  ●  ●  ●
0.0 |________________________
    0  10  20  30  40  50  X (Sample Index)
```

**Characteristics**:
- **Most points at top** (high entropy)
- **Mean > 1.0**
- **High standard deviation**
- **Wide spread** of values

**Interpretation**:
- ❌ Model is **uncertain** on zero-day samples
- ❌ TTT adaptation **didn't help** zero-day detection
- ❌ Zero-day attacks are **hard to detect** or **confused with other classes**

---

### **Scenario 3: Mixed Performance** ⚠️

```
Y-Axis (Entropy)
2.0 |        ●              ●
    |    ●              ●
1.5 |  ●  ●          ●  ●
    |●  ●  ●      ●  ●  ●
1.0 |●  ●  ●  ●  ●  ●  ●  ●
    |●  ●  ●  ●  ●  ●  ●  ●
0.5 |●  ●  ●  ●  ●  ●  ●  ●
    |●  ●  ●  ●  ●  ●  ●  ●
0.0 |________________________
    0  10  20  30  40  50  X (Sample Index)
```

**Characteristics**:
- **Bimodal distribution** (two clusters)
- **Medium mean** (~0.7-1.0)
- **High standard deviation**
- **Clear separation** between easy and hard samples

**Interpretation**:
- ⚠️ Some zero-day samples are **easy** (low entropy)
- ⚠️ Some zero-day samples are **hard** (high entropy)
- ⚠️ TTT adaptation **partially worked**
- ⚠️ Need to investigate **why some samples are harder**

---

## 🔍 **Key Questions to Answer**

### **1. Is TTT Adaptation Working for Zero-Day?**

**Check**: Compare entropy distribution
- **Before TTT**: High entropy (uncertain)
- **After TTT**: Lower entropy (more confident) ✅

**If entropy is still high after TTT**:
- ❌ TTT adaptation **not effective** for zero-day
- ❌ May need **zero-day weighted TTT** (already implemented)
- ❌ Zero-day samples may be **too different** from training data

---

### **2. Are Zero-Day Samples Detectable?**

**Check**: Mean entropy value
- **Mean < 0.5**: ✅ **Highly detectable** (confident predictions)
- **Mean 0.5-1.0**: ⚠️ **Moderately detectable** (some uncertainty)
- **Mean > 1.0**: ❌ **Poorly detectable** (high uncertainty)

**Also check**: Number of samples with entropy < 0.5
- **> 70%**: ✅ Good detection rate expected
- **50-70%**: ⚠️ Moderate detection rate
- **< 50%**: ❌ Poor detection rate

---

### **3. Which Zero-Day Samples Are Hardest?**

**Check**: Points at the top of the plot (high entropy)
- **Identify**: Sample indices with entropy > 1.5
- **Investigate**: Why these specific samples are hard
  - Feature distribution?
  - Similarity to normal traffic?
  - Attack characteristics?

**Action**: Focus TTT adaptation on these hard samples

---

### **4. Is There a Pattern in Hard Samples?**

**Check**: Distribution of high-entropy points
- **Clustered**: Specific group of samples is hard (pattern exists)
- **Scattered**: Random hard samples (no clear pattern)
- **All high**: All samples are hard (systematic issue)

**Interpretation**:
- **Clustered**: May indicate **specific attack variant** or **feature pattern**
- **Scattered**: Random difficulty (normal variation)
- **All high**: **Fundamental problem** (model can't detect zero-day)

---

## 📊 **Practical Examples**

### **Example 1: PortScan Zero-Day (Good Performance)**

```
Statistics:
Mean: 0.32
Median: 0.28
Std: 0.15
Min: 0.12
Max: 0.78
Samples: 45
```

**Interpretation**:
- ✅ **Mean 0.32**: Very confident (low entropy)
- ✅ **Std 0.15**: Low variation (consistent confidence)
- ✅ **Max 0.78**: Even worst sample is reasonably confident
- ✅ **Conclusion**: PortScan is **well-detected** as zero-day

---

### **Example 2: DoS Hulk Zero-Day (Poor Performance)**

```
Statistics:
Mean: 1.45
Median: 1.38
Std: 0.42
Min: 0.65
Max: 2.10
Samples: 38
```

**Interpretation**:
- ❌ **Mean 1.45**: Very uncertain (high entropy)
- ⚠️ **Std 0.42**: High variation (some easy, some hard)
- ❌ **Max 2.10**: Some samples are completely uncertain
- ❌ **Conclusion**: DoS Hulk is **poorly detected** as zero-day

**Possible Reasons**:
- DoS Hulk similar to other DoS variants (confusion)
- Feature patterns overlap with known attacks
- TTT adaptation optimized for non-zero-day samples (70% majority)

---

### **Example 3: WebAttack Zero-Day (Mixed Performance)**

```
Statistics:
Mean: 0.85
Median: 0.72
Std: 0.58
Min: 0.15
Max: 1.95
Samples: 25
```

**Interpretation**:
- ⚠️ **Mean 0.85**: Moderate uncertainty
- ⚠️ **Std 0.58**: High variation (bimodal distribution)
- ✅ **Min 0.15**: Some samples very confident
- ❌ **Max 1.95**: Some samples very uncertain
- ⚠️ **Conclusion**: WebAttack has **mixed detectability**

**Analysis**:
- **Easy samples** (entropy < 0.5): ~40% of samples
- **Hard samples** (entropy > 1.2): ~30% of samples
- **Action**: Investigate why some WebAttack samples are harder

---

## 🎯 **Actionable Insights**

### **If Mean Entropy is High (> 1.0)**:

1. **Check TTT Weighting**:
   - Verify `ttt_zero_day_weight` is set (should be 3.0)
   - Zero-day samples should have 3x weight in loss

2. **Check Category Grouping**:
   - If using grouping, zero-day might be confused with known attacks
   - Try disabling grouping (`use_category_grouping: False`)

3. **Check Feature Distribution**:
   - Zero-day samples may have different feature patterns
   - Compare zero-day vs known attack features

---

### **If Standard Deviation is High (> 0.5)**:

1. **Identify Hard Samples**:
   - Find samples with entropy > 1.5
   - Analyze their characteristics

2. **Investigate Patterns**:
   - Are hard samples from specific time periods?
   - Do they have specific feature values?
   - Are they similar to normal traffic?

3. **Focus TTT Adaptation**:
   - Increase zero-day weight for hard samples
   - Use two-stage TTT (overall + zero-day specific)

---

### **If Most Samples Have Low Entropy (< 0.5)**:

1. **Verify Detection**:
   - Check zero-day detection rate (should be high)
   - Verify predictions are correct

2. **Check for Overconfidence**:
   - Low entropy might indicate overconfidence
   - Verify false positive rate

3. **Document Success**:
   - TTT adaptation is working well
   - Zero-day attacks are detectable

---

## 📝 **Summary: Quick Interpretation Guide**

| Mean Entropy | Std Dev | Interpretation | Action |
|-------------|---------|----------------|--------|
| < 0.5 | < 0.2 | ✅ Excellent - Highly confident | Verify detection rate |
| < 0.5 | > 0.3 | ✅ Good - Mostly confident | Check outliers |
| 0.5-1.0 | < 0.3 | ⚠️ Moderate - Some uncertainty | Investigate high-entropy samples |
| 0.5-1.0 | > 0.5 | ⚠️ Mixed - Bimodal distribution | Focus on hard samples |
| > 1.0 | < 0.3 | ❌ Poor - Consistently uncertain | Check TTT weighting |
| > 1.0 | > 0.5 | ❌ Very Poor - High uncertainty + variation | Major issue - review approach |

---

## 🔗 **Related Metrics**

Compare scatter plot with:

1. **Zero-Day Detection Rate (ZDR)**: Should correlate with low entropy
2. **False Alarm Rate (FAR)**: High entropy might indicate false positives
3. **TTT Adaptation Loss**: Should decrease during adaptation
4. **Base Model Performance**: Compare before/after TTT

---

**Document Created**: Guide for interpreting zero-day entropy scatter plot  
**Purpose**: Help understand TTT adaptation effectiveness on zero-day samples  
**Status**: Ready for use



