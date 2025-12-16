# FAR (False Acceptance Rate) Analysis

## FAR Calculation

**FAR = False Acceptance Rate = False Positive Rate (FPR)**
- **Formula**: FAR = FP / (FP + TN)
- **Meaning**: Rate of normal samples incorrectly classified as attacks
- **Security Impact**: Lower FAR is better (fewer false alarms)

## Base Model FAR

**Confusion Matrix:**
```
                Predicted
              Normal  Attack
Actual Normal   94     14
       Attack   63    161
```

- **TN (True Negatives)**: 94 (normal correctly classified as normal)
- **FP (False Positives)**: 14 (normal incorrectly classified as attack)
- **FN (False Negatives)**: 63 (attack incorrectly classified as normal)
- **TP (True Positives)**: 161 (attack correctly classified as attack)

**FAR Calculation:**
- FAR = FP / (FP + TN) = 14 / (14 + 94) = 14 / 108 = **0.1296 = 12.96%**

**Interpretation:**
- 12.96% of normal samples are incorrectly flagged as attacks
- This means ~13 out of 100 normal samples trigger false alarms

## TTT Model FAR

**Confusion Matrix:**
```
                Predicted
              Normal  Attack
Actual Normal   85     23
       Attack   12    212
```

- **TN (True Negatives)**: 85 (normal correctly classified as normal)
- **FP (False Positives)**: 23 (normal incorrectly classified as attack)
- **FN (False Negatives)**: 12 (attack incorrectly classified as normal)
- **TP (True Positives)**: 212 (attack correctly classified as attack)

**FAR Calculation:**
- FAR = FP / (FP + TN) = 23 / (23 + 85) = 23 / 108 = **0.2130 = 21.30%**

**Interpretation:**
- 21.30% of normal samples are incorrectly flagged as attacks
- This means ~21 out of 100 normal samples trigger false alarms

## Comparison

| Model | FAR | Change |
|-------|-----|--------|
| Base Model | 12.96% | - |
| TTT Model | 21.30% | +8.34% |

## Analysis

### ⚠️ **FAR Increased with TTT Adaptation**

**Observation:**
- FAR increased from 12.96% → 21.30% (+8.34 percentage points)
- This is a **trade-off** for improved recall

**Why This Happened:**
1. **Class-Balanced Loss Focus**: The class-balanced loss weights minority class (Attack) higher
2. **Higher Recall Priority**: TTT adaptation prioritizes catching attacks (recall improved from 71.88% → 94.64%)
3. **Trade-off**: To catch more attacks, the model becomes more conservative and flags more normal samples as attacks

**Context:**
- **Recall improved**: 71.88% → 94.64% (+22.76%)
- **FAR increased**: 12.96% → 21.30% (+8.34%)
- **Net Effect**: Catching 22.76% more attacks, but with 8.34% more false alarms

### Security Application Perspective

**For Zero-Day Attack Detection:**

✅ **Positive Aspects:**
- **High Recall (94.64%)**: Catches nearly all attacks
- **Low False Negatives (12)**: Only 12 attacks missed vs 63 in base model
- **Critical for Security**: Missing attacks is more costly than false alarms

⚠️ **Trade-off:**
- **Higher FAR (21.30%)**: More false alarms
- **Operational Impact**: More manual review needed
- **Acceptable for Security**: False alarms are manageable, missed attacks are not

### Recommendations

1. **Current Configuration is Appropriate for Security Applications**:
   - High recall (94.64%) is critical for zero-day detection
   - FAR of 21.30% is acceptable when catching attacks is priority

2. **If FAR Needs to be Reduced**:
   - Could adjust threshold to reduce false positives
   - But this would reduce recall (trade-off)
   - Current balance (94.64% recall, 21.30% FAR) is reasonable

3. **Operational Considerations**:
   - Implement alert filtering/prioritization
   - Use confidence scores to reduce false alarms
   - Consider ensemble methods for better precision

## Conclusion

**FAR Results:**
- **Base Model**: 12.96%
- **TTT Model**: 21.30% (+8.34%)

**Assessment:**
- FAR increased as a trade-off for dramatically improved recall
- This is **acceptable for security applications** where catching attacks is critical
- The system prioritizes detection (94.64% recall) over reducing false alarms
- For zero-day attack detection, this is a **reasonable trade-off**

