# TTT Attack vs Normal Scatter Plot Interpretation Guide

## What the Scatter Plot Shows

The scatter plot visualizes how well TTT separates **attack samples** from **normal samples** during adaptation:

- **X-axis**: Sample index (which sample in the dataset)
- **Y-axis**: Attack probability (0.0 = definitely normal, 1.0 = definitely attack)
- **Blue dots**: Normal samples
- **Red dots**: Attack samples

## Key Statistics Explained

### 1. **Normal Mean Attack Probability**
- **What it means**: Average attack probability assigned to normal samples
- **Good value**: < 0.3 (normal samples should have low attack probability)
- **Bad value**: > 0.5 (normal samples being misclassified as attacks)

### 2. **Attack Mean Attack Probability**
- **What it means**: Average attack probability assigned to attack samples
- **Good value**: > 0.7 (attack samples should have high attack probability)
- **Bad value**: < 0.5 (attack samples being misclassified as normal)

### 3. **Separation (Attack Mean - Normal Mean)**
- **What it means**: The gap between attack and normal probabilities
- **Excellent**: > 0.5 (clear separation)
- **Good**: 0.3 - 0.5 (reasonable separation)
- **Moderate**: 0.1 - 0.3 (some separation, but could be better)
- **Poor**: < 0.1 (very little separation, TTT struggling)

### 4. **Improvement (End Separation - Beginning Separation)**
- **What it means**: How much TTT improved the separation during adaptation
- **Positive value**: TTT improved separation (good!)
- **Zero**: TTT didn't change separation
- **Negative value**: TTT made separation worse (bad!)

## How to Interpret Success

### ✅ **TTT is Successful if:**

1. **Separation > 0.3**: Clear gap between attack and normal probabilities
2. **Attack Mean > 0.7**: Attack samples have high attack probability
3. **Normal Mean < 0.3**: Normal samples have low attack probability
4. **Improvement > 0**: Separation increased during adaptation
5. **Visual separation**: Red dots (attacks) are clearly above blue dots (normal) in the scatter plot

### ❌ **TTT is NOT Successful if:**

1. **Separation < 0.1**: Very little gap between attack and normal
2. **Attack Mean < 0.5**: Attack samples have low attack probability (being misclassified)
3. **Normal Mean > 0.5**: Normal samples have high attack probability (false alarms)
4. **Improvement < 0**: Separation decreased during adaptation
5. **Visual overlap**: Red and blue dots are mixed together in the scatter plot

## Example Interpretations

### Example 1: Excellent Separation
```
Normal Mean: 0.150
Attack Mean: 0.850
Separation: 0.700
Improvement: +0.200
```
**Interpretation**: ✅ **EXCELLENT** - TTT successfully separates attacks (85% probability) from normal (15% probability) with a large gap (0.70). Adaptation improved separation by 0.20.

### Example 2: Good Separation
```
Normal Mean: 0.250
Attack Mean: 0.650
Separation: 0.400
Improvement: +0.100
```
**Interpretation**: ✅ **GOOD** - TTT provides reasonable separation (0.40 gap). Attacks have 65% probability, normal has 25% probability. Adaptation improved by 0.10.

### Example 3: Poor Separation
```
Normal Mean: 0.450
Attack Mean: 0.550
Separation: 0.100
Improvement: -0.050
```
**Interpretation**: ❌ **POOR** - Very small separation (0.10). Attacks (55%) and normal (45%) are too close. Adaptation actually made it worse (-0.05).

## Next Steps

To see these statistics in your next run:
1. Run `python main.py` again
2. Look for log messages starting with `📊 BEGINNING TTT` and `📊 END TTT`
3. The statistics will be printed in the terminal output
4. The scatter plot image will also show these statistics in text boxes

## Visual Guide

When looking at the scatter plot image:
- **Good separation**: Red dots (attacks) clustered in the top half (high probability), blue dots (normal) clustered in the bottom half (low probability)
- **Poor separation**: Red and blue dots mixed together, no clear pattern
- **Improvement**: End plot should show better separation than beginning plot



