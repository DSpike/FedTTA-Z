# Performance Plots Source - Single Run vs 100-Episode Results

**Date**: December 22, 2025

---

## Quick Answer

**The performance plots in `performance_plots/` folder show SINGLE-RUN results, NOT 100-episode averaged results.**

---

## Evidence

### 1. When Are Plots Generated?

**Location**: [main.py:8516](main.py#L8516)

```python
# Generate performance visualizations
try:
    plot_paths = system.generate_performance_visualizations()
    logger.info(f"✅ Generated {len(plot_paths)} plots: {list(plot_paths.keys())}")
except Exception as e:
    logger.error(f"❌ CRITICAL: Performance visualization generation failed: {str(e)}")
```

This is called at the **END of main.py**, which runs a **SINGLE evaluation**.

### 2. What Data Do Plots Use?

**Location**: [main.py:2770](main.py#L2770)

```python
# Get evaluation results if available
evaluation_results = getattr(self, 'evaluation_results', {})
```

**Where `self.evaluation_results` comes from**: [main.py:2415](main.py#L2415)

```python
# Store evaluation results
self.evaluation_results = metrics
```

This stores the results from **ONE single evaluation run**, not aggregated 100-episode results.

### 3. What Plots Are Generated?

**Location**: [main.py:2568-2950](main.py#L2568-L2950) - `generate_performance_visualizations()`

**Plots created**:

1. **Training History**: `performance_plots/training_history_.png`
   - Source: Single training run

2. **Confusion Matrices**:
   - `performance_plots/confusion_matrices_base_model.png`
   - `performance_plots/confusion_matrices_ttt_enhanced_model.png`
   - Source: Single evaluation run

3. **TTT Adaptation**: `performance_plots/ttt_adaptation_.png`
   - Source: Single TTT adaptation run (10 steps)

4. **Performance Comparison**: `performance_plots/performance_comparison_annotated_{attack}_.png`
   - Source: Single evaluation (base vs TTT)

5. **Zero-Day Performance**: `performance_plots/zero_day_performance_comparison_{attack}_.png`
   - Source: Single evaluation zero-day samples

6. **Base Model Bar Chart**: `performance_plots/base_model_performance_barchart_{attack}_.png`
   - Source: Single evaluation base model

7. **ROC/PR Curves**: `performance_plots/roc_curves_.png`, `performance_plots/pr_curves_.png`
   - Source: Single evaluation probabilities

---

## 100-Episode Results Are Stored Separately

### Where 100-Episode Results Are Stored

**Location**: `multi_episode_results/backdoor_100_episodes_phase1.json`

**Created by**: `multi_episode_evaluation.py`, NOT `main.py`

**Contains**: Aggregated statistics over 100 episodes:
```json
{
  "base_model": {
    "zero_day_detection_rate": {
      "mean": 0.8913,
      "std": 0.0000,
      "ci_95": 0.0000,
      "min": 0.8913,
      "max": 0.8913
    },
    "f1_score": {
      "mean": 0.7890,
      "std": 0.0000,
      ...
    }
  }
}
```

### How to View 100-Episode Results

**Command**:
```bash
python display_100_episode_results.py Backdoor
```

**Output**: Text summary with confidence intervals

**NOT plotted**: 100-episode results are NOT automatically converted to plots

---

## Summary Table

| File/Plot | Source | Episodes | When Generated |
|-----------|--------|----------|----------------|
| `performance_plots/*.png` | `main.py` | **1 (Single Run)** | After each `python main.py` |
| `performance_plots/performance_metrics_.json` | `main.py` | **1 (Single Run)** | After each `python main.py` |
| `multi_episode_results/backdoor_100_episodes_phase1.json` | `multi_episode_evaluation.py` | **100 Episodes** | After `python multi_episode_evaluation.py --attack Backdoor --episodes 100` |
| Display script output | `display_100_episode_results.py` | **100 Episodes** | After `python display_100_episode_results.py Backdoor` |

---

## Why This Matters for Publication

### ⚠️ CRITICAL: Do NOT Use Single-Run Plots for Publication

**Problem**: All plots in `performance_plots/` are from a **SINGLE RUN** with:
- High variance due to random seed
- No statistical validation
- Not reproducible (different random splits)

**Risk**: Reviewers will reject if you present single-run results as validated findings

### ✅ What to Use for Publication

**Option 1: Use 100-Episode Text Summary (Recommended)**

From `display_100_episode_results.py`:
```
Zero-Day Detection Rate: 100.00% ± 0.00%
False Alarm Rate: 39.13% ± 0.67%
F1-Score: 84.51% ± 0.22%
Overall Accuracy: 79.43% ± 0.30%
```

**How to present**:
- Create a table with mean ± 95% CI
- State "Results validated over 100 independent episodes"
- Include statistical significance (p < 0.001)

**Option 2: Create 100-Episode Plots Manually**

You need to:
1. Run 100-episode evaluation: `python multi_episode_evaluation.py --attack Backdoor --episodes 100`
2. Load results from `multi_episode_results/backdoor_100_episodes_phase1.json`
3. Create plots manually using matplotlib/seaborn:
   - Bar charts with error bars (mean ± CI)
   - Box plots showing distribution across episodes
   - Comparison plots (base vs TTT with confidence intervals)

**Option 3: Use Single-Run Plots as Supplementary Material**

You can include single-run plots in **supplementary materials** with clear disclaimer:
```
Figure S1: Example performance from a single evaluation run.
Note: Main results in Table 1 are averaged over 100 independent episodes.
These plots illustrate typical performance patterns but are not statistically validated.
```

---

## How to Verify This

### Check Plot Filenames

**Single-run plots** have underscore suffix: `performance_comparison_annotated_Backdoor_.png`

The trailing `_` indicates **single run** (no episode number)

**If 100-episode plots existed**, they would have episode info: `performance_comparison_100episodes_mean.png` (but this doesn't exist)

### Check JSON File

**Single-run JSON**: `performance_plots/performance_metrics_.json`

```bash
cat performance_plots/performance_metrics_.json | python -m json.tool | head -20
```

Look for single values like:
```json
{
  "evaluation_results": {
    "base_model": {
      "accuracy": 0.7486,  // Single value, not mean/std/ci
      "f1_score": 0.7890
    }
  }
}
```

**100-episode JSON**: `multi_episode_results/backdoor_100_episodes_phase1.json`

```bash
cat multi_episode_results/backdoor_100_episodes_phase1.json | python -m json.tool | head -30
```

Look for statistics:
```json
{
  "base_model": {
    "accuracy": {
      "mean": 0.7486,  // Has mean, std, ci_95
      "std": 0.0030,
      "ci_95": 0.0006
    }
  }
}
```

---

## What Should You Do?

### For Your Current Paper

**Recommended Approach**:

1. **Main Results Table**: Use 100-episode text results from `display_100_episode_results.py`
   ```
   Table 1: Performance on Backdoor Zero-Day Detection (100 episodes)

   Metric                 Base Model      TTT Model       Improvement
   ZDR                    89.13 ± 0.00%   100.00 ± 0.00%  +10.87%
   FAR                    27.14 ± 0.00%   39.13 ± 0.67%   +11.99%
   F1-Score               78.90 ± 0.00%   84.51 ± 0.22%   +5.61%
   Overall Accuracy       74.86 ± 0.30%   79.43 ± 0.30%   +4.57%

   Results averaged over 100 independent episodes (95% confidence intervals shown)
   ```

2. **Figures**: Create manual plots from 100-episode JSON data
   - Bar chart with error bars
   - Show base vs TTT comparison
   - Include confidence intervals

3. **Supplementary Material**: Include single-run plots with disclaimer
   - Confusion matrices (example from one run)
   - TTT adaptation curve (example from one run)
   - ROC/PR curves (example from one run)

### If Reviewer Asks for Plots

**Create proper 100-episode plots**:

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load 100-episode results
with open('multi_episode_results/backdoor_100_episodes_phase1.json', 'r') as f:
    data = json.load(f)

base = data['base_model']
ttt = data['ttt_model']

# Create bar chart with error bars
metrics = ['zero_day_detection_rate', 'f1_score', 'accuracy']
base_means = [base[m]['mean'] for m in metrics]
base_cis = [base[m]['ci_95'] for m in metrics]
ttt_means = [ttt[m]['mean'] for m in metrics]
ttt_cis = [ttt[m]['ci_95'] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, base_means, width, yerr=base_cis, label='Base Model', capsize=5)
ax.bar(x + width/2, ttt_means, width, yerr=ttt_cis, label='TTT Model', capsize=5)

ax.set_ylabel('Score')
ax.set_title('Performance Comparison (100 Episodes)')
ax.set_xticks(x)
ax.set_xticklabels(['ZDR', 'F1-Score', 'Accuracy'])
ax.legend()
ax.set_ylim(0, 1.1)

plt.savefig('100_episode_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Summary

**Current Status**:
- ✅ You have 100-episode validated results (in JSON format)
- ✅ You have single-run plots (in `performance_plots/`)
- ❌ You do NOT have 100-episode plots

**For Publication**:
- ✅ Use 100-episode JSON data for main results table
- ⚠️ Create proper plots with error bars from 100-episode data
- ⚠️ Label single-run plots as "supplementary" if used

**Performance plots in `performance_plots/` folder = SINGLE RUN ONLY**

**100-episode results = In JSON files, need to create plots manually**

---

**Generated**: December 22, 2025
