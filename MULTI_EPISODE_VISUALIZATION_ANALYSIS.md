# Multi-Episode Visualization Analysis

## Current State

### What the Multi-Episode Evaluator Does

The current [multi_episode_evaluation.py](multi_episode_evaluation.py) script:

1. ✅ **Runs multiple evaluation episodes** (10 episodes per attack)
2. ✅ **Computes statistics** (mean, std, 95% CI) across episodes
3. ✅ **Saves JSON results** to `multi_episode_results.json`
4. ✅ **Logs summary** to console

### What It DOESN'T Do

The script currently does **NOT**:

1. ❌ Generate any visualization plots
2. ❌ Create performance comparison charts
3. ❌ Show confidence intervals visually
4. ❌ Save confusion matrices
5. ❌ Generate ROC/PR curves

---

## The Problem

### During Multi-Episode Evaluation

When you run:
```bash
python multi_episode_evaluation.py --attack DoS --episodes 10
```

**What happens**:
1. Trains model once (40 meta-epochs) ✅
2. For each episode (10 times):
   - Samples episode data ✅
   - Calls `system.evaluate_base_model_only()` ✅
   - Calls `system.perform_coordinator_side_ttt_adaptation()` ✅
   - Calls `system.evaluate_adapted_model(adapted_model)` ✅
   - Stores metrics in memory ✅
3. Aggregates results across episodes ✅
4. Saves to JSON ✅

**What DOESN'T happen**:
- ❌ No plots are generated
- ❌ Visualizer is never called
- ❌ No performance_plots/ directory created

### Why This Happens

Looking at the code flow:

1. **Main.py workflow** (normal single-episode run):
   ```python
   # In main.py main() function around line 8250-8500
   system.evaluate_base_model_only()  # Computes metrics
   system.perform_coordinator_side_ttt_adaptation()  # TTT
   system.evaluate_adapted_model(adapted_model)  # Computes metrics

   # THEN - Generate plots using the visualizer
   system.generate_performance_plots(  # Around line 2750
       training_history=system.training_history,
       evaluation_results={
           'base_model': base_results,
           'adapted_model': adapted_results
       },
       ...
   )
   ```

2. **Multi-episode evaluator** (our new script):
   ```python
   # In multi_episode_evaluation.py
   for episode in range(10):
       system.evaluate_base_model_only()  # Computes metrics
       system.perform_coordinator_side_ttt_adaptation()  # TTT
       system.evaluate_adapted_model(adapted_model)  # Computes metrics
       # ❌ NO VISUALIZATION CALLS

   aggregate_results()  # Compute mean ± CI
   save_to_json()  # Save JSON
   # ❌ NO VISUALIZATION CALLS
   ```

**Key insight**: The evaluation methods (`evaluate_base_model_only`, `evaluate_adapted_model`) only **compute metrics** - they don't generate plots. Plot generation is a **separate step** that happens in `generate_performance_plots()`.

---

## Impact on Your Results

### Current Multi-Episode Results

You have:
- ✅ Statistical rigor (mean ± std ± CI)
- ✅ JSON with all metrics
- ❌ **No visual plots**

This means:
- You can report: "ZDR: 93.65% ± 1.36% (95% CI, n=10)"
- But you **cannot show**:
  - Bar charts with error bars
  - ROC curves
  - Confusion matrices
  - Performance comparison plots

### For Paper Submission

**Current state**:
- ✅ Statistically robust numbers
- ❌ Missing visualization figures

**What reviewers expect**:
- ✅ Mean ± confidence intervals (you have this)
- ✅ **Visual plots with error bars** (you DON'T have this)
- ✅ Confusion matrices
- ✅ ROC/PR curves

---

## Solutions

### Option 1: Generate Plots After Multi-Episode Evaluation (Quick Fix)

Add visualization calls at the end of multi-episode evaluation:

```python
# In multi_episode_evaluation.py, after aggregating results
def run_evaluation(self):
    # ... existing training and evaluation code ...

    # Aggregate results
    aggregated = self.aggregate_results(episode_results)

    # NEW: Generate plots with confidence intervals
    self._generate_multi_episode_plots(system, aggregated, episode_results)

    return aggregated

def _generate_multi_episode_plots(self, system, aggregated, episode_results):
    """Generate plots showing mean ± confidence intervals"""
    from visualization.performance_visualization import PerformanceVisualizer

    visualizer = PerformanceVisualizer(
        output_dir="multi_episode_plots",
        attack_name=self.config.zero_day_attack
    )

    # 1. Performance comparison with error bars
    # 2. Confusion matrices (aggregated)
    # 3. Per-episode ZDR plot
    # etc.
```

**Pros**:
- Quick to implement
- Reuses existing visualizer
- Generates all needed plots

**Cons**:
- Need to adapt visualizer to handle mean ± CI data
- Current visualizer expects single-episode data

---

### Option 2: Save Per-Episode Plots + Final Aggregated Plots (Comprehensive)

Generate plots for:
1. **Each episode** (optional, for debugging)
2. **Final aggregated results** with confidence intervals

```python
def run_evaluation(self):
    # ... training ...

    for episode_idx in range(self.n_episodes):
        result = self.evaluate_single_episode(system, episode_idx, test_pool)
        episode_results.append(result)

        # OPTIONAL: Save per-episode plots
        if self.save_per_episode_plots:
            self._save_episode_plots(system, episode_idx, result)

    # Aggregate results
    aggregated = self.aggregate_results(episode_results)

    # Generate final plots with confidence intervals
    self._generate_aggregated_plots(system, aggregated, episode_results)
```

**Pros**:
- Complete visualization coverage
- Can debug individual episodes
- Publication-ready aggregated plots

**Cons**:
- More complex
- Takes longer to run
- Generates many files

---

### Option 3: Use Existing Plots from Last Episode (Hacky but Fast)

Just run the normal visualization on the **last episode**:

```python
def run_evaluation(self):
    # ... training and multi-episode evaluation ...

    # Use last episode for visualization
    # (Not statistically representative, but gives visual reference)
    last_episode_base = episode_results[-1]['base_model']
    last_episode_ttt = episode_results[-1]['ttt_model']

    system.generate_performance_plots(
        training_history=system.training_history,
        evaluation_results={
            'base_model': last_episode_base,
            'adapted_model': last_episode_ttt
        }
    )
```

**Pros**:
- Minimal code changes
- Reuses existing visualization

**Cons**:
- ❌ **Philosophically wrong** - plots don't show CI
- ❌ Misleading - shows one episode, not aggregated
- ❌ Not publication-ready

---

## Recommended Approach

### Recommended: Option 1 (Enhanced)

Create a **multi-episode-aware visualizer** that:

1. **Takes aggregated results** (mean ± std ± CI)
2. **Generates publication-ready plots**:
   - Bar charts with error bars (ZDR, Accuracy, FAR)
   - Box plots showing distribution across episodes
   - Per-episode scatter plot showing variability
   - Aggregated confusion matrix (mean)

### Implementation Plan

1. **Add visualization method to multi_episode_evaluation.py**:
   ```python
   def _generate_multi_episode_plots(self, system, aggregated, episode_results):
       """Generate plots for multi-episode results with confidence intervals"""
   ```

2. **Create specialized plot functions**:
   - `plot_performance_with_error_bars()` - Bar chart with CI
   - `plot_per_episode_trends()` - Line plot showing all episodes
   - `plot_aggregated_confusion_matrix()` - Mean confusion matrix

3. **Modify existing visualizer** (optional):
   - Add support for multi-episode data
   - Handle mean ± CI in existing plot functions

---

## What You Should Do

### Immediate Next Steps

1. **For quick results** (to see plots now):
   ```bash
   # Run single-episode evaluation (uses existing visualization)
   python main.py
   ```
   This generates all plots, but for **single episode only** (no CI).

2. **For publication-quality results** (multi-episode with plots):
   - I can implement Option 1 (add visualization to multi-episode evaluator)
   - This will generate plots with error bars showing mean ± CI
   - Takes ~30 minutes to implement

3. **For final paper submission**:
   - Use multi-episode evaluation (10 episodes per attack)
   - Generate plots with confidence intervals
   - Report: "ZDR: 93.65% ± 1.36% (95% CI, n=10 episodes)"
   - Show: Bar chart with error bars

---

## Current Multi-Episode Results (From Your Last Run)

From `multi_episode_results.json`:

```json
{
  "base_model": {
    "zero_day_detection_rate": {
      "mean": 0.8213,  // 82.13%
      "std": 0.0420,   // ±4.20%
      "ci_95": 0.0260  // ±2.60% (95% CI)
    }
  },
  "ttt_model": {
    "zero_day_detection_rate": {
      "mean": 0.9365,  // 93.65%
      "std": 0.0219,   // ±2.19%
      "ci_95": 0.0136  // ±1.36% (95% CI)
    }
  }
}
```

**What you can report**:
> "Our TTT-enhanced model achieves 93.65% ± 1.36% zero-day detection rate
> across 10 independent episodes (95% CI), compared to 82.13% ± 2.60% for
> the base transductive meta-learning model."

**What you're missing**:
- ❌ Figure showing this visually with error bars
- ❌ Confusion matrices
- ❌ ROC/PR curves with confidence bands

---

## Conclusion

**Current status**:
- ✅ Multi-episode evaluation works correctly
- ✅ Statistical rigor achieved (mean ± CI)
- ✅ Results saved to JSON
- ❌ **No visualization plots generated**

**To make it publication-ready**:
- Need to add visualization support
- Generate plots with error bars
- Show confidence intervals visually

**Would you like me to**:
1. ✅ Implement visualization for multi-episode results?
2. ✅ Create plots with error bars and confidence intervals?
3. ✅ Make it publication-ready?

Let me know if you want me to proceed with implementation!
