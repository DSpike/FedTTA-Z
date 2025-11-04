# Diagnostic Tests for TTT Analysis

## Test 1: Confidence Distribution Analysis

### Purpose

This diagnostic test analyzes how TTT adaptation affects model confidence distributions. It compares:

- Prediction confidence before and after TTT
- Attack probability distributions before and after TTT
- High/low confidence prediction changes

### Usage

**Option 1: Run after main.py completes**

```bash
cd blockchain_federated_learning_project
python diagnostic_confidence_distribution.py
```

**Option 2: Run standalone (requires trained model)**

```bash
cd blockchain_federated_learning_project
# First ensure you have run main.py at least once to generate preprocessed data
python diagnostic_confidence_distribution.py
```

### Requirements

- System must have been initialized (run `main.py` at least once)
- Preprocessed data must exist in `system.preprocessed_data`
- Trained model must be available in `system.coordinator.model`

### Output Files

1. **Plots:**

   - `performance_plots/diagnostic_confidence_distribution.png` - High-resolution PNG plot
   - `performance_plots/diagnostic_confidence_distribution.pdf` - PDF plot for publications

2. **Statistics:**
   - `performance_plots/diagnostic_confidence_statistics.json` - Detailed statistics in JSON format

### What It Analyzes

1. **Confidence Distribution:**

   - Histogram of prediction confidence (max probability) before and after TTT
   - Mean confidence shift
   - Standard deviation changes

2. **Attack Probability Distribution:**

   - Histogram of P(Attack) before and after TTT
   - Shows how TTT affects attack probability estimates
   - Includes decision threshold line at 0.5

3. **Confidence Categories:**
   - High-confidence predictions (≥0.9)
   - Low-confidence predictions (<0.5)
   - Percentage changes in each category

### Interpreting Results

**Expected Behavior (Good TTT Adaptation):**

- ✅ Mean confidence increases (model becomes more confident)
- ✅ Attack probability distribution shifts appropriately
- ✅ High-confidence predictions increase
- ✅ Low-confidence predictions decrease

**Warning Signs:**

- ⚠️ Confidence decreases (possible overfitting)
- ⚠️ Extreme confidence (>0.99) for all samples (overconfidence)
- ⚠️ Attack probability stuck at extremes (0 or 1) for all samples

### Example Output

```
📊 STATISTICAL SUMMARY:
   Confidence mean change: +0.0523
   Confidence std change: -0.0124
   Attack prob mean change: +0.0345

🔍 ADDITIONAL ANALYSIS:
   High-confidence predictions (≥0.9):
     Before TTT: 145 (14.50%)
     After TTT: 198 (19.80%)
     Change: +53 (+5.30%)

   Low-confidence predictions (<0.5):
     Before TTT: 234 (23.40%)
     After TTT: 167 (16.70%)
     Change: -67 (-6.70%)
```

### Troubleshooting

**Error: "Preprocessed data not found"**

- Solution: Run `main.py` first to generate preprocessed data

**Error: "Model not found"**

- Solution: Ensure the system has been trained (run `main.py` with training enabled)

**Error: "CUDA out of memory"**

- Solution: The script uses a subset of test data (1000 samples). You can reduce this in the script if needed.

### Integration with Main System

The diagnostic test can be integrated into `main.py` by adding:

```python
# After TTT evaluation
from diagnostic_confidence_distribution import run_confidence_distribution_analysis
run_confidence_distribution_analysis()
```
