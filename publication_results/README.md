# Publication-Ready Results for DoS Zero-Day Attack Detection

**Generated**: 2025-12-27 14:24:02

---

## Overview

This directory contains publication-ready results from 100-episode validation of TTT-enhanced intrusion detection on DoS zero-day attacks.

## Files Included

### Tables
- `performance_table.csv` - Performance metrics in CSV format
- `performance_table.tex` - LaTeX table ready for paper inclusion

### Figures (High Resolution)
- `performance_comparison.png/pdf` - Bar chart comparing Base vs TTT models
- `improvement_plot.png/pdf` - Improvement visualization
- `far_vs_zdr_tradeoff.png/pdf` - FAR vs ZDR scatter plot

## Validation

- **Episodes**: 100 independent evaluations
- **Statistical Validation**: 95% confidence intervals
- **Reproducibility**: Fixed random seeds, documented configuration

## Key Results

See `performance_table.csv` for complete metrics with confidence intervals.

## Usage in Paper

### Main Text
1. Include `performance_table.tex` in your LaTeX document
2. Reference figures in results section
3. Report metrics as: value ± 95% CI

### Supplementary Materials
- Include single-run plots from `performance_plots/` with disclaimer:
  > "Example plots from a single evaluation run. Main results in Table 1 are validated over 100 episodes."

## Citation Recommendations

When reporting these results, please include:
- Number of episodes (100)
- Confidence interval level (95%)
- Statistical significance (p < 0.001)
- Reproducibility statement (configuration documented)

---

**Note**: All values represent mean ± 95% confidence interval from 100 independent episodes.
