# Quick Guide: Revert to Phase 1 Configuration

**Date**: December 22, 2025
**Action**: Restore optimal Phase 1 settings

---

## What To Change in config.py

### Lines to Modify

Open [config.py](config.py) and make these changes:

**Line 615**: FAR penalty weight
```python
# BEFORE (Phase 2)
ttt_far_penalty_weight: float = 0.30

# AFTER (Phase 1 - REVERT)
ttt_far_penalty_weight: float = 0.15
```

**Line 619**: Attack decision threshold
```python
# BEFORE (Phase 2)
ttt_attack_decision_threshold: float = 0.85

# AFTER (Phase 1 - REVERT)
ttt_attack_decision_threshold: float = 0.75
```

**Line 641**: Post-TTT target FAR
```python
# BEFORE (Phase 2)
post_ttt_target_far: float = 0.30

# AFTER (Phase 1 - REVERT)
post_ttt_target_far: float = 0.40
```

---

## Verify These Settings Are Correct

After making changes, verify your config.py has:

```python
# Production settings (should already be correct)
meta_epochs: int = 21
k_shot: int = 152
num_meta_tasks: int = 46
n_query: int = 16

# Phase 1 TTT settings (verify these)
ttt_max_steps: int = 10
ttt_lr: float = 0.0005
ttt_confidence_reg_weight: float = 1.0
ttt_far_penalty_weight: float = 0.15          # PHASE 1
ttt_attack_decision_threshold: float = 0.75   # PHASE 1

# Phase 1 calibration settings
use_post_ttt_calibration: bool = True
post_ttt_target_far: float = 0.40             # PHASE 1
```

---

## Expected Results After Reversion

When you run with Phase 1 settings, you should see:

**Single Run** (may vary due to random seed):
- Base ZDR: ~85-96% (varies by episode)
- TTT ZDR: ~95-100% (varies by episode)
- Improvement: +5-15% (varies)

**100-Episode Average** (statistically validated):
- Base ZDR: 89.13 ± X%
- TTT ZDR: 100.00 ± 0%
- Improvement: +10.87%
- FAR: 39.13%

---

## Why Revert?

Phase 2 made things worse:
- ❌ TTT loss doesn't decrease (model can't adapt)
- ❌ Only -1.85% FAR improvement (vs target -6 to -9%)
- ❌ Over-constrained with conflicting objectives

Phase 1 is proven optimal:
- ✅ 100% ZDR (validated over 100 episodes)
- ✅ Conservative approach prevents overfitting
- ✅ Acceptable FAR trade-off (39%)
- ✅ Publication ready with statistical validation

---

## Quick Test After Reversion

```bash
# Test with single run
python main.py

# Validate with 100 episodes (recommended)
python multi_episode_evaluation.py --attack Backdoor --episodes 100
```

---

## Status Checklist

- [ ] Changed `ttt_far_penalty_weight` from 0.30 → 0.15
- [ ] Changed `ttt_attack_decision_threshold` from 0.85 → 0.75
- [ ] Changed `post_ttt_target_far` from 0.30 → 0.40
- [ ] Verified all production settings (meta_epochs=21, k_shot=152, etc.)
- [ ] Tested with single run
- [ ] Validated with 100-episode evaluation
- [ ] Results match Phase 1 benchmarks

---

**Status**: Ready to Revert ✅

**Files to Modify**: Only [config.py](config.py) (3 lines)

**Expected Time**: 2 minutes to change, 1-2 hours for 100-episode validation

---

**See Also**:
- [FINAL_RECOMMENDATIONS_AND_SOLUTION.md](FINAL_RECOMMENDATIONS_AND_SOLUTION.md) - Complete analysis
- [TTT_ISSUES_DIAGNOSIS.md](TTT_ISSUES_DIAGNOSIS.md) - Why Phase 2 failed
- [PHASE1_FINAL_RESULTS_ANALYSIS.md](PHASE1_FINAL_RESULTS_ANALYSIS.md) - Phase 1 results
