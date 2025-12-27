# Running Evaluation with Confidence Regularization Fix

## Command to Run

```bash
cd "c:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn"
python run_comprehensive_multi_episode_evaluation.py --episodes 10 --episode-size 800
```

## What This Does

1. Runs evaluation on ALL 9 attack types
2. 10 episodes per attack (90 total evaluations)
3. Each episode tests ~800 samples
4. Generates comprehensive results with confidence intervals

## Expected Timeline

- **Per attack**: ~15-20 minutes
- **Total time**: ~3 hours
- **Progress**: Watch for attack-by-attack completion

## What to Monitor

### During Training (Each Attack):
Look for in logs:
```
TTT Step X/100: Loss=X.XX, Entropy=X.XX, Pseudo=X.XX,
                L2_Reg=X.XX, FAR_Penalty=X.XX, ConfReg=X.XX
```

**ConfReg should be ~0.02-0.10** (active regularization)

### After Each Attack:
Check metrics:
- **FAR**: Should be 10-20% (down from ~41%)
- **ZDR**: Should be >85% (maintain from ~93%)
- **Accuracy**: Should be 75-85%
- **F1-Score**: Should be 75-85%

## Output Files

When complete:
- `multi_episode_results/comprehensive_multi_episode_results.json`
- `multi_episode_results/comprehensive_multi_episode_results.md`

## Next Steps After Completion

1. **Check Results**:
   ```bash
   cat multi_episode_results/comprehensive_multi_episode_results.md
   ```

2. **Analyze FAR**:
   - If FAR 8-12%: ✅ **SUCCESS** - Proceed to paper writing
   - If FAR 12-18%: ⚠️ **GOOD** - Tune parameters or accept mid-tier
   - If FAR >18%: ⏸️ **TUNE** - Adjust `ttt_confidence_reg_weight`

3. **Update Excel**:
   - Add new results to comparison
   - Show before/after FAR reduction
   - Highlight improvement

4. **Prepare for Publication**:
   - If target met: Write for top-tier (IEEE TIFS, TDSC)
   - If close: Write for mid-tier with honest analysis

## Tuning Parameters (If Needed)

Edit `config.py`:

### To Reduce FAR Further:
```python
ttt_confidence_reg_weight: float = 0.5  # Increase from 0.4
ttt_target_confidence: float = 0.70     # Decrease from 0.75
```

### To Maintain ZDR:
```python
ttt_confidence_reg_weight: float = 0.3  # Decrease from 0.4
ttt_target_confidence: float = 0.80     # Increase from 0.75
```

## Ready to Start!

Run the command above and monitor progress.
Expected completion: ~3 hours from now.
