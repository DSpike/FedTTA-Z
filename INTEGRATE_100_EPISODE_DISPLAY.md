# How to Display 100-Episode Results in main.py

**Issue**: main.py only shows single-run results, not the 100-episode validation results.

**Reason**: The 100-episode evaluation runs separately via `multi_episode_evaluation.py` and saves results to JSON files.

---

## Quick Solution: Add Display Code to main.py

### Location
Add this code in [main.py](main.py) after line 8505 (after the comprehensive summary generation).

### Code to Add

```python
        # ============================================================================
        # DISPLAY 100-EPISODE RESULTS (if available)
        # ============================================================================
        try:
            import os
            from pathlib import Path

            # Check for 100-episode results file
            attack_type = config.zero_day_attack
            phase1_file = f"multi_episode_results/{attack_type.lower()}_100_episodes_phase1.json"
            phase2_file = f"multi_episode_results/{attack_type.lower()}_100_episodes_phase2.json"

            if Path(phase1_file).exists():
                logger.info("\n" + "="*80)
                logger.info("100-EPISODE VALIDATION RESULTS (STATISTICAL GOLD STANDARD)")
                logger.info("="*80)

                with open(phase1_file, 'r') as f:
                    phase1_data = json.load(f)

                base = phase1_data['base_model']
                ttt = phase1_data['ttt_model']

                logger.info(f"\n📊 PHASE 1 RESULTS (100 independent episodes):")
                logger.info(f"  Attack Type: {attack_type}")
                logger.info(f"  Episodes: {phase1_data['metadata']['total_episodes']}")
                logger.info("")
                logger.info(f"  Zero-Day Detection Rate:")
                logger.info(f"    Base Model:  {base['zero_day_detection_rate']['mean']*100:6.2f}% ± {base['zero_day_detection_rate']['std']*100:.2f}%")
                logger.info(f"    TTT Model:   {ttt['zero_day_detection_rate']['mean']*100:6.2f}% ± {ttt['zero_day_detection_rate']['std']*100:.2f}%")
                logger.info(f"    Improvement: +{(ttt['zero_day_detection_rate']['mean'] - base['zero_day_detection_rate']['mean'])*100:5.2f}%")
                logger.info("")
                logger.info(f"  False Alarm Rate:")
                logger.info(f"    Base Model:  {base['false_alarm_rate']['mean']*100:6.2f}% ± {base['false_alarm_rate']['std']*100:.2f}%")
                logger.info(f"    TTT Model:   {ttt['false_alarm_rate']['mean']*100:6.2f}% ± {ttt['false_alarm_rate']['std']*100:.2f}%")
                logger.info(f"    Change:      +{(ttt['false_alarm_rate']['mean'] - base['false_alarm_rate']['mean'])*100:5.2f}%")
                logger.info("")
                logger.info(f"  F1-Score:")
                logger.info(f"    Base Model:  {base['f1_score']['mean']*100:6.2f}% ± {base['f1_score']['std']*100:.2f}%")
                logger.info(f"    TTT Model:   {ttt['f1_score']['mean']*100:6.2f}% ± {ttt['f1_score']['std']*100:.2f}%")
                logger.info(f"    Improvement: +{(ttt['f1_score']['mean'] - base['f1_score']['mean'])*100:5.2f}%")
                logger.info("")
                logger.info(f"  Overall Accuracy:")
                logger.info(f"    Base Model:  {base['accuracy']['mean']*100:6.2f}% ± {base['accuracy']['std']*100:.2f}%")
                logger.info(f"    TTT Model:   {ttt['accuracy']['mean']*100:6.2f}% ± {ttt['accuracy']['std']*100:.2f}%")
                logger.info(f"    Improvement: +{(ttt['accuracy']['mean'] - base['accuracy']['mean'])*100:5.2f}%")
                logger.info("")

                # Check if Phase 2 results exist
                if Path(phase2_file).exists():
                    with open(phase2_file, 'r') as f:
                        phase2_data = json.load(f)

                    ttt_p2 = phase2_data['ttt_model']

                    logger.info(f"📊 PHASE 2 RESULTS (100 episodes with aggressive threshold tuning):")
                    logger.info(f"  Zero-Day Detection Rate: {ttt_p2['zero_day_detection_rate']['mean']*100:6.2f}% ± {ttt_p2['zero_day_detection_rate']['std']*100:.2f}%")
                    logger.info(f"  False Alarm Rate:        {ttt_p2['false_alarm_rate']['mean']*100:6.2f}% ± {ttt_p2['false_alarm_rate']['std']*100:.2f}%")
                    logger.info(f"  F1-Score:                {ttt_p2['f1_score']['mean']*100:6.2f}% ± {ttt_p2['f1_score']['std']*100:.2f}%")
                    logger.info(f"  Accuracy:                {ttt_p2['accuracy']['mean']*100:6.2f}% ± {ttt_p2['accuracy']['std']*100:.2f}%")
                    logger.info("")
                    logger.info(f"  FAR Change (Phase 1 → Phase 2): {(ttt_p2['false_alarm_rate']['mean'] - ttt['false_alarm_rate']['mean'])*100:+.2f}%")
                    logger.info("")

                logger.info("⚠️  NOTE: Single-run results (shown above) may vary due to random seed.")
                logger.info("    For publication, ALWAYS use 100-episode average results!")
                logger.info("="*80 + "\n")

            else:
                logger.info(f"\nℹ️  100-episode validation results not found at: {phase1_file}")
                logger.info(f"   Run: python multi_episode_evaluation.py --attack {attack_type} --episodes 100")
                logger.info(f"   to generate statistical validation results.\n")

        except Exception as e:
            logger.warning(f"⚠️ Could not load 100-episode results: {str(e)}")
        # ============================================================================
```

---

## What This Does

When you run `python main.py`, it will:

1. ✅ Run the normal single evaluation (as before)
2. ✅ Show comprehensive summary (as integrated)
3. ✅ **NEW**: Check if 100-episode results exist
4. ✅ **NEW**: If found, display them with clear formatting
5. ✅ **NEW**: Show both Phase 1 and Phase 2 results (if available)
6. ✅ **NEW**: Remind user to use 100-episode results for publication

---

## Example Output

After adding this code, main.py will show:

```
================================================================================
100-EPISODE VALIDATION RESULTS (STATISTICAL GOLD STANDARD)
================================================================================

📊 PHASE 1 RESULTS (100 independent episodes):
  Attack Type: Backdoor
  Episodes: 100

  Zero-Day Detection Rate:
    Base Model:   89.13% ± 0.00%
    TTT Model:   100.00% ± 0.00%
    Improvement: +10.87%

  False Alarm Rate:
    Base Model:   27.14% ± 0.00%
    TTT Model:    39.13% ± 0.67%
    Change:      +11.99%

  F1-Score:
    Base Model:   78.90% ± 0.00%
    TTT Model:    84.51% ± 0.22%
    Improvement:  +5.61%

  Overall Accuracy:
    Base Model:   74.86% ± 0.00%
    TTT Model:    79.43% ± 0.30%
    Improvement:  +4.57%

📊 PHASE 2 RESULTS (100 episodes with aggressive threshold tuning):
  Zero-Day Detection Rate:  99.98% ± 0.23%
  False Alarm Rate:         37.28% ± 0.58%
  F1-Score:                 84.76% ± 0.21%
  Accuracy:                 80.05% ± 0.28%

  FAR Change (Phase 1 → Phase 2): -1.85%

⚠️  NOTE: Single-run results (shown above) may vary due to random seed.
    For publication, ALWAYS use 100-episode average results!
================================================================================
```

---

## Alternative: Standalone Script

If you don't want to modify main.py, create a standalone script:

### display_100_episode_results.py

```python
#!/usr/bin/env python3
"""Display 100-episode validation results"""

import json
from pathlib import Path
import sys

def display_results(attack_type="Backdoor"):
    phase1_file = f"multi_episode_results/{attack_type.lower()}_100_episodes_phase1.json"
    phase2_file = f"multi_episode_results/{attack_type.lower()}_100_episodes_phase2.json"

    if not Path(phase1_file).exists():
        print(f"❌ Results not found: {phase1_file}")
        print(f"   Run: python multi_episode_evaluation.py --attack {attack_type} --episodes 100")
        return

    with open(phase1_file, 'r') as f:
        phase1_data = json.load(f)

    base = phase1_data['base_model']
    ttt = phase1_data['ttt_model']

    print("="*80)
    print("100-EPISODE VALIDATION RESULTS")
    print("="*80)
    print(f"\nAttack Type: {attack_type}")
    print(f"Episodes: {phase1_data['metadata']['total_episodes']}")
    print(f"\n📊 PHASE 1 RESULTS:\n")

    print("Zero-Day Detection Rate:")
    print(f"  Base:  {base['zero_day_detection_rate']['mean']*100:6.2f}% ± {base['zero_day_detection_rate']['std']*100:.2f}%")
    print(f"  TTT:   {ttt['zero_day_detection_rate']['mean']*100:6.2f}% ± {ttt['zero_day_detection_rate']['std']*100:.2f}%")
    print(f"  Gain:  +{(ttt['zero_day_detection_rate']['mean'] - base['zero_day_detection_rate']['mean'])*100:5.2f}%\n")

    print("False Alarm Rate:")
    print(f"  Base:  {base['false_alarm_rate']['mean']*100:6.2f}% ± {base['false_alarm_rate']['std']*100:.2f}%")
    print(f"  TTT:   {ttt['false_alarm_rate']['mean']*100:6.2f}% ± {ttt['false_alarm_rate']['std']*100:.2f}%")
    print(f"  Change: +{(ttt['false_alarm_rate']['mean'] - base['false_alarm_rate']['mean'])*100:5.2f}%\n")

    print("F1-Score:")
    print(f"  Base:  {base['f1_score']['mean']*100:6.2f}% ± {base['f1_score']['std']*100:.2f}%")
    print(f"  TTT:   {ttt['f1_score']['mean']*100:6.2f}% ± {ttt['f1_score']['std']*100:.2f}%")
    print(f"  Gain:  +{(ttt['f1_score']['mean'] - base['f1_score']['mean'])*100:5.2f}%\n")

    print("Accuracy:")
    print(f"  Base:  {base['accuracy']['mean']*100:6.2f}% ± {base['accuracy']['std']*100:.2f}%")
    print(f"  TTT:   {ttt['accuracy']['mean']*100:6.2f}% ± {ttt['accuracy']['std']*100:.2f}%")
    print(f"  Gain:  +{(ttt['accuracy']['mean'] - base['accuracy']['mean'])*100:5.2f}%\n")

    if Path(phase2_file).exists():
        with open(phase2_file, 'r') as f:
            phase2_data = json.load(f)
        ttt_p2 = phase2_data['ttt_model']

        print("📊 PHASE 2 RESULTS:\n")
        print(f"  ZDR: {ttt_p2['zero_day_detection_rate']['mean']*100:6.2f}% ± {ttt_p2['zero_day_detection_rate']['std']*100:.2f}%")
        print(f"  FAR: {ttt_p2['false_alarm_rate']['mean']*100:6.2f}% ± {ttt_p2['false_alarm_rate']['std']*100:.2f}%")
        print(f"  F1:  {ttt_p2['f1_score']['mean']*100:6.2f}% ± {ttt_p2['f1_score']['std']*100:.2f}%")
        print(f"  Acc: {ttt_p2['accuracy']['mean']*100:6.2f}% ± {ttt_p2['accuracy']['std']*100:.2f}%")
        print(f"\n  FAR Improvement (P1→P2): {(ttt_p2['false_alarm_rate']['mean'] - ttt['false_alarm_rate']['mean'])*100:+.2f}%\n")

    print("="*80)

if __name__ == "__main__":
    attack = sys.argv[1] if len(sys.argv) > 1 else "Backdoor"
    display_results(attack)
```

Then run:
```bash
python display_100_episode_results.py Backdoor
```

---

## Summary

**Why main.py doesn't show 100-episode results**:
- main.py runs a single evaluation (one random seed)
- 100-episode evaluation is a separate script (`multi_episode_evaluation.py`)
- Results are saved to JSON files, not displayed automatically

**Solutions**:
1. ✅ View manually with the command shown above
2. ✅ Add display code to main.py (recommended for convenience)
3. ✅ Create standalone display script (cleanest separation)

**Recommendation**: Add the display code to main.py so every run automatically shows both:
- Single-run results (for quick testing)
- 100-episode results (for publication validity)

---

**Current Status**:
- ✅ 100-episode Phase 1 results exist: `multi_episode_results/backdoor_100_episodes_phase1.json`
- ✅ 100-episode Phase 2 results exist: `multi_episode_results/backdoor_100_episodes_phase2.json`
- ⚠️  main.py doesn't automatically display them (but can be easily added)
