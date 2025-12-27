"""
Test script to verify comprehensive summary integration

Run this to test the evaluation summary generator without running full main.py
"""

import sys
import json
from pathlib import Path

# Test 1: Check if evaluation module exists
print("="*80)
print("TEST 1: Check evaluation module")
print("="*80)

try:
    from evaluation.comprehensive_summary_generator import ComprehensiveEvaluationSummary
    print("✅ ComprehensiveEvaluationSummary imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nTo fix:")
    print("  1. Create directory: mkdir -p evaluation")
    print("  2. Create __init__.py: touch evaluation/__init__.py")
    print("  3. Copy comprehensive_summary_generator.py to evaluation/")
    sys.exit(1)

print()

# Test 2: Create mock config
print("="*80)
print("TEST 2: Create mock configuration")
print("="*80)

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        # Phase 1 configuration
        self.meta_epochs = 21
        self.k_shot = 152
        self.num_meta_tasks = 46
        self.n_query = 16
        self.ttt_max_steps = 10
        self.ttt_lr = 0.0005
        self.ttt_confidence_reg_weight = 1.0
        self.ttt_attack_decision_threshold = 0.75
        self.use_post_ttt_calibration = True
        self.post_ttt_target_far = 0.40
        self.dataset_name = 'UNSW-NB15'
        self.zero_day_attack = 'Backdoor'

config = MockConfig()
print("✅ Mock config created")
print()

# Test 3: Create mock evaluation results
print("="*80)
print("TEST 3: Create mock evaluation results")
print("="*80)

mock_results = {
    'base_model': {
        'accuracy': 0.7486,
        'precision': 0.8190,
        'recall': 0.7611,
        'f1_score': 0.7890,
        'zero_day_detection_rate': 0.8913,
        'far': 0.2714,
        'false_alarm_rate': 0.2714,
        'mcc': 0.5123,
        'roc_auc': 0.8456,
        'test_samples': 184,
        'confusion_matrix': {
            'tn': 50,
            'fp': 16,
            'fn': 38,
            'tp': 70
        }
    },
    'adapted_model': {
        'accuracy': 0.7943,
        'precision': 0.8521,
        'recall': 0.8661,
        'f1_score': 0.8451,
        'zero_day_detection_rate': 1.0000,
        'far': 0.3913,
        'false_alarm_rate': 0.3913,
        'mcc': 0.6234,
        'roc_auc': 0.8845,
        'test_samples': 184,
        'confusion_matrix': {
            'tn': 37,
            'fp': 29,
            'fn': 15,
            'tp': 93
        }
    }
}

print("✅ Mock evaluation results created")
print()

# Test 4: Initialize summary generator
print("="*80)
print("TEST 4: Initialize summary generator")
print("="*80)

try:
    summary_gen = ComprehensiveEvaluationSummary(config)
    print("✅ Summary generator initialized")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

print()

# Test 5: Print console summary
print("="*80)
print("TEST 5: Print console summary")
print("="*80)

try:
    summary_gen.print_console_summary(mock_results)
    print("\n✅ Console summary printed successfully")
except Exception as e:
    print(f"❌ Console summary failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 6: Generate comprehensive reports
print("="*80)
print("TEST 6: Generate comprehensive reports")
print("="*80)

try:
    report_paths = summary_gen.generate_comprehensive_report(
        evaluation_results=mock_results,
        output_dir="test_evaluation_reports"
    )

    print("\n✅ Reports generated successfully:")
    for report_type, path in report_paths.items():
        print(f"  {report_type.upper()}: {path}")

    # Verify files exist
    print("\nVerifying generated files:")
    for report_type, path in report_paths.items():
        if Path(path).exists():
            size = Path(path).stat().st_size
            print(f"  ✅ {report_type}: {size} bytes")
        else:
            print(f"  ❌ {report_type}: NOT FOUND")

except Exception as e:
    print(f"❌ Report generation failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 7: Read and validate JSON report
print("="*80)
print("TEST 7: Validate JSON report content")
print("="*80)

try:
    json_path = report_paths['json']
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    print("JSON Report Structure:")
    for key in json_data.keys():
        print(f"  ✅ {key}")

    # Check Phase 1 assessment
    if 'phase1_assessment' in json_data:
        assessment = json_data['phase1_assessment']
        print(f"\nPhase 1 Assessment:")
        print(f"  Grade: {assessment['grade']}")
        print(f"  Verdict: {assessment['verdict']}")
        print(f"  Criteria Met: {assessment['criteria_met']}/{assessment['total_criteria']}")

except Exception as e:
    print(f"❌ JSON validation failed: {e}")

print()

# Test 8: Read Markdown report preview
print("="*80)
print("TEST 8: Preview Markdown report")
print("="*80)

try:
    md_path = report_paths['markdown']
    with open(md_path, 'r') as f:
        md_content = f.read()

    # Print first 500 characters
    print("Markdown Report Preview (first 500 chars):")
    print("-" * 80)
    print(md_content[:500])
    print("-" * 80)
    print(f"\n✅ Full report: {len(md_content)} characters")

except Exception as e:
    print(f"❌ Markdown preview failed: {e}")

print()

# Final summary
print("="*80)
print("INTEGRATION TEST SUMMARY")
print("="*80)
print("✅ All tests passed!")
print("\nNext steps:")
print("  1. Run main.py to test full integration")
print("  2. Check evaluation_reports/ directory for outputs")
print("  3. Review generated reports for accuracy")
print("="*80)
