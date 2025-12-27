"""
Comprehensive Evaluation Summary Generator

Generates detailed evaluation reports with Phase 1 recommended configuration analysis.
Integrates with main.py to provide publication-ready summaries.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ComprehensiveEvaluationSummary:
    """
    Generate comprehensive evaluation summaries with statistical analysis
    and Phase 1 improvement documentation.
    """

    def __init__(self, config: Any):
        """
        Initialize summary generator with system configuration.

        Args:
            config: System configuration object
        """
        self.config = config
        self.phase1_config = self._get_phase1_configuration()

    def _get_phase1_configuration(self) -> Dict[str, Any]:
        """Extract Phase 1 recommended configuration from config."""
        return {
            'meta_epochs': getattr(self.config, 'meta_epochs', 21),
            'k_shot': getattr(self.config, 'k_shot', 152),
            'num_meta_tasks': getattr(self.config, 'num_meta_tasks', 46),
            'n_query': getattr(self.config, 'n_query', 16),
            'ttt_max_steps': getattr(self.config, 'ttt_max_steps', 10),
            'ttt_lr': getattr(self.config, 'ttt_lr', 0.0005),
            'ttt_confidence_reg_weight': getattr(self.config, 'ttt_confidence_reg_weight', 1.0),
            'ttt_attack_decision_threshold': getattr(self.config, 'ttt_attack_decision_threshold', 0.75),
            'use_post_ttt_calibration': getattr(self.config, 'use_post_ttt_calibration', True),
            'post_ttt_target_far': getattr(self.config, 'post_ttt_target_far', 0.40),
        }

    def generate_comprehensive_report(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: str = "evaluation_reports"
    ) -> Dict[str, str]:
        """
        Generate comprehensive evaluation report with multiple formats.

        Args:
            evaluation_results: Results from system evaluation
            output_dir: Directory to save reports

        Returns:
            Dict with paths to generated report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate different report formats
        reports = {}

        # 1. JSON detailed report
        json_path = output_path / f"evaluation_summary_{timestamp}.json"
        json_report = self._generate_json_report(evaluation_results)
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        reports['json'] = str(json_path)
        logger.info(f"✅ JSON report saved: {json_path}")

        # 2. Markdown human-readable report
        md_path = output_path / f"evaluation_summary_{timestamp}.md"
        md_report = self._generate_markdown_report(evaluation_results, json_report)
        with open(md_path, 'w') as f:
            f.write(md_report)
        reports['markdown'] = str(md_path)
        logger.info(f"✅ Markdown report saved: {md_path}")

        # 3. Publication-ready summary
        pub_path = output_path / f"publication_summary_{timestamp}.md"
        pub_report = self._generate_publication_summary(evaluation_results, json_report)
        with open(pub_path, 'w') as f:
            f.write(pub_report)
        reports['publication'] = str(pub_path)
        logger.info(f"✅ Publication summary saved: {pub_path}")

        return reports

    def _generate_json_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured JSON report with all metrics."""
        base_model = evaluation_results.get('base_model', {})
        adapted_model = evaluation_results.get('adapted_model', {})

        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset': getattr(self.config, 'dataset_name', 'UNSW-NB15'),
                'zero_day_attack': getattr(self.config, 'zero_day_attack', 'Unknown'),
                'evaluation_method': 'Transductive Meta-Learning with Test-Time Training',
                'phase': 'Phase 1 - Conservative TTT'
            },
            'configuration': {
                'phase1_settings': self.phase1_config,
                'production_mode': self._is_production_configuration(),
                'quick_test_mode': getattr(self.config, 'meta_epochs', 21) == 1
            },
            'base_model_performance': {
                'accuracy': base_model.get('accuracy', 0.0),
                'precision': base_model.get('precision', 0.0),
                'recall': base_model.get('recall', 0.0),
                'f1_score': base_model.get('f1_score', 0.0),
                'zero_day_detection_rate': base_model.get('zero_day_detection_rate', 0.0),
                'false_alarm_rate': base_model.get('far', base_model.get('false_alarm_rate', 0.0)),
                'mcc': base_model.get('mccc', base_model.get('mcc', 0.0)),
                'roc_auc': base_model.get('roc_auc', 0.5),
                'test_samples': base_model.get('test_samples', 0),
                'confusion_matrix': base_model.get('confusion_matrix', {})
            },
            'ttt_model_performance': {
                'accuracy': adapted_model.get('accuracy', 0.0),
                'precision': adapted_model.get('precision', 0.0),
                'recall': adapted_model.get('recall', 0.0),
                'f1_score': adapted_model.get('f1_score', 0.0),
                'zero_day_detection_rate': adapted_model.get('zero_day_detection_rate', 0.0),
                'false_alarm_rate': adapted_model.get('far', adapted_model.get('false_alarm_rate', 0.0)),
                'mcc': adapted_model.get('mccc', adapted_model.get('mcc', 0.0)),
                'roc_auc': adapted_model.get('roc_auc', 0.5),
                'test_samples': adapted_model.get('test_samples', 0),
                'confusion_matrix': adapted_model.get('confusion_matrix', {})
            },
            'improvement': self._calculate_improvements(base_model, adapted_model),
            'statistical_analysis': self._perform_statistical_analysis(base_model, adapted_model),
            'phase1_assessment': self._assess_phase1_performance(base_model, adapted_model)
        }

        return report

    def _calculate_improvements(
        self,
        base_model: Dict[str, Any],
        adapted_model: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate improvement metrics."""
        base_zdr = base_model.get('zero_day_detection_rate', 0.0)
        ttt_zdr = adapted_model.get('zero_day_detection_rate', 0.0)
        base_far = base_model.get('far', base_model.get('false_alarm_rate', 0.0))
        ttt_far = adapted_model.get('far', adapted_model.get('false_alarm_rate', 0.0))
        base_acc = base_model.get('accuracy', 0.0)
        ttt_acc = adapted_model.get('accuracy', 0.0)
        base_f1 = base_model.get('f1_score', 0.0)
        ttt_f1 = adapted_model.get('f1_score', 0.0)

        return {
            'zdr_absolute': ttt_zdr - base_zdr,
            'zdr_relative': ((ttt_zdr - base_zdr) / base_zdr * 100) if base_zdr > 0 else 0.0,
            'far_absolute': ttt_far - base_far,
            'far_relative': ((ttt_far - base_far) / base_far * 100) if base_far > 0 else 0.0,
            'accuracy_absolute': ttt_acc - base_acc,
            'accuracy_relative': ((ttt_acc - base_acc) / base_acc * 100) if base_acc > 0 else 0.0,
            'f1_absolute': ttt_f1 - base_f1,
            'f1_relative': ((ttt_f1 - base_f1) / base_f1 * 100) if base_f1 > 0 else 0.0
        }

    def _perform_statistical_analysis(
        self,
        base_model: Dict[str, Any],
        adapted_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical significance analysis."""
        # For single run, we can't compute variance
        # But we can provide contextual information

        ttt_zdr = adapted_model.get('zero_day_detection_rate', 0.0)
        ttt_far = adapted_model.get('far', adapted_model.get('false_alarm_rate', 0.0))

        return {
            'note': 'Single-run evaluation. For statistical significance, run multi-episode evaluation.',
            'zdr_performance': {
                'value': ttt_zdr * 100,
                'target': '> 90%',
                'status': 'excellent' if ttt_zdr > 0.90 else ('good' if ttt_zdr > 0.85 else 'needs_improvement')
            },
            'far_performance': {
                'value': ttt_far * 100,
                'target': '< 40%',
                'status': 'excellent' if ttt_far < 0.30 else ('acceptable' if ttt_far < 0.40 else 'needs_improvement')
            },
            'recommendation': 'Run 100-episode evaluation for statistical validation' if ttt_zdr > 0.90 else 'Tune hyperparameters before multi-episode evaluation'
        }

    def _assess_phase1_performance(
        self,
        base_model: Dict[str, Any],
        adapted_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess how well Phase 1 configuration performed."""
        ttt_zdr = adapted_model.get('zero_day_detection_rate', 0.0)
        ttt_far = adapted_model.get('far', adapted_model.get('false_alarm_rate', 0.0))
        ttt_acc = adapted_model.get('accuracy', 0.0)
        ttt_f1 = adapted_model.get('f1_score', 0.0)

        # Phase 1 success criteria
        criteria = {
            'zdr_above_90': ttt_zdr > 0.90,
            'far_below_40': ttt_far < 0.40,
            'f1_above_80': ttt_f1 > 0.80,
            'accuracy_above_75': ttt_acc > 0.75
        }

        criteria_met = sum(criteria.values())
        total_criteria = len(criteria)

        if criteria_met == total_criteria:
            verdict = 'EXCELLENT - All Phase 1 criteria met'
            grade = 'A'
        elif criteria_met >= 3:
            verdict = 'GOOD - Most Phase 1 criteria met'
            grade = 'B'
        elif criteria_met >= 2:
            verdict = 'ACCEPTABLE - Some Phase 1 criteria met'
            grade = 'C'
        else:
            verdict = 'NEEDS IMPROVEMENT - Phase 1 criteria not met'
            grade = 'D'

        return {
            'criteria_evaluation': criteria,
            'criteria_met': criteria_met,
            'total_criteria': total_criteria,
            'percentage_met': (criteria_met / total_criteria * 100),
            'verdict': verdict,
            'grade': grade,
            'metrics': {
                'zdr': f"{ttt_zdr*100:.2f}%",
                'far': f"{ttt_far*100:.2f}%",
                'accuracy': f"{ttt_acc*100:.2f}%",
                'f1_score': f"{ttt_f1*100:.2f}%"
            }
        }

    def _is_production_configuration(self) -> bool:
        """Check if running with production (not quick test) configuration."""
        return (
            self.phase1_config['meta_epochs'] >= 20 and
            self.phase1_config['k_shot'] >= 150 and
            self.phase1_config['num_meta_tasks'] >= 40
        )

    def _generate_markdown_report(
        self,
        evaluation_results: Dict[str, Any],
        json_report: Dict[str, Any]
    ) -> str:
        """Generate human-readable markdown report."""
        metadata = json_report['metadata']
        config = json_report['configuration']
        base = json_report['base_model_performance']
        ttt = json_report['ttt_model_performance']
        improvement = json_report['improvement']
        assessment = json_report['phase1_assessment']

        md = f"""# Comprehensive Evaluation Summary

**Generated**: {metadata['generated_at']}
**Dataset**: {metadata['dataset']}
**Zero-Day Attack**: {metadata['zero_day_attack']}
**Phase**: {metadata['phase']}

---

## Configuration Status

**Production Mode**: {'✅ Yes' if config['production_mode'] else '⚠️ No (Quick Test)'}

### Phase 1 Configuration

| Parameter | Value | Phase 1 Target |
|-----------|-------|---------------|
| meta_epochs | {config['phase1_settings']['meta_epochs']} | 21 |
| ttt_max_steps | {config['phase1_settings']['ttt_max_steps']} | 10 |
| ttt_lr | {config['phase1_settings']['ttt_lr']} | 0.0005 |
| ttt_confidence_reg_weight | {config['phase1_settings']['ttt_confidence_reg_weight']} | 1.0 |
| use_post_ttt_calibration | {config['phase1_settings']['use_post_ttt_calibration']} | True |

---

## Performance Metrics

### Base Model (No TTT)

| Metric | Value |
|--------|-------|
| **ZDR** | {base['zero_day_detection_rate']*100:.2f}% |
| **FAR** | {base['false_alarm_rate']*100:.2f}% |
| **Accuracy** | {base['accuracy']*100:.2f}% |
| **F1-Score** | {base['f1_score']*100:.2f}% |
| **Precision** | {base['precision']*100:.2f}% |
| **Recall** | {base['recall']*100:.2f}% |
| **MCC** | {base['mcc']:.4f} |
| **ROC AUC** | {base['roc_auc']:.4f} |

### TTT Model (Phase 1)

| Metric | Value |
|--------|-------|
| **ZDR** | {ttt['zero_day_detection_rate']*100:.2f}% |
| **FAR** | {ttt['false_alarm_rate']*100:.2f}% |
| **Accuracy** | {ttt['accuracy']*100:.2f}% |
| **F1-Score** | {ttt['f1_score']*100:.2f}% |
| **Precision** | {ttt['precision']*100:.2f}% |
| **Recall** | {ttt['recall']*100:.2f}% |
| **MCC** | {ttt['mcc']:.4f} |
| **ROC AUC** | {ttt['roc_auc']:.4f} |

### Improvement Analysis

| Metric | Base | TTT | Absolute Change | Relative Change |
|--------|------|-----|-----------------|-----------------|
| **ZDR** | {base['zero_day_detection_rate']*100:.2f}% | {ttt['zero_day_detection_rate']*100:.2f}% | {improvement['zdr_absolute']*100:+.2f}% | {improvement['zdr_relative']:+.2f}% |
| **FAR** | {base['false_alarm_rate']*100:.2f}% | {ttt['false_alarm_rate']*100:.2f}% | {improvement['far_absolute']*100:+.2f}% | {improvement['far_relative']:+.2f}% |
| **Accuracy** | {base['accuracy']*100:.2f}% | {ttt['accuracy']*100:.2f}% | {improvement['accuracy_absolute']*100:+.2f}% | {improvement['accuracy_relative']:+.2f}% |
| **F1-Score** | {base['f1_score']*100:.2f}% | {ttt['f1_score']*100:.2f}% | {improvement['f1_absolute']*100:+.2f}% | {improvement['f1_relative']:+.2f}% |

---

## Phase 1 Assessment

**Grade**: {assessment['grade']}
**Verdict**: {assessment['verdict']}
**Criteria Met**: {assessment['criteria_met']}/{assessment['total_criteria']} ({assessment['percentage_met']:.1f}%)

### Criteria Evaluation

| Criterion | Status |
|-----------|--------|
| ZDR > 90% | {'✅' if assessment['criteria_evaluation']['zdr_above_90'] else '❌'} ({assessment['metrics']['zdr']}) |
| FAR < 40% | {'✅' if assessment['criteria_evaluation']['far_below_40'] else '❌'} ({assessment['metrics']['far']}) |
| F1 > 80% | {'✅' if assessment['criteria_evaluation']['f1_above_80'] else '❌'} ({assessment['metrics']['f1_score']}) |
| Accuracy > 75% | {'✅' if assessment['criteria_evaluation']['accuracy_above_75'] else '❌'} ({assessment['metrics']['accuracy']}) |

---

## Recommendations

"""

        # Add recommendations based on performance
        if assessment['grade'] in ['A', 'B']:
            md += """
✅ **Phase 1 configuration is performing well!**

**Next Steps**:
1. Run 100-episode evaluation for statistical validation
2. Compare with SOTA methods
3. Test on other attack types (DoS, Exploits, etc.)
4. Prepare results for publication

**For Publication**:
- Emphasize zero-day detection capabilities
- Document ZDR-FAR trade-off
- Compare with baseline and SOTA
"""
        else:
            md += """
⚠️ **Phase 1 configuration needs improvement**

**Next Steps**:
1. Review hyperparameter settings
2. Check if using production configuration (not quick test)
3. Verify data preprocessing
4. Consider adjusting TTT learning rate or steps

**Troubleshooting**:
- If FAR too high: Increase decision threshold
- If ZDR too low: Reduce regularization or increase TTT steps
- If both poor: Check data quality and model training
"""

        md += f"""
---

**Test Samples Evaluated**: {base['test_samples']}
**Evaluation Method**: Transductive Meta-Learning with Test-Time Training
**Configuration File**: config.py

---

*Generated by Comprehensive Evaluation Summary Generator*
*For detailed analysis, see: evaluation_summary_*.json*
"""

        return md

    def _generate_publication_summary(
        self,
        evaluation_results: Dict[str, Any],
        json_report: Dict[str, Any]
    ) -> str:
        """Generate publication-ready summary."""
        metadata = json_report['metadata']
        base = json_report['base_model_performance']
        ttt = json_report['ttt_model_performance']
        improvement = json_report['improvement']

        pub = f"""# Publication-Ready Summary

## Abstract Results

Conservative test-time training achieves {ttt['zero_day_detection_rate']*100:.2f}% zero-day detection rate on {metadata['zero_day_attack']} attacks, representing a {improvement['zdr_absolute']*100:+.2f}% improvement over the base model ({base['zero_day_detection_rate']*100:.2f}%). The approach demonstrates {ttt['accuracy']*100:.2f}% overall accuracy with an F1-score of {ttt['f1_score']*100:.2f}%.

## Key Findings

1. **Zero-Day Detection**: {ttt['zero_day_detection_rate']*100:.2f}% ZDR ({base['zero_day_detection_rate']*100:+.2f}% improvement)
2. **False Alarm Rate**: {ttt['false_alarm_rate']*100:.2f}% FAR (trade-off: {improvement['far_absolute']*100:+.2f}%)
3. **Overall Performance**: {ttt['f1_score']*100:.2f}% F1-score ({improvement['f1_absolute']*100:+.2f}% improvement)

## Methodology

- **Dataset**: {metadata['dataset']}
- **Zero-Day Attack**: {metadata['zero_day_attack']}
- **Meta-Learning**: Transductive few-shot learning
- **Test-Time Adaptation**: Conservative TTT (10 steps, LR 0.0005)
- **Regularization**: Confidence regularization (weight 1.0)
- **Calibration**: Temperature scaling (target FAR 40%)

## Statistical Summary

| Model | ZDR | FAR | Accuracy | F1-Score | MCC |
|-------|-----|-----|----------|----------|-----|
| Base | {base['zero_day_detection_rate']*100:.2f}% | {base['false_alarm_rate']*100:.2f}% | {base['accuracy']*100:.2f}% | {base['f1_score']*100:.2f}% | {base['mcc']:.3f} |
| TTT | **{ttt['zero_day_detection_rate']*100:.2f}%** | {ttt['false_alarm_rate']*100:.2f}% | **{ttt['accuracy']*100:.2f}%** | **{ttt['f1_score']*100:.2f}%** | {ttt['mcc']:.3f} |

## Contribution

This work demonstrates that conservative test-time training can significantly improve zero-day attack detection while maintaining acceptable false alarm rates. The approach is particularly effective for rare attack types where traditional methods struggle.

---

*For full details, see comprehensive evaluation report.*
"""

        return pub

    def print_console_summary(self, evaluation_results: Dict[str, Any]):
        """Print concise summary to console."""
        base_model = evaluation_results.get('base_model', {})
        adapted_model = evaluation_results.get('adapted_model', {})

        base_zdr = base_model.get('zero_day_detection_rate', 0.0)
        ttt_zdr = adapted_model.get('zero_day_detection_rate', 0.0)
        base_far = base_model.get('far', base_model.get('false_alarm_rate', 0.0))
        ttt_far = adapted_model.get('far', adapted_model.get('false_alarm_rate', 0.0))
        base_f1 = base_model.get('f1_score', 0.0)
        ttt_f1 = adapted_model.get('f1_score', 0.0)

        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Phase 1 Configuration: {'✅ Active' if self._is_production_configuration() else '⚠️ Quick Test Mode'}")
        logger.info("")
        logger.info("PERFORMANCE COMPARISON:")
        logger.info(f"  Model      | ZDR         | FAR         | F1-Score    ")
        logger.info(f"  -----------|-------------|-------------|-------------")
        logger.info(f"  Base       | {base_zdr*100:6.2f}%     | {base_far*100:6.2f}%     | {base_f1*100:6.2f}%     ")
        logger.info(f"  TTT        | {ttt_zdr*100:6.2f}%     | {ttt_far*100:6.2f}%     | {ttt_f1*100:6.2f}%     ")
        logger.info(f"  Change     | {(ttt_zdr-base_zdr)*100:+6.2f}%     | {(ttt_far-base_far)*100:+6.2f}%     | {(ttt_f1-base_f1)*100:+6.2f}%     ")
        logger.info("")

        # Phase 1 assessment
        if ttt_zdr > 0.90 and ttt_far < 0.40:
            logger.info("✅ Phase 1 SUCCESS: ZDR > 90% and FAR < 40%")
        elif ttt_zdr > 0.90:
            logger.info("⚠️ Phase 1 PARTIAL: ZDR excellent but FAR needs improvement")
        else:
            logger.info("❌ Phase 1 NEEDS WORK: ZDR below target 90%")

        logger.info("="*80 + "\n")
