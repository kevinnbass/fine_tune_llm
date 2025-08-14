#!/usr/bin/env python
"""Evaluate the full ensemble system."""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Any
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from voters.regex_voter.predict import RegexVoter
from voters.classical.tfidf_lr import TfidfLRVoter
from voters.classical.tfidf_svm import TfidfSVMVoter
from voters.ws_label_model.snorkel_like import LabelModelVoter
from voters.llm.infer import LLMVoterInference
from arbiter.stacker_lr import LogisticRegressionStacker
from arbiter.conformal import ConformalPredictor
from eval.metrics import EnsembleEvaluator, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleSystem:
    """Full ensemble system for evaluation."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ensemble system.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        
        # Load configurations
        with open(self.config_dir / "voters.yaml") as f:
            self.voters_config = yaml.safe_load(f)['voters']
        
        with open(self.config_dir / "conformal.yaml") as f:
            self.conformal_config = yaml.safe_load(f)
        
        # Initialize voters
        self.voters = self._initialize_voters()
        
        # Initialize arbiter components
        self.stacker = LogisticRegressionStacker()
        self.conformal = ConformalPredictor(str(self.config_dir / "conformal.yaml"))
        
        # Initialize evaluator
        self.evaluator = EnsembleEvaluator()
        
    def _initialize_voters(self) -> Dict[str, Any]:
        """Initialize enabled voters."""
        voters = {}
        
        # Regex voter
        if self.voters_config['regex_dsl']['enabled']:
            voters['regex_dsl'] = RegexVoter()
        
        # Classical voters
        if self.voters_config['tfidf_lr']['enabled']:
            voter = TfidfLRVoter()
            model_path = self.voters_config['tfidf_lr']['model_path']
            if Path(model_path).exists():
                voter.load(model_path)
                voters['tfidf_lr'] = voter
        
        if self.voters_config['tfidf_svm']['enabled']:
            voter = TfidfSVMVoter()
            model_path = self.voters_config['tfidf_svm']['model_path']
            if Path(model_path).exists():
                voter.load(model_path)
                voters['tfidf_svm'] = voter
        
        # Label model voter
        if self.voters_config['ws_label_model']['enabled']:
            voter = LabelModelVoter()
            model_path = self.voters_config['ws_label_model']['model_path']
            if Path(model_path).exists():
                voter.load(model_path)
                voters['ws_label_model'] = voter
        
        # LLM voter
        if self.voters_config['llm_lora']['enabled']:
            model_path = self.voters_config['llm_lora']['adapter_path']
            if Path(model_path).exists():
                voter = LLMVoterInference(model_path)
                voters['llm_lora'] = voter
        
        logger.info(f"Initialized {len(voters)} voters: {list(voters.keys())}")
        return voters
    
    def run_voters(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Run all voters on a single text.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Dictionary of voter outputs
        """
        voter_outputs = {}
        
        for voter_id, voter in self.voters.items():
            try:
                if voter_id == 'llm_lora':
                    output = voter.predict(text, metadata)
                elif voter_id == 'ws_label_model':
                    # Label model needs LF votes - simplified for demo
                    output = {'probs': {}, 'abstain': True, 'reason': 'No LF votes'}
                else:
                    output = voter.predict(text)
                
                voter_outputs[voter_id] = output
            except Exception as e:
                logger.error(f"Voter {voter_id} failed: {e}")
                voter_outputs[voter_id] = {
                    'probs': {},
                    'abstain': True,
                    'reason': f'Error: {str(e)}'
                }
        
        return voter_outputs
    
    def predict_with_cascade(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        slice_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction using full cascade.
        
        Args:
            text: Input text
            metadata: Optional metadata
            slice_name: Optional slice name
            
        Returns:
            Cascade prediction result
        """
        # Run voters
        voter_outputs = self.run_voters(text, metadata)
        
        # Tier 0: Fast rules
        tier0_result = self.stacker.apply_tier0_rules(
            voter_outputs,
            consensus_threshold=self.voters_config['cascade']['tier_0_consensus_threshold']
        )
        
        if tier0_result:
            return tier0_result
        
        # Tier 1: Stacker
        if hasattr(self.stacker, 'model') and self.stacker.model is not None:
            stacker_result = self.stacker.predict(voter_outputs, text, metadata)
            probs = stacker_result['probs']
            
            # Convert to numpy array
            class_names = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
            probs_array = np.array([probs[c] for c in class_names])
            
            # Tier 2: Conformal prediction
            if self.conformal.global_threshold is not None:
                conformal_result = self.conformal.predict(probs_array, slice_name)
                return conformal_result
            else:
                # No conformal calibration - use stacker result
                stacker_result['tier'] = 1
                return stacker_result
        else:
            # No stacker model - use simple voting
            return self._simple_voting(voter_outputs)
    
    def _simple_voting(self, voter_outputs: Dict[str, Dict]) -> Dict[str, Any]:
        """Simple majority voting fallback."""
        predictions = []
        
        for output in voter_outputs.values():
            if not output.get('abstain'):
                probs = output.get('probs', {})
                if probs:
                    pred = max(probs, key=probs.get)
                    predictions.append(pred)
        
        if not predictions:
            return {
                'abstain': True,
                'reason': 'All voters abstained',
                'tier': 1
            }
        
        from collections import Counter
        vote_counts = Counter(predictions)
        decision = vote_counts.most_common(1)[0][0]
        
        return {
            'decision': decision,
            'confidence': vote_counts[decision] / len(predictions),
            'tier': 1,
            'abstain': False
        }
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        safety_set_only: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate system on test data.
        
        Args:
            test_data: Test dataframe with 'text' and 'label' columns
            safety_set_only: Whether to evaluate only on safety set
            
        Returns:
            Evaluation results
        """
        if safety_set_only:
            # Filter to safety set if marked
            if 'is_safety' in test_data.columns:
                test_data = test_data[test_data['is_safety']]
                logger.info(f"Evaluating on safety set: {len(test_data)} samples")
        
        predictions = []
        y_true = []
        y_pred = []
        
        class_names = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
        
        for _, row in test_data.iterrows():
            # Get prediction
            result = self.predict_with_cascade(
                row['text'],
                metadata=row.get('metadata'),
                slice_name=row.get('slice')
            )
            
            predictions.append(result)
            
            # Get true label
            true_label = row['label']
            if isinstance(true_label, str):
                true_label = class_names.index(true_label)
            y_true.append(true_label)
            
            # Get predicted label
            if result.get('abstain'):
                y_pred.append(-1)  # Abstention marker
            else:
                pred_label = result['decision']
                if pred_label in class_names:
                    y_pred.append(class_names.index(pred_label))
                else:
                    y_pred.append(-1)
        
        # Convert to arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute metrics
        results = {}
        
        # Overall metrics (excluding abstentions)
        non_abstain_mask = y_pred != -1
        if non_abstain_mask.sum() > 0:
            results['overall'] = self.evaluator.evaluate_predictions(
                y_true[non_abstain_mask],
                y_pred[non_abstain_mask]
            )
        
        # Cascade metrics
        results['cascade'] = self.evaluator.evaluate_cascade(predictions, y_true)
        
        # Slice metrics if available
        if 'slice' in test_data.columns:
            slice_membership = {}
            for slice_name in test_data['slice'].unique():
                if pd.notna(slice_name):
                    slice_membership[slice_name] = (test_data['slice'] == slice_name).values
            
            if slice_membership:
                slice_results = self.evaluator.evaluate_with_slices(
                    y_true[non_abstain_mask],
                    y_pred[non_abstain_mask],
                    {k: v[non_abstain_mask] for k, v in slice_membership.items()}
                )
                results.update(slice_results)
        
        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate full ensemble system")
    parser.add_argument("--test_data", required=True, help="Test data CSV path")
    parser.add_argument("--config_dir", default="configs", help="Configuration directory")
    parser.add_argument("--safety_set_only", action="store_true", help="Evaluate only on safety set")
    parser.add_argument("--output", default="eval_results.json", help="Output file for results")
    parser.add_argument("--report", default="eval_report.md", help="Output file for report")
    
    args = parser.parse_args()
    
    # Load test data
    test_data = pd.read_csv(args.test_data)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize system
    system = EnsembleSystem(args.config_dir)
    
    # Evaluate
    results = system.evaluate(test_data, args.safety_set_only)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {args.output}")
    
    # Generate report
    report = system.evaluator.generate_report(results)
    with open(args.report, 'w') as f:
        f.write(report)
    logger.info(f"Saved report to {args.report}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if 'overall' in results:
        print(f"Overall Accuracy: {results['overall']['accuracy']:.3f}")
        print(f"Overall F1 (weighted): {results['overall']['f1_weighted']:.3f}")
    
    if 'cascade' in results:
        cascade = results['cascade']
        print(f"Coverage: {cascade['coverage']:.1%}")
        print(f"Abstention Rate: {cascade['abstention_rate']:.1%}")
        print(f"LLM Call Rate: {cascade['llm_call_rate']:.1%}")
        print(f"Mean Cost (cents): {cascade['mean_cost_cents']:.4f}")
        print(f"P95 Latency (ms): {cascade['p95_latency_ms']:.1f}")
    
    if 'worst_slice' in results:
        worst = results['worst_slice']
        print(f"Worst Slice: {worst['name']} (F1: {worst['f1']:.3f})")


if __name__ == "__main__":
    main()