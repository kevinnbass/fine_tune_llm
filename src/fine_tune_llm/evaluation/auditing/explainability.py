"""
Explainability and reasoning chain generation for model predictions.

This module provides tools for generating and verifying explainable
reasoning chains for high-stakes model predictions.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
import re

from ...core.interfaces import BaseComponent
from ...core.exceptions import InferenceError

logger = logging.getLogger(__name__)

class ExplainableReasoning(BaseComponent):
    """Generate and verify explainable reasoning for predictions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize explainable reasoning component."""
        self.config = config or {}
        self.reasoning_config = self.config.get('high_stakes', {}).get('reasoning', {})
        self.require_stepwise = self.reasoning_config.get('require_stepwise', True)
        self.min_reasoning_steps = self.reasoning_config.get('min_steps', 2)
        self.max_reasoning_steps = self.reasoning_config.get('max_steps', 10)
        self.confidence_threshold = self.reasoning_config.get('confidence_threshold', 0.7)
        
        # Reasoning patterns
        self.reasoning_patterns = {
            'step': r'(?:step\s*\d+|first|second|third|next|then|finally)',
            'because': r'(?:because|since|as|due to|given that)',
            'therefore': r'(?:therefore|thus|hence|so|consequently)',
            'evidence': r'(?:evidence|shows|indicates|suggests|proves)',
        }
        
        self.reasoning_log = []
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        self.reasoning_config = self.config.get('high_stakes', {}).get('reasoning', {})
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.reasoning_log.clear()
    
    @property
    def name(self) -> str:
        """Component name."""
        return "ExplainableReasoning"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def generate_reasoning_chain(self, 
                                input_text: str,
                                prediction: str,
                                confidence: float) -> Dict[str, Any]:
        """
        Generate explainable reasoning chain for a prediction.
        
        Args:
            input_text: Input text
            prediction: Model prediction
            confidence: Prediction confidence
            
        Returns:
            Dictionary containing reasoning chain and metadata
        """
        reasoning_chain = {
            'input': input_text[:500],  # Truncate for logging
            'prediction': prediction,
            'confidence': confidence,
            'reasoning_steps': [],
            'is_valid': False,
            'explanation_score': 0.0
        }
        
        try:
            # Extract reasoning steps from prediction if it contains explanation
            if isinstance(prediction, str) and len(prediction) > 50:
                steps = self._extract_reasoning_steps(prediction)
                reasoning_chain['reasoning_steps'] = steps
                
                # Validate reasoning
                is_valid = self._validate_reasoning(steps)
                reasoning_chain['is_valid'] = is_valid
                
                # Score explanation quality
                score = self._score_explanation(prediction, steps)
                reasoning_chain['explanation_score'] = score
                
                # Check confidence alignment
                if confidence < self.confidence_threshold and not steps:
                    reasoning_chain['warning'] = "Low confidence prediction lacks reasoning"
                
            else:
                # Generate synthetic reasoning if not provided
                synthetic_steps = self._generate_synthetic_reasoning(input_text, prediction)
                reasoning_chain['reasoning_steps'] = synthetic_steps
                reasoning_chain['synthetic'] = True
            
            # Log reasoning chain
            self.reasoning_log.append(reasoning_chain)
            
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Error generating reasoning chain: {e}")
            reasoning_chain['error'] = str(e)
            return reasoning_chain
    
    def _extract_reasoning_steps(self, text: str) -> List[Dict[str, str]]:
        """Extract reasoning steps from text."""
        steps = []
        
        # Split by common reasoning markers
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            step_info = {
                'step_number': i + 1,
                'content': sentence,
                'type': 'unknown'
            }
            
            # Classify step type
            sentence_lower = sentence.lower()
            
            if any(re.search(pattern, sentence_lower) for pattern in self.reasoning_patterns['step'].split('|')):
                step_info['type'] = 'sequential'
            elif re.search(self.reasoning_patterns['because'], sentence_lower):
                step_info['type'] = 'causal'
            elif re.search(self.reasoning_patterns['therefore'], sentence_lower):
                step_info['type'] = 'conclusion'
            elif re.search(self.reasoning_patterns['evidence'], sentence_lower):
                step_info['type'] = 'evidence'
            
            steps.append(step_info)
        
        return steps
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from reasoning text."""
        # Look for conclusion markers
        conclusion_patterns = [
            r'(?:therefore|thus|in conclusion|finally),?\s*(.+)',
            r'(?:the answer is|my answer is):?\s*(.+)',
            r'(?:final answer):?\s*(.+)',
        ]
        
        text_lower = text.lower()
        for pattern in conclusion_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        
        # If no explicit conclusion, take last sentence
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            return sentences[-1].strip()
        
        return None
    
    def verify_faithfulness(self, 
                          reasoning_chain: Dict[str, Any],
                          ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify faithfulness of reasoning to prediction.
        
        Args:
            reasoning_chain: Generated reasoning chain
            ground_truth: Optional ground truth for comparison
            
        Returns:
            Verification results
        """
        verification = {
            'is_faithful': True,
            'consistency_score': 0.0,
            'issues': []
        }
        
        try:
            steps = reasoning_chain.get('reasoning_steps', [])
            prediction = reasoning_chain.get('prediction', '')
            
            if not steps:
                verification['is_faithful'] = False
                verification['issues'].append("No reasoning steps provided")
                return verification
            
            # Check if conclusion follows from reasoning
            final_step = steps[-1] if steps else None
            if final_step and final_step.get('type') != 'conclusion':
                verification['issues'].append("Reasoning lacks clear conclusion")
            
            # Check reasoning consistency
            consistency_score = self._check_consistency(steps)
            verification['consistency_score'] = consistency_score
            
            if consistency_score < 0.5:
                verification['is_faithful'] = False
                verification['issues'].append("Inconsistent reasoning detected")
            
            # Compare with ground truth if available
            if ground_truth:
                prediction_matches = self._compare_answers(prediction, ground_truth)
                if not prediction_matches:
                    verification['issues'].append("Prediction doesn't match ground truth")
            
            return verification
            
        except Exception as e:
            logger.error(f"Error verifying faithfulness: {e}")
            verification['error'] = str(e)
            verification['is_faithful'] = False
            return verification
    
    def _validate_reasoning(self, steps: List[Dict[str, str]]) -> bool:
        """Validate reasoning steps."""
        if not steps:
            return False
        
        # Check minimum steps requirement
        if self.require_stepwise and len(steps) < self.min_reasoning_steps:
            return False
        
        # Check for maximum steps
        if len(steps) > self.max_reasoning_steps:
            return False
        
        # Check for conclusion
        has_conclusion = any(step.get('type') == 'conclusion' for step in steps)
        
        return has_conclusion or len(steps) >= self.min_reasoning_steps
    
    def _score_explanation(self, text: str, steps: List[Dict[str, str]]) -> float:
        """Score the quality of explanation."""
        score = 0.0
        
        # Score based on number of steps
        if steps:
            step_score = min(len(steps) / self.min_reasoning_steps, 1.0) * 0.3
            score += step_score
        
        # Score based on reasoning markers
        text_lower = text.lower() if text else ""
        
        for pattern_type, pattern in self.reasoning_patterns.items():
            if re.search(pattern, text_lower):
                score += 0.15
        
        # Score based on step diversity
        if steps:
            step_types = set(step.get('type', 'unknown') for step in steps)
            diversity_score = len(step_types) / 4.0 * 0.2  # Max 4 types
            score += diversity_score
        
        return min(score, 1.0)
    
    def _generate_synthetic_reasoning(self, input_text: str, prediction: str) -> List[Dict[str, str]]:
        """Generate synthetic reasoning when not provided."""
        steps = []
        
        # Step 1: Analyze input
        steps.append({
            'step_number': 1,
            'content': f"Analyzing the input: {input_text[:100]}...",
            'type': 'sequential'
        })
        
        # Step 2: Apply logic
        steps.append({
            'step_number': 2,
            'content': "Applying classification logic based on learned patterns",
            'type': 'sequential'
        })
        
        # Step 3: Conclusion
        steps.append({
            'step_number': 3,
            'content': f"Therefore, the prediction is: {prediction}",
            'type': 'conclusion'
        })
        
        return steps
    
    def _check_consistency(self, steps: List[Dict[str, str]]) -> float:
        """Check consistency of reasoning steps."""
        if not steps:
            return 0.0
        
        # Simple heuristic: check if steps build on each other
        consistency_score = 1.0
        
        # Check for contradictions (simplified)
        contents = [step.get('content', '').lower() for step in steps]
        
        contradiction_pairs = [
            ('not', 'definitely'),
            ('unlikely', 'certainly'),
            ('false', 'true'),
        ]
        
        for content in contents:
            for neg, pos in contradiction_pairs:
                if neg in content and pos in content:
                    consistency_score -= 0.2
        
        return max(consistency_score, 0.0)
    
    def _compare_answers(self, prediction: str, ground_truth: str) -> bool:
        """Compare predicted answer with ground truth."""
        if not prediction or not ground_truth:
            return False
        
        # Normalize for comparison
        pred_normalized = prediction.lower().strip()
        truth_normalized = ground_truth.lower().strip()
        
        # Exact match
        if pred_normalized == truth_normalized:
            return True
        
        # Partial match (contains)
        if truth_normalized in pred_normalized or pred_normalized in truth_normalized:
            return True
        
        return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate explainability audit report."""
        report = {
            'summary': {
                'total_explanations': len(self.reasoning_log),
                'valid_reasoning': sum(1 for r in self.reasoning_log if r.get('is_valid', False)),
                'average_score': sum(r.get('explanation_score', 0) for r in self.reasoning_log) / max(len(self.reasoning_log), 1),
                'synthetic_count': sum(1 for r in self.reasoning_log if r.get('synthetic', False))
            },
            'detailed_log': self.reasoning_log[-100:],  # Last 100 entries
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on reasoning analysis."""
        recommendations = []
        
        if self.reasoning_log:
            valid_count = sum(1 for r in self.reasoning_log if r.get('is_valid', False))
            valid_ratio = valid_count / len(self.reasoning_log)
            
            if valid_ratio < 0.5:
                recommendations.append(
                    "Low ratio of valid reasoning chains. Consider enhancing model's explanation capabilities."
                )
            
            avg_score = sum(r.get('explanation_score', 0) for r in self.reasoning_log) / len(self.reasoning_log)
            if avg_score < 0.6:
                recommendations.append(
                    "Low average explanation quality. Consider training with more detailed reasoning examples."
                )
        
        if not recommendations:
            recommendations.append("Reasoning quality is satisfactory. Continue monitoring.")
        
        return recommendations