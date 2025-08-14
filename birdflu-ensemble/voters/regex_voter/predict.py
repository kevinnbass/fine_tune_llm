"""Regex voter wrapper for DSL engine."""

import time
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ...dsl.engine import DSLEngine

logger = logging.getLogger(__name__)


class RegexVoter:
    """Regex-based voter using DSL engine."""
    
    def __init__(
        self,
        rules_path: str = "dsl/rules.yaml",
        cost_cents: float = 0.0001,
        timeout_ms: float = 10.0
    ):
        """
        Initialize regex voter.
        
        Args:
            rules_path: Path to DSL rules YAML
            cost_cents: Cost per prediction
            timeout_ms: Timeout in milliseconds
        """
        self.rules_path = rules_path
        self.cost_cents = cost_cents
        self.timeout_ms = timeout_ms
        
        # Initialize DSL engine
        self.engine = DSLEngine(rules_path)
        
        # Validate rules on initialization
        warnings = self.engine.validate_rules()
        if warnings:
            for warning in warnings:
                logger.warning(f"Rule validation warning: {warning}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make prediction using regex rules.
        
        Args:
            text: Input text
            
        Returns:
            Voter output dictionary
        """
        start_time = time.time()
        
        # Apply DSL rules
        result = self.engine.predict(text)
        
        # Format output
        latency = (time.time() - start_time) * 1000
        
        # Check timeout
        if latency > self.timeout_ms:
            logger.warning(f"Regex voter exceeded timeout: {latency:.1f}ms > {self.timeout_ms}ms")
        
        # Build standard voter output
        output = {
            'probs': result.get('probs', {}),
            'abstain': result.get('abstain', False),
            'latency_ms': latency,
            'cost_cents': self.cost_cents,
            'model_id': 'regex_dsl'
        }
        
        # Add additional information
        if result.get('abstain'):
            output['reason'] = result.get('reason', 'No rules matched')
        else:
            output['decision'] = max(result['probs'], key=result['probs'].get)
            output['max_prob'] = max(result['probs'].values())
            output['matched_rules'] = result.get('matched_rules', [])
            output['primary_rule'] = result.get('primary_rule')
        
        return output
    
    def batch_predict(self, texts: list) -> list:
        """
        Make predictions for batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of voter outputs
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def reload_rules(self, rules_path: Optional[str] = None):
        """
        Reload rules from file.
        
        Args:
            rules_path: Optional new rules path
        """
        if rules_path:
            self.rules_path = rules_path
        
        self.engine.load_rules(self.rules_path)
        
        # Validate new rules
        warnings = self.engine.validate_rules()
        if warnings:
            for warning in warnings:
                logger.warning(f"Rule validation warning: {warning}")
        
        logger.info(f"Reloaded rules from {self.rules_path}")
    
    def explain_prediction(self, text: str) -> str:
        """
        Get explanation for prediction.
        
        Args:
            text: Input text
            
        Returns:
            Human-readable explanation
        """
        return self.engine.explain(text)
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded rules.
        
        Returns:
            Dictionary with rules information
        """
        summary = {
            'n_rules': len(self.engine.rules),
            'n_combination_rules': len(self.engine.combination_rules),
            'thresholds': self.engine.thresholds,
            'rules': []
        }
        
        # Add rule details
        for rule in self.engine.rules[:10]:  # Show first 10 rules
            summary['rules'].append({
                'id': rule.id,
                'label': rule.label,
                'action': rule.action,
                'priority': rule.priority,
                'description': rule.description
            })
        
        return summary