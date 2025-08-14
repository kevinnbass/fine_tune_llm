"""DSL engine for regex-based classification rules."""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """Single classification rule."""
    id: str
    pattern: str
    label: Optional[str] = None
    confidence: float = 1.0
    priority: int = 100
    case_insensitive: bool = False
    action: Optional[str] = None  # e.g., ABSTAIN
    modifier: Optional[float] = None  # Confidence modifier
    max_distance: Optional[int] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Compile regex pattern."""
        flags = re.IGNORECASE if self.case_insensitive else 0
        self.regex = re.compile(self.pattern, flags)
    
    def match(self, text: str) -> bool:
        """Check if rule matches text."""
        return bool(self.regex.search(text))


@dataclass
class CombinationRule:
    """Rule requiring multiple patterns."""
    id: str
    requires_all: List[Dict[str, Any]]
    requires_any: Optional[List[Dict[str, Any]]] = None
    label: str = None
    confidence: float = 1.0
    priority: int = 100
    
    def __post_init__(self):
        """Compile all patterns."""
        self.all_patterns = []
        for pattern_dict in self.requires_all:
            pattern = pattern_dict.get('pattern', pattern_dict)
            flags = re.IGNORECASE if pattern_dict.get('case_insensitive', False) else 0
            self.all_patterns.append(re.compile(pattern, flags))
        
        self.any_patterns = []
        if self.requires_any:
            for pattern_dict in self.requires_any:
                pattern = pattern_dict.get('pattern', pattern_dict)
                flags = re.IGNORECASE if pattern_dict.get('case_insensitive', False) else 0
                self.any_patterns.append(re.compile(pattern, flags))
    
    def match(self, text: str) -> bool:
        """Check if combination rule matches."""
        # All patterns must match
        if not all(regex.search(text) for regex in self.all_patterns):
            return False
        
        # If any_patterns exist, at least one must match
        if self.any_patterns:
            return any(regex.search(text) for regex in self.any_patterns)
        
        return True


class DSLEngine:
    """Engine for evaluating DSL rules."""
    
    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize DSL engine.
        
        Args:
            rules_path: Path to rules YAML file
        """
        self.rules: List[Rule] = []
        self.combination_rules: List[CombinationRule] = []
        self.thresholds: Dict[str, float] = {}
        
        if rules_path:
            self.load_rules(rules_path)
    
    def load_rules(self, rules_path: str):
        """Load rules from YAML file."""
        with open(rules_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load simple rules
        for rule_dict in config.get('rules', []):
            rule = Rule(**rule_dict)
            self.rules.append(rule)
        
        # Load combination rules
        for combo_dict in config.get('combinations', []):
            combo_rule = CombinationRule(**combo_dict)
            self.combination_rules.append(combo_rule)
        
        # Load thresholds
        self.thresholds = config.get('thresholds', {})
        
        # Sort by priority (lower number = higher priority)
        self.rules.sort(key=lambda r: r.priority)
        self.combination_rules.sort(key=lambda r: r.priority)
        
        logger.info(f"Loaded {len(self.rules)} rules and {len(self.combination_rules)} combination rules")
    
    def predict(self, text: str, return_all_matches: bool = False) -> Dict[str, Any]:
        """
        Apply rules to text and return prediction.
        
        Args:
            text: Input text to classify
            return_all_matches: Whether to return all matching rules
            
        Returns:
            Dictionary with prediction results
        """
        matches = []
        modifiers = []
        
        # Check simple rules
        for rule in self.rules:
            if rule.match(text):
                if rule.action == 'ABSTAIN':
                    # Immediate abstention
                    return {
                        'probs': {},
                        'abstain': True,
                        'reason': rule.description or f"Rule {rule.id} triggered abstention",
                        'matched_rules': [rule.id]
                    }
                
                if rule.modifier is not None:
                    modifiers.append((rule.id, rule.modifier))
                else:
                    matches.append({
                        'rule_id': rule.id,
                        'label': rule.label,
                        'confidence': rule.confidence,
                        'priority': rule.priority
                    })
                
                if not return_all_matches and rule.label:
                    break  # Use first matching rule
        
        # Check combination rules
        for combo_rule in self.combination_rules:
            if combo_rule.match(text):
                matches.append({
                    'rule_id': combo_rule.id,
                    'label': combo_rule.label,
                    'confidence': combo_rule.confidence,
                    'priority': combo_rule.priority
                })
                
                if not return_all_matches:
                    break
        
        if not matches:
            # No rules matched - abstain
            return {
                'probs': {},
                'abstain': True,
                'reason': "No rules matched",
                'matched_rules': []
            }
        
        # Get highest priority match
        best_match = min(matches, key=lambda m: m['priority'])
        confidence = best_match['confidence']
        
        # Apply modifiers
        for mod_id, modifier in modifiers:
            confidence *= modifier
            logger.debug(f"Applied modifier {mod_id}: confidence now {confidence}")
        
        # Check confidence threshold
        min_conf = self.thresholds.get('min_confidence_to_predict', 0.5)
        if confidence < min_conf:
            return {
                'probs': {},
                'abstain': True,
                'reason': f"Confidence {confidence} below threshold {min_conf}",
                'matched_rules': [m['rule_id'] for m in matches]
            }
        
        # Build probability distribution
        # Simple heuristic: assign confidence to predicted class, 
        # distribute remainder among other classes
        label = best_match['label']
        labels = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
        
        probs = {}
        remaining = 1.0 - confidence
        other_labels = [l for l in labels if l != label]
        
        probs[label] = confidence
        for other_label in other_labels:
            probs[other_label] = remaining / len(other_labels)
        
        result = {
            'probs': probs,
            'abstain': False,
            'matched_rules': [m['rule_id'] for m in matches],
            'confidence': confidence,
            'primary_rule': best_match['rule_id']
        }
        
        if return_all_matches:
            result['all_matches'] = matches
        
        return result
    
    def explain(self, text: str) -> str:
        """
        Explain why a particular prediction was made.
        
        Args:
            text: Input text
            
        Returns:
            Human-readable explanation
        """
        result = self.predict(text, return_all_matches=True)
        
        if result['abstain']:
            return f"ABSTAINED: {result['reason']}"
        
        matches = result.get('all_matches', [])
        if not matches:
            return "No rules matched"
        
        explanation = []
        explanation.append(f"Prediction: {max(result['probs'], key=result['probs'].get)}")
        explanation.append(f"Confidence: {result['confidence']:.2f}")
        explanation.append(f"Primary rule: {result['primary_rule']}")
        explanation.append("\nAll matching rules:")
        
        for match in matches:
            explanation.append(f"  - {match['rule_id']}: {match['label']} (conf={match['confidence']:.2f}, priority={match['priority']})")
        
        return "\n".join(explanation)
    
    def validate_rules(self) -> List[str]:
        """
        Validate loaded rules for common issues.
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check for duplicate IDs
        rule_ids = [r.id for r in self.rules] + [r.id for r in self.combination_rules]
        if len(rule_ids) != len(set(rule_ids)):
            warnings.append("Duplicate rule IDs detected")
        
        # Check for rules without labels or actions
        for rule in self.rules:
            if not rule.label and not rule.action and rule.modifier is None:
                warnings.append(f"Rule {rule.id} has no label, action, or modifier")
        
        # Check for overlapping priorities
        priorities = [r.priority for r in self.rules]
        if len(priorities) != len(set(priorities)):
            warnings.append("Multiple rules with same priority (may cause non-deterministic behavior)")
        
        # Check regex validity (already compiled in __post_init__, so just check for common issues)
        for rule in self.rules:
            if '.*.*' in rule.pattern:
                warnings.append(f"Rule {rule.id} has redundant .*.* in pattern")
            if rule.pattern.startswith('*') or rule.pattern.startswith('+'):
                warnings.append(f"Rule {rule.id} pattern starts with quantifier")
        
        return warnings


def create_default_engine() -> DSLEngine:
    """Create engine with default rules."""
    engine = DSLEngine()
    
    # Add some default rules programmatically
    engine.rules = [
        Rule(
            id='high_risk_h5n1',
            pattern=r'\bH5N1\b',
            label='HIGH_RISK',
            confidence=0.95,
            priority=1
        ),
        Rule(
            id='safety_corrupted',
            pattern=r'[\x00-\x08\x0B-\x0C\x0E-\x1F]{5,}',
            action='ABSTAIN',
            priority=0,
            description='Corrupted text detected'
        ),
        Rule(
            id='default_unknown',
            pattern=r'.*',  # Matches everything
            label='NO_RISK',
            confidence=0.3,
            priority=1000  # Lowest priority
        )
    ]
    
    engine.thresholds = {
        'min_confidence_to_predict': 0.5,
        'high_confidence': 0.8
    }
    
    return engine