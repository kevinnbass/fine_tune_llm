"""Tests for DSL engine."""

import pytest
from dsl.engine import DSLEngine, Rule, CombinationRule


class TestRule:
    """Test Rule class."""
    
    def test_rule_creation(self):
        """Test rule creation and compilation."""
        rule = Rule(
            id="test_rule",
            pattern=r"\bH5N1\b",
            label="HIGH_RISK",
            confidence=0.9,
            priority=1
        )
        
        assert rule.id == "test_rule"
        assert rule.label == "HIGH_RISK"
        assert rule.confidence == 0.9
        assert rule.priority == 1
        assert rule.regex is not None
    
    def test_rule_match(self):
        """Test rule matching."""
        rule = Rule(
            id="h5n1_rule",
            pattern=r"\bH5N1\b",
            label="HIGH_RISK",
            confidence=0.9
        )
        
        assert rule.match("H5N1 outbreak detected")
        assert rule.match("The H5N1 virus is dangerous")
        assert not rule.match("H1N1 is different")
        assert not rule.match("h5n1 lowercase")  # Case sensitive
    
    def test_case_insensitive_rule(self):
        """Test case insensitive matching."""
        rule = Rule(
            id="case_rule",
            pattern=r"\bavian\b",
            label="MEDIUM_RISK",
            case_insensitive=True
        )
        
        assert rule.match("Avian flu outbreak")
        assert rule.match("AVIAN influenza")
        assert rule.match("avian birds")


class TestCombinationRule:
    """Test CombinationRule class."""
    
    def test_combination_rule_all(self):
        """Test combination rule with requires_all."""
        combo = CombinationRule(
            id="combo_test",
            requires_all=[
                {"pattern": r"\bbird\b"},
                {"pattern": r"\bflu\b"}
            ],
            label="HIGH_RISK",
            confidence=0.8
        )
        
        assert combo.match("bird flu outbreak")
        assert combo.match("flu in birds")
        assert not combo.match("bird health")
        assert not combo.match("flu season")
    
    def test_combination_rule_any(self):
        """Test combination rule with requires_any."""
        combo = CombinationRule(
            id="combo_any",
            requires_all=[{"pattern": r"\bpoultry\b"}],
            requires_any=[
                {"pattern": r"\bflu\b"},
                {"pattern": r"\bvirus\b"}
            ],
            label="MEDIUM_RISK"
        )
        
        assert combo.match("poultry flu detected")
        assert combo.match("poultry virus outbreak")
        assert not combo.match("poultry farm")
        assert not combo.match("flu season")


class TestDSLEngine:
    """Test DSL engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = DSLEngine()
        assert engine.rules == []
        assert engine.combination_rules == []
        assert engine.thresholds == {}
    
    def test_predict_with_rules(self):
        """Test prediction with simple rules."""
        engine = DSLEngine()
        
        # Add test rules
        engine.rules = [
            Rule(
                id="h5n1_rule",
                pattern=r"\bH5N1\b",
                label="HIGH_RISK",
                confidence=0.9,
                priority=1
            ),
            Rule(
                id="bird_flu_rule",
                pattern=r"\bbird\s+flu\b",
                label="MEDIUM_RISK",
                confidence=0.7,
                priority=2,
                case_insensitive=True
            )
        ]
        
        engine.thresholds = {"min_confidence_to_predict": 0.5}
        
        # Test H5N1 detection
        result = engine.predict("H5N1 outbreak reported")
        assert not result['abstain']
        assert result['probs']['HIGH_RISK'] > 0.5
        assert 'h5n1_rule' in result['matched_rules']
        
        # Test bird flu detection
        result = engine.predict("Bird flu spreads")
        assert not result['abstain']
        assert result['probs']['MEDIUM_RISK'] > 0.5
        
        # Test no match
        result = engine.predict("Weather is sunny")
        assert result['abstain']
        assert result['reason'] == "No rules matched"
    
    def test_abstention_rule(self):
        """Test abstention rules."""
        engine = DSLEngine()
        
        engine.rules = [
            Rule(
                id="safety_rule",
                pattern=r"[\x00-\x08]+",  # Control characters
                action="ABSTAIN",
                priority=0,
                description="Corrupted text"
            )
        ]
        
        result = engine.predict("Normal text")
        # Should abstain due to no other rules
        assert result['abstain']
        
        # Test with corrupted text
        result = engine.predict("Text with \x00\x01 corruption")
        assert result['abstain']
        assert "Corrupted text" in result['reason']
    
    def test_confidence_modifiers(self):
        """Test confidence modifiers."""
        engine = DSLEngine()
        
        engine.rules = [
            Rule(
                id="base_rule",
                pattern=r"\bbird\s+flu\b",
                label="HIGH_RISK",
                confidence=0.8,
                priority=2
            ),
            Rule(
                id="negation_modifier",
                pattern=r"\bnot\s+bird\s+flu\b",
                modifier=0.3,  # Reduce confidence
                priority=1
            )
        ]
        
        engine.thresholds = {"min_confidence_to_predict": 0.5}
        
        # Normal case
        result = engine.predict("bird flu outbreak")
        assert result['confidence'] == 0.8
        
        # With negation
        result = engine.predict("not bird flu outbreak")
        expected_confidence = 0.8 * 0.3  # Modified by negation
        assert abs(result['confidence'] - expected_confidence) < 0.01
    
    def test_combination_rules(self):
        """Test combination rules."""
        engine = DSLEngine()
        
        engine.combination_rules = [
            CombinationRule(
                id="research_h5n1",
                requires_all=[
                    {"pattern": r"\babstract\b"},
                    {"pattern": r"\bH5N1\b"}
                ],
                label="HIGH_RISK",
                confidence=0.95,
                priority=1
            )
        ]
        
        engine.thresholds = {"min_confidence_to_predict": 0.5}
        
        # Should match
        result = engine.predict("Abstract: H5N1 virus study")
        assert not result['abstain']
        assert result['probs']['HIGH_RISK'] > 0.9
        
        # Should not match
        result = engine.predict("H5N1 virus study")
        assert result['abstain']  # No simple rules to match
    
    def test_priority_ordering(self):
        """Test that rules are applied in priority order."""
        engine = DSLEngine()
        
        engine.rules = [
            Rule(
                id="low_priority",
                pattern=r"\bflu\b",
                label="LOW_RISK",
                confidence=0.6,
                priority=10
            ),
            Rule(
                id="high_priority",
                pattern=r"\bH5N1\b",
                label="HIGH_RISK",
                confidence=0.9,
                priority=1
            )
        ]
        
        engine.thresholds = {"min_confidence_to_predict": 0.5}
        
        # Should match high priority rule first
        result = engine.predict("H5N1 flu outbreak")
        assert result['probs']['HIGH_RISK'] > result['probs']['LOW_RISK']
        assert result['primary_rule'] == 'high_priority'
    
    def test_explain(self):
        """Test explanation functionality."""
        engine = DSLEngine()
        
        engine.rules = [
            Rule(
                id="test_rule",
                pattern=r"\btest\b",
                label="LOW_RISK",
                confidence=0.6,
                priority=1
            )
        ]
        
        engine.thresholds = {"min_confidence_to_predict": 0.5}
        
        explanation = engine.explain("test message")
        assert "Prediction: LOW_RISK" in explanation
        assert "test_rule" in explanation
        assert "conf=0.60" in explanation
    
    def test_validate_rules(self):
        """Test rule validation."""
        engine = DSLEngine()
        
        # Valid rules
        engine.rules = [
            Rule(id="rule1", pattern=r"\btest\b", label="HIGH_RISK"),
            Rule(id="rule2", pattern=r"\bother\b", label="LOW_RISK")
        ]
        
        warnings = engine.validate_rules()
        assert len(warnings) == 0
        
        # Add duplicate ID
        engine.rules.append(
            Rule(id="rule1", pattern=r"\bdupe\b", label="MEDIUM_RISK")
        )
        
        warnings = engine.validate_rules()
        assert any("Duplicate rule IDs" in w for w in warnings)
        
        # Add rule without label
        engine.rules.append(
            Rule(id="rule3", pattern=r"\bno_label\b")
        )
        
        warnings = engine.validate_rules()
        assert any("no label" in w for w in warnings)
    
    def test_threshold_enforcement(self):
        """Test confidence threshold enforcement."""
        engine = DSLEngine()
        
        engine.rules = [
            Rule(
                id="low_conf_rule",
                pattern=r"\btest\b",
                label="HIGH_RISK",
                confidence=0.3,  # Below threshold
                priority=1
            )
        ]
        
        engine.thresholds = {"min_confidence_to_predict": 0.5}
        
        result = engine.predict("test message")
        assert result['abstain']
        assert "below threshold" in result['reason']


def test_create_default_engine():
    """Test default engine creation."""
    from dsl.engine import create_default_engine
    
    engine = create_default_engine()
    
    assert len(engine.rules) > 0
    assert engine.thresholds['min_confidence_to_predict'] == 0.5
    
    # Test with sample text
    result = engine.predict("H5N1 detected")
    assert not result['abstain']
    assert result['probs']['HIGH_RISK'] > 0.8