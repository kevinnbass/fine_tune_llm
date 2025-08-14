"""
Facade pattern for backward compatibility with original high_stakes_audit.py.

This module provides a compatibility layer that maintains the original API
while delegating to the new decomposed components.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import new decomposed components
from src.fine_tune_llm.evaluation.auditing import (
    BiasAuditor,
    ExplainableReasoning,
    ProceduralAlignment,
    VerifiableTraining,
    AdvancedHighStakesAuditor as NewAuditor
)

# Deprecation warning
warnings.warn(
    "high_stakes_audit.py has been decomposed into multiple modules. "
    "Please update your imports to use the new modular components from "
    "src.fine_tune_llm.evaluation.auditing/. This facade will be removed in v3.0.0",
    DeprecationWarning,
    stacklevel=2
)

class AdvancedHighStakesAuditor:
    """
    Backward compatibility facade for the original AdvancedHighStakesAuditor.
    
    This class maintains the original API while delegating to new components.
    All original methods are preserved for backward compatibility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with backward compatibility."""
        self._new_auditor = NewAuditor(config)
        self.config = self._new_auditor.config
        
        # Expose sub-components for compatibility
        self.bias_auditor = self._new_auditor.bias_auditor
        self.explainable_reasoning = self._new_auditor.explainable_reasoning
        self.procedural_alignment = self._new_auditor.procedural_alignment
        self.verifiable_training = self._new_auditor.verifiable_training
        
        # Expose configuration attributes
        self.enable_advanced_metrics = self._new_auditor.enable_advanced_metrics
        self.risk_config = self._new_auditor.risk_config
        self.risk_threshold = self._new_auditor.risk_threshold
        self.cost_matrix = self._new_auditor.cost_matrix
        self.coverage_config = self._new_auditor.coverage_config
        self.target_coverage = self._new_auditor.target_coverage
        self.calibration_bins = self._new_auditor.calibration_bins
        
        # Expose audit results
        self.audit_results = self._new_auditor.audit_results
    
    def run_comprehensive_audit(self, *args, **kwargs):
        """Delegate to new auditor's conduct_comprehensive_audit method."""
        return self._new_auditor.conduct_comprehensive_audit(*args, **kwargs)
    
    def conduct_comprehensive_audit(self, *args, **kwargs):
        """Direct delegation to new implementation."""
        return self._new_auditor.conduct_comprehensive_audit(*args, **kwargs)
    
    # Preserve all original private methods for compatibility
    def _conduct_bias_audit(self, *args, **kwargs):
        return self._new_auditor._conduct_bias_audit(*args, **kwargs)
    
    def _conduct_procedural_audit(self, *args, **kwargs):
        return self._new_auditor._conduct_procedural_audit(*args, **kwargs)
    
    def _conduct_explainability_audit(self, *args, **kwargs):
        return self._new_auditor._conduct_explainability_audit(*args, **kwargs)
    
    def _conduct_advanced_metrics_audit(self, *args, **kwargs):
        return self._new_auditor._conduct_advanced_metrics_audit(*args, **kwargs)
    
    def _conduct_risk_assessment_audit(self, *args, **kwargs):
        return self._new_auditor._conduct_risk_assessment_audit(*args, **kwargs)
    
    def _conduct_coverage_calibration_audit(self, *args, **kwargs):
        return self._new_auditor._conduct_coverage_calibration_audit(*args, **kwargs)
    
    def _calculate_misclassification_risk(self, *args, **kwargs):
        return self._new_auditor._calculate_misclassification_risk(*args, **kwargs)
    
    def _label_to_index(self, *args, **kwargs):
        return self._new_auditor._label_to_index(*args, **kwargs)
    
    def _generate_overall_assessment(self, *args, **kwargs):
        return self._new_auditor._generate_overall_assessment(*args, **kwargs)
    
    def _generate_recommendations(self, *args, **kwargs):
        return self._new_auditor._generate_recommendations(*args, **kwargs)
    
    # Add any other methods that might be used externally
    def __getattr__(self, name):
        """Forward any undefined attributes to the new auditor."""
        return getattr(self._new_auditor, name)


# Export original class names for full backward compatibility
__all__ = ['AdvancedHighStakesAuditor']

# Create module-level shortcuts for common imports
def create_auditor(config: Optional[Dict[str, Any]] = None) -> AdvancedHighStakesAuditor:
    """Factory function for creating auditor (backward compatibility)."""
    return AdvancedHighStakesAuditor(config)