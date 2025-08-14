"""
LLM fine-tuning modules with high-stakes precision features.
"""

# Lazy imports to avoid heavy dependencies during testing
__all__ = [
    # Dataset functions
    "load_labels",
    "build_examples",
    "create_abstention_examples",
    "create_balanced_dataset",
    "validate_output_format",
    
    # Training
    "LoRATrainer",
    
    # Inference
    "LLMVoterInference",
    
    # Evaluation
    "ModelEvaluator",
    
    # Uncertainty
    "MCDropoutWrapper",
    "compute_uncertainty_aware_loss",
    "should_abstain",
    
    # Fact checking
    "FactChecker",
    "create_factual_test_data",
    
    # High-stakes audit
    "BiasAuditor",
    "ExplainableReasoning", 
    "ProceduralAlignment",
    "VerifiableTraining",
    
    # Advanced metrics
    "compute_ece",
    "compute_mce",
    "compute_brier_score",
    "compute_abstention_metrics",
    "compute_risk_aware_metrics",
    "MetricsAggregator",
    
    # Conformal prediction
    "ConformalPredictor", 
    "RiskControlledPredictor",
    
    # Utilities
    "ConfigManager",
    "ModelLoader",
    "PromptFormatter",
    "MetricsTracker",
    "ErrorHandler",
    "load_config",
    "format_prompt",
    "parse_model_output"
]

def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies."""
    if name in ["load_labels", "build_examples", "create_abstention_examples", 
                "create_balanced_dataset", "validate_output_format"]:
        from . import dataset
        return getattr(dataset, name)
    elif name == "LoRATrainer":
        from . import sft_lora
        return getattr(sft_lora, name)
    elif name == "LLMVoterInference":
        from . import infer
        return getattr(infer, name)
    elif name == "ModelEvaluator":
        from . import evaluate
        return getattr(evaluate, name)
    elif name in ["MCDropoutWrapper", "compute_uncertainty_aware_loss", "should_abstain"]:
        from . import uncertainty
        return getattr(uncertainty, name)
    elif name in ["FactChecker", "create_factual_test_data"]:
        from . import fact_check
        return getattr(fact_check, name)
    elif name in ["BiasAuditor", "ExplainableReasoning", "ProceduralAlignment", "VerifiableTraining"]:
        from . import high_stakes_audit
        return getattr(high_stakes_audit, name)
    elif name in ["compute_ece", "compute_mce", "compute_brier_score", "compute_abstention_metrics",
                   "compute_risk_aware_metrics", "MetricsAggregator"]:
        from . import metrics
        return getattr(metrics, name)
    elif name in ["ConformalPredictor", "RiskControlledPredictor"]:
        from . import conformal
        return getattr(conformal, name)
    elif name in ["ConfigManager", "ModelLoader", "PromptFormatter", "MetricsTracker", 
                   "ErrorHandler", "load_config", "format_prompt", "parse_model_output"]:
        from . import utils
        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")