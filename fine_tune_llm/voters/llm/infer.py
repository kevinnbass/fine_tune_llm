"""LLM inference with JSON schema validation and abstention."""

import json
import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# Import advanced metrics and conformal prediction if available
try:
    from .metrics import compute_ece, compute_confidence_metrics, MetricsAggregator
    from .conformal import ConformalPredictor, RiskControlledPredictor
    from .utils import PromptFormatter, ErrorHandler
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMVoterInference:
    """LLM voter with LoRA adapter for inference."""

    def __init__(
        self,
        model_path: str,
        base_model_id: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        abstain_threshold: float = 0.8,
        temperature: float = 1.0,
        max_new_tokens: int = 256,
    ):
        """
        Initialize LLM voter for inference.

        Args:
            model_path: Path to LoRA adapter
            base_model_id: Base model ID (if not in adapter config)
            device: Device to run on
            abstain_threshold: Confidence threshold for abstention
            temperature: Temperature for generation
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = Path(model_path)
        self.device = device
        self.abstain_threshold = abstain_threshold
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Load config
        config_path = self.model_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
                if not base_model_id:
                    base_model_id = self.config.get("model_id")

        self.base_model_id = base_model_id or "Qwen/Qwen2.5-7B"

        # Initialize advanced inference components
        self.conformal_predictor = None
        self.risk_controlled_predictor = None
        self.metrics_tracker = None
        
        if ADVANCED_METRICS_AVAILABLE:
            # Initialize conformal prediction (will be calibrated externally)
            self.conformal_predictor = ConformalPredictor(alpha=0.1)
            self.risk_controlled_predictor = RiskControlledPredictor(alpha=0.1)
            
            # Initialize metrics tracking for inference
            metrics_path = self.model_path.parent / "inference_metrics.json"
            self.metrics_tracker = MetricsAggregator(save_path=metrics_path)
            logger.info("Initialized advanced inference components")

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load base model, LoRA adapter, and tokenizer."""
        logger.info(f"Loading base model: {self.base_model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        # Load LoRA adapter
        logger.info(f"Loading LoRA adapter from {self.model_path}")
        self.model = PeftModel.from_pretrained(self.model, self.model_path)

        # Move to device if needed
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully")

    def format_input(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Format input for the model.

        Args:
            text: Input text to classify
            metadata: Optional metadata

        Returns:
            Formatted input string
        """
        system_prompt = """You are a specialized classifier for bird flu content.
Analyze the text and return a JSON response with your classification.
You MUST return valid JSON with "decision", "rationale", and "abstain" fields.
If you are uncertain or the text is ambiguous, set "abstain": true.

Valid labels: HIGH_RISK, MEDIUM_RISK, LOW_RISK, NO_RISK"""

        input_parts = [f"Text to classify:\n{text}"]

        if metadata:
            meta_str = json.dumps(metadata, indent=2)
            input_parts.append(f"\nMetadata:\n{meta_str}")

        formatted = f"### Instruction:\n{system_prompt}\n\n"
        formatted += f"### Input:\n{'\n'.join(input_parts)}\n\n"
        formatted += "### Output:\n"

        return formatted

    def generate(
        self, input_text: str, return_logprobs: bool = False
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate output from model.

        Args:
            input_text: Formatted input text
            return_logprobs: Whether to return logprobs

        Returns:
            Tuple of (generated_text, logprobs_dict)
        """
        # Tokenize
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=return_logprobs,
            )

        # Decode
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract logprobs if requested
        logprobs = None
        if return_logprobs and outputs.scores:
            logprobs = self._compute_logprobs(outputs.scores, generated_ids)

        return generated_text, logprobs

    def _compute_logprobs(self, scores, generated_ids) -> Dict[str, float]:
        """
        Compute label-conditional logprobs.

        Args:
            scores: Generation scores from model
            generated_ids: Generated token IDs

        Returns:
            Dictionary of label probabilities
        """
        # Simple heuristic: look for label tokens in first few positions
        label_tokens = {
            "HIGH_RISK": self.tokenizer.encode("HIGH_RISK", add_special_tokens=False),
            "MEDIUM_RISK": self.tokenizer.encode("MEDIUM_RISK", add_special_tokens=False),
            "LOW_RISK": self.tokenizer.encode("LOW_RISK", add_special_tokens=False),
            "NO_RISK": self.tokenizer.encode("NO_RISK", add_special_tokens=False),
        }

        probs = {}
        for label, tokens in label_tokens.items():
            # Find if label appears in generated text
            label_prob = 0.0
            for i, score in enumerate(scores[:10]):  # Check first 10 tokens
                if i < len(generated_ids):
                    token_probs = torch.softmax(score, dim=-1)
                    for token_id in tokens:
                        if token_id < token_probs.shape[-1]:
                            label_prob = max(label_prob, token_probs[0, token_id].item())

            probs[label] = label_prob

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def parse_output(self, generated_text: str) -> Tuple[bool, Optional[Dict]]:
        """
        Parse and validate generated JSON.

        Args:
            generated_text: Raw generated text

        Returns:
            Tuple of (is_valid, parsed_dict)
        """
        try:
            # Try to extract JSON from the text
            # Look for JSON-like structure
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                return False, None

            json_str = generated_text[json_start:json_end]
            output = json.loads(json_str)

            # Validate schema
            required_fields = {"decision", "rationale", "abstain"}
            if not all(field in output for field in required_fields):
                return False, None

            # Validate decision
            if not output["abstain"]:
                valid_labels = {"HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"}
                if output["decision"] not in valid_labels:
                    return False, None

            # Validate types
            if not isinstance(output["abstain"], bool):
                return False, None

            if not isinstance(output["rationale"], str):
                return False, None

            return True, output

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse output: {e}")
            return False, None

    def predict(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make prediction on input text.

        Args:
            text: Input text to classify
            metadata: Optional metadata

        Returns:
            Voter output dictionary
        """
        start_time = time.time()

        # Format input
        formatted_input = self.format_input(text, metadata)

        # Generate
        generated_text, logprobs = self.generate(formatted_input, return_logprobs=True)

        # Parse output
        is_valid, parsed_output = self.parse_output(generated_text)

        # Handle invalid output - abstain
        if not is_valid:
            logger.warning("Invalid JSON output, abstaining")
            return {
                "probs": {},
                "abstain": True,
                "reason": "Invalid JSON output",
                "latency_ms": (time.time() - start_time) * 1000,
                "cost_cents": 0.05,  # Approximate
                "model_id": f"llm_lora_{self.base_model_id}",
                "raw_output": generated_text,
            }

        # Check if model abstained
        if parsed_output["abstain"]:
            return {
                "probs": {},
                "abstain": True,
                "reason": parsed_output.get("rationale", "Model abstained"),
                "latency_ms": (time.time() - start_time) * 1000,
                "cost_cents": 0.05,
                "model_id": f"llm_lora_{self.base_model_id}",
                "raw_output": parsed_output,
            }

        # Build probability distribution
        # Use logprobs if available, otherwise use heuristic
        if logprobs:
            probs = logprobs
        else:
            # Simple heuristic based on decision
            decision = parsed_output["decision"]
            probs = {"HIGH_RISK": 0.1, "MEDIUM_RISK": 0.1, "LOW_RISK": 0.1, "NO_RISK": 0.1}
            probs[decision] = 0.7  # High confidence for chosen label

            # Normalize
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}

        # Check abstention threshold
        max_prob = max(probs.values())
        if max_prob < self.abstain_threshold:
            return {
                "probs": {},
                "abstain": True,
                "reason": f"Low confidence: {max_prob:.3f} < {self.abstain_threshold}",
                "latency_ms": (time.time() - start_time) * 1000,
                "cost_cents": 0.05,
                "model_id": f"llm_lora_{self.base_model_id}",
                "raw_output": parsed_output,
            }

        # Enhanced prediction with conformal prediction
        result = {
            "probs": probs,
            "abstain": False,
            "decision": parsed_output["decision"],
            "rationale": parsed_output["rationale"],
            "latency_ms": (time.time() - start_time) * 1000,
            "cost_cents": 0.05,
            "model_id": f"llm_lora_{self.base_model_id}",
            "max_prob": max_prob,
            "raw_output": parsed_output,
        }
        
        # Add conformal prediction if available
        if ADVANCED_METRICS_AVAILABLE and self.conformal_predictor is not None:
            try:
                # Convert probs to numpy array for conformal prediction
                prob_array = np.array([[probs.get(label, 0) for label in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]]])
                
                if self.conformal_predictor.is_calibrated:
                    # Get prediction sets
                    prediction_sets, set_sizes = self.conformal_predictor.predict_sets(prob_array, return_sizes=True)
                    
                    # Convert to interpretable format
                    label_names = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]
                    prediction_set = [label_names[i] for i in range(len(label_names)) if prediction_sets[0, i]]
                    
                    result["conformal_prediction_set"] = prediction_set
                    result["conformal_set_size"] = int(set_sizes[0])
                    result["conformal_coverage_level"] = 1.0 - self.conformal_predictor.alpha
                    
                    # Check if we should abstain based on set size
                    conformal_abstain = set_sizes[0] > 2  # Abstain if uncertain between >2 classes
                    result["conformal_abstain"] = bool(conformal_abstain)
                
                # Risk-controlled prediction if available
                if self.risk_controlled_predictor is not None and hasattr(self.risk_controlled_predictor, 'risk_threshold'):
                    risk_result = self.risk_controlled_predictor.predict_with_risk_control(prob_array)
                    result["risk_score"] = float(risk_result["risk_scores"][0])
                    result["risk_abstain"] = bool(risk_result["abstentions"][0])
                    result["risk_threshold"] = self.risk_controlled_predictor.risk_threshold
                
                # Track inference metrics
                if self.metrics_tracker is not None:
                    inference_metrics = {
                        "confidence": max_prob,
                        "latency_ms": result["latency_ms"],
                        "abstained": result["abstain"]
                    }
                    
                    if "conformal_set_size" in result:
                        inference_metrics["conformal_set_size"] = result["conformal_set_size"]
                    if "risk_score" in result:
                        inference_metrics["risk_score"] = result["risk_score"]
                    
                    self.metrics_tracker.add_metrics(inference_metrics)
                    
            except Exception as e:
                logger.warning(f"Advanced inference features failed: {e}")
        
        return result
    
    def calibrate_conformal_predictor(self, validation_texts: List[str], validation_labels: List[str], 
                                      validation_metadata: Optional[List[Dict]] = None):
        """
        Calibrate conformal predictor using validation data.
        
        Args:
            validation_texts: List of validation texts
            validation_labels: List of true labels
            validation_metadata: Optional metadata for each text
        """
        if not ADVANCED_METRICS_AVAILABLE or self.conformal_predictor is None:
            logger.warning("Advanced metrics not available for conformal calibration")
            return
        
        logger.info(f"Calibrating conformal predictor with {len(validation_texts)} samples...")
        
        # Get predictions for validation set
        predictions = []
        probabilities = []
        
        for i, text in enumerate(validation_texts):
            metadata = validation_metadata[i] if validation_metadata else None
            pred_result = self.predict(text, metadata)
            
            if not pred_result["abstain"]:
                predictions.append(pred_result["decision"])
                probs = pred_result["probs"]
                prob_array = [probs.get(label, 0) for label in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]]
                probabilities.append(prob_array)
        
        if len(probabilities) > 0:
            probabilities = np.array(probabilities)
            
            # Convert string labels to indices
            label_to_idx = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2, "NO_RISK": 3}
            label_indices = np.array([label_to_idx.get(label, 3) for label in validation_labels[:len(probabilities)]])
            
            # Calibrate conformal predictor
            self.conformal_predictor.calibrate(probabilities, label_indices)
            
            # Calibrate risk-controlled predictor
            if self.risk_controlled_predictor is not None:
                self.risk_controlled_predictor.calibrate(probabilities, label_indices)
            
            logger.info("Conformal predictor calibration completed")
        else:
            logger.warning("No valid predictions for calibration")
    
    def get_inference_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of inference metrics."""
        if self.metrics_tracker is not None:
            return self.metrics_tracker.get_summary()
        return {}

    def batch_predict(self, texts: list, metadata_list: Optional[list] = None) -> list:
        """
        Make predictions on batch of texts.

        Args:
            texts: List of input texts
            metadata_list: Optional list of metadata dicts

        Returns:
            List of voter outputs
        """
        results = []

        if metadata_list is None:
            metadata_list = [None] * len(texts)

        for text, metadata in zip(texts, metadata_list):
            result = self.predict(text, metadata)
            results.append(result)

        return results


def calibrate_temperature(
    model_path: str,
    calibration_data: list,
    temp_range: Tuple[float, float] = (0.1, 2.0),
    n_steps: int = 20,
) -> float:
    """
    Find optimal temperature for calibration.

    Args:
        model_path: Path to model
        calibration_data: List of (text, label) tuples
        temp_range: Range of temperatures to try
        n_steps: Number of steps in range

    Returns:
        Optimal temperature
    """
    from ..classical.calibrate import compute_ece

    temperatures = np.linspace(temp_range[0], temp_range[1], n_steps)
    best_temp = 1.0
    best_ece = float("inf")

    for temp in temperatures:
        voter = LLMVoterInference(model_path, temperature=temp)

        y_true = []
        y_prob = []

        for text, true_label in calibration_data:
            result = voter.predict(text)

            if not result["abstain"]:
                y_true.append(true_label)
                # Get probability for true label
                prob = result["probs"].get(true_label, 0.0)
                y_prob.append(prob)

        if len(y_true) > 0:
            ece = compute_ece(np.array(y_true), np.array(y_prob))

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

            logger.info(f"Temperature {temp:.2f}: ECE = {ece:.4f}")

    logger.info(f"Best temperature: {best_temp:.2f} with ECE = {best_ece:.4f}")
    return best_temp
