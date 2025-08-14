"""
LLM-specific evaluator implementation.

This module provides evaluation capabilities for large language models,
handling text generation, classification, and structured output parsing.
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseEvaluator
from ..metrics import MetricsComputer
from ...core.exceptions import ModelError, InferenceError

logger = logging.getLogger(__name__)


class LLMEvaluator(BaseEvaluator):
    """Evaluator for large language models with LoRA/QLoRA support."""
    
    def __init__(self, 
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 model_path: Optional[str] = None,
                 config_path: str = "configs/llm_lora.yaml"):
        """
        Initialize LLM evaluator.
        
        Args:
            model: Pre-trained model
            tokenizer: Model tokenizer
            model_path: Path to model checkpoint
            config_path: Path to configuration file
        """
        super().__init__(model, tokenizer)
        
        self.model_path = model_path
        self.config_path = Path(config_path)
        
        # Load configuration
        self.load_config()
        
        # Load model if not provided
        if not self.model and self.model_path:
            self.load_model()
        
        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(self.config)
        
        # Generation parameters
        self.generation_config = self.config.get('generation', {})
        self.max_new_tokens = self.generation_config.get('max_new_tokens', 512)
        self.temperature = self.generation_config.get('temperature', 0.7)
        self.top_p = self.generation_config.get('top_p', 0.9)
        self.do_sample = self.generation_config.get('do_sample', False)
        
        # Evaluation parameters
        self.parse_json_output = self.config.get('parse_json_output', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # Label configuration
        self.label_list = self.config.get('labels', [])
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        
        # Cache for results
        self.evaluation_cache = {}
    
    def load_config(self):
        """Load configuration from YAML file."""
        import yaml
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}
    
    def load_model(self):
        """Load model and tokenizer."""
        try:
            if self.model_path:
                # Load from checkpoint
                from peft import PeftModel
                
                base_model_id = self.config.get('model_id', 'ZHIPU-AI/glm-4-9b-chat')
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_id,
                    trust_remote_code=True
                )
                
                logger.info(f"Loaded model from {self.model_path}")
            else:
                raise ModelError("No model path provided")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelError(f"Failed to load model: {e}")
    
    def evaluate_single(self, 
                       text: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single text input.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Evaluation results
        """
        return self.predict_single(text, metadata)
    
    def predict_single(self, 
                      text: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate prediction for single input.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Prediction results with confidence and reasoning
        """
        try:
            # Format input with instruction template
            formatted_input = self._format_input(text, metadata)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse response
            result = self._parse_response(response)
            
            # Add metadata
            result['input_text'] = text[:500]  # Truncate for storage
            result['raw_response'] = response
            result['timestamp'] = datetime.now().isoformat()
            
            if metadata:
                result['metadata'] = metadata
            
            # Compute confidence if not provided
            if 'confidence' not in result:
                result['confidence'] = self._compute_confidence(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'error': str(e),
                'input_text': text[:500],
                'prediction': 'error',
                'confidence': 0.0
            }
    
    def evaluate_dataset(self,
                        dataset: Union[List[Dict], Any],
                        batch_size: int = 1,
                        output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Evaluate entire dataset.
        
        Args:
            dataset: Dataset to evaluate
            batch_size: Batch size for evaluation
            output_path: Optional path to save results
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Convert dataset to list if needed
        if hasattr(dataset, '__iter__'):
            dataset_list = list(dataset)
        else:
            dataset_list = dataset
        
        # Process in batches
        for i in tqdm(range(0, len(dataset_list), batch_size), desc="Evaluating"):
            batch = dataset_list[i:i + batch_size]
            
            for sample in batch:
                # Extract text and metadata
                if isinstance(sample, dict):
                    text = sample.get('text', str(sample))
                    metadata = {k: v for k, v in sample.items() if k != 'text'}
                else:
                    text = str(sample)
                    metadata = {}
                
                # Get prediction
                result = self.predict_single(text, metadata)
                
                # Add ground truth if available
                if 'label' in sample:
                    result['ground_truth'] = sample['label']
                
                results.append(result)
        
        # Save results if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved evaluation results to {output_path}")
        
        return results
    
    def compute_metrics(self,
                       predictions: List[Any] = None,
                       labels: List[Any] = None,
                       detailed_results: List[Dict] = None,
                       **kwargs) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: List of predictions
            labels: List of ground truth labels
            detailed_results: Detailed evaluation results
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of computed metrics
        """
        # Use detailed results if provided
        if detailed_results:
            predictions = [r.get('prediction') for r in detailed_results]
            labels = [r.get('ground_truth') for r in detailed_results]
            confidences = [r.get('confidence', 0.5) for r in detailed_results]
        else:
            confidences = kwargs.get('confidences', [0.5] * len(predictions))
        
        # Filter valid predictions
        valid_indices = [
            i for i, (p, l) in enumerate(zip(predictions, labels))
            if p is not None and l is not None
        ]
        
        if not valid_indices:
            logger.warning("No valid predictions for metric computation")
            return {}
        
        # Extract valid data
        valid_preds = [predictions[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        valid_confs = [confidences[i] for i in valid_indices]
        
        # Convert to indices
        pred_indices = [self._label_to_index(p) for p in valid_preds]
        label_indices = [self._label_to_index(l) for l in valid_labels]
        
        # Compute metrics using metrics computer
        metrics = self.metrics_computer.compute_all_metrics(
            predictions=pred_indices,
            labels=label_indices,
            probabilities=valid_confs
        )
        
        return metrics
    
    def _format_input(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Format input text with instruction template."""
        system_prompt = self.config.get('system_prompt', 
            "You are a helpful assistant for classification tasks.")
        
        instruction = self.config.get('instruction_template',
            "Classify the following text and provide your reasoning.")
        
        # Build formatted input
        formatted = f"{system_prompt}\n\n{instruction}\n\nText: {text}"
        
        if metadata:
            formatted += f"\n\nMetadata: {json.dumps(metadata)}"
        
        formatted += "\n\nResponse:"
        
        return formatted
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response."""
        result = {
            'raw_response': response,
            'prediction': None,
            'confidence': 0.5,
            'reasoning': None
        }
        
        if self.parse_json_output:
            try:
                # Try to parse as JSON
                json_match = response.find('{')
                if json_match >= 0:
                    json_str = response[json_match:]
                    json_end = json_str.rfind('}') + 1
                    json_str = json_str[:json_end]
                    
                    parsed = json.loads(json_str)
                    
                    result['prediction'] = parsed.get('decision', parsed.get('label'))
                    result['confidence'] = float(parsed.get('confidence', 0.5))
                    result['reasoning'] = parsed.get('rationale', parsed.get('reasoning'))
                    result['abstain'] = parsed.get('abstain', False)
                    
            except json.JSONDecodeError:
                # Fallback to text parsing
                result['prediction'] = self._extract_prediction_from_text(response)
        else:
            # Direct text parsing
            result['prediction'] = self._extract_prediction_from_text(response)
        
        return result
    
    def _extract_prediction_from_text(self, text: str) -> Optional[str]:
        """Extract prediction from text response."""
        text_lower = text.lower()
        
        # Check for label mentions
        for label in self.label_list:
            if label.lower() in text_lower:
                return label
        
        # Check for common patterns
        patterns = [
            r"(?:label|class|category):\s*(\w+)",
            r"(?:prediction|decision):\s*(\w+)",
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        return None
    
    def _compute_confidence(self, result: Dict) -> float:
        """Compute confidence score for prediction."""
        if 'confidence' in result:
            return float(result['confidence'])
        
        # Heuristic based on response characteristics
        confidence = 0.5
        
        if result.get('reasoning'):
            confidence += 0.2
        
        if result.get('prediction') in self.label_list:
            confidence += 0.2
        
        if not result.get('abstain'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _label_to_index(self, label: Any) -> int:
        """Convert label to index."""
        if label in self.label_to_id:
            return self.label_to_id[label]
        
        # Try to convert string to existing label
        label_str = str(label).lower()
        for orig_label, idx in self.label_to_id.items():
            if orig_label.lower() == label_str:
                return idx
        
        # Default to first label
        return 0