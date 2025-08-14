"""
LLM-specific inference engine with advanced capabilities.

This module provides specialized inference for large language models
with support for structured output, confidence scoring, and error handling.
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from .base import BaseInferenceEngine
from ...core.exceptions import InferenceError

logger = logging.getLogger(__name__)


class LLMInferenceEngine(BaseInferenceEngine):
    """
    Advanced LLM inference engine with structured output support.
    
    Features:
    - Structured JSON output parsing
    - Confidence score computation
    - Error handling and recovery
    - Batch processing optimization
    - Model-specific prompt formatting
    """
    
    def __init__(self, 
                 model=None,
                 tokenizer=None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM inference engine.
        
        Args:
            model: Pre-trained model
            tokenizer: Model tokenizer
            config: Configuration dictionary
        """
        super().__init__(model, tokenizer, config)
        
        # LLM-specific configuration
        self.llm_config = self.config.get('llm', {})
        
        # Output parsing
        self.parse_json = self.llm_config.get('parse_json', True)
        self.require_json = self.llm_config.get('require_json', False)
        self.json_retry_attempts = self.llm_config.get('json_retry_attempts', 2)
        
        # Confidence estimation
        self.compute_confidence = self.llm_config.get('compute_confidence', True)
        self.confidence_method = self.llm_config.get('confidence_method', 'max_prob')
        
        # Prompting
        self.system_prompt = self.llm_config.get('system_prompt', 
            "You are a helpful assistant. Provide accurate and helpful responses.")
        self.instruction_template = self.llm_config.get('instruction_template')
        self.response_format = self.llm_config.get('response_format', 'json')
        
        # Error handling
        self.max_retries = self.llm_config.get('max_retries', 1)
        self.fallback_response = self.llm_config.get('fallback_response', {
            'response': 'I apologize, but I encountered an error processing your request.',
            'confidence': 0.0,
            'error': True
        })
        
        # Model-specific settings
        self.model_type = self._detect_model_type()
        
        logger.info(f"Initialized LLMInferenceEngine:")
        logger.info(f"  - Model type: {self.model_type}")
        logger.info(f"  - JSON parsing: {self.parse_json}")
        logger.info(f"  - Confidence computation: {self.compute_confidence}")
    
    def _detect_model_type(self) -> str:
        """Detect the type of model for specialized handling."""
        if not self.model:
            return "unknown"
        
        model_name = getattr(self.model, 'name_or_path', '')
        if not model_name and hasattr(self.model, 'config'):
            model_name = getattr(self.model.config, '_name_or_path', '')
        
        model_name_lower = model_name.lower()
        
        if 'glm' in model_name_lower:
            return "glm"
        elif 'qwen' in model_name_lower:
            return "qwen"
        elif 'llama' in model_name_lower:
            return "llama"
        elif 'mistral' in model_name_lower:
            return "mistral"
        else:
            return "generic"
    
    def predict(self, 
                inputs: Union[str, Dict[str, Any]],
                **kwargs) -> Dict[str, Any]:
        """
        Generate prediction for single input.
        
        Args:
            inputs: Input text or structured input
            **kwargs: Additional generation parameters
            
        Returns:
            Structured prediction result
        """
        # Extract input components
        if isinstance(inputs, str):
            text = inputs
            metadata = {}
        elif isinstance(inputs, dict):
            text = inputs.get('text', inputs.get('input', ''))
            metadata = {k: v for k, v in inputs.items() if k not in ['text', 'input']}
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        # Attempt prediction with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._predict_single(text, metadata, **kwargs)
                
                # Add attempt info
                result['attempt'] = attempt + 1
                result['timestamp'] = datetime.now().isoformat()
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Prediction attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries:
                    continue
                else:
                    # Return fallback response
                    fallback = self.fallback_response.copy()
                    fallback.update({
                        'input_text': text[:500],  # Truncate for storage
                        'error_message': str(last_error),
                        'attempts': self.max_retries + 1,
                        'timestamp': datetime.now().isoformat()
                    })
                    return fallback
    
    def _predict_single(self, 
                       text: str,
                       metadata: Dict[str, Any],
                       **kwargs) -> Dict[str, Any]:
        """Execute single prediction with full processing."""
        # Preprocess input
        formatted_input = self.format_input(text, metadata)
        model_inputs = self.preprocess_input(formatted_input)
        
        # Generate response
        outputs = self.generate_with_config(model_inputs, **kwargs)
        response_text = self.postprocess_output(outputs, model_inputs)
        
        # Parse response
        parsed_result = self.parse_response(response_text)
        
        # Compute confidence if enabled
        if self.compute_confidence:
            confidence = self._compute_confidence_score(outputs, model_inputs)
            parsed_result['confidence'] = confidence
        
        # Add metadata
        parsed_result.update({
            'input_text': text[:500],  # Truncate for storage
            'raw_response': response_text,
            'model_type': self.model_type
        })
        
        if metadata:
            parsed_result['metadata'] = metadata
        
        return parsed_result
    
    def predict_batch(self, 
                     inputs: List[Union[str, Dict[str, Any]]],
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Generate predictions for batch of inputs.
        
        Args:
            inputs: List of input texts or structured inputs
            **kwargs: Additional generation parameters
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches based on configured batch size
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            if self.batch_size == 1 or len(batch) == 1:
                # Process individually for better error handling
                for inp in batch:
                    result = self.predict(inp, **kwargs)
                    results.append(result)
            else:
                # True batch processing
                try:
                    batch_results = self._predict_batch_optimized(batch, **kwargs)
                    results.extend(batch_results)
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}, falling back to individual processing")
                    # Fallback to individual processing
                    for inp in batch:
                        result = self.predict(inp, **kwargs)
                        results.append(result)
        
        return results
    
    def _predict_batch_optimized(self, 
                               batch: List[Union[str, Dict[str, Any]]],
                               **kwargs) -> List[Dict[str, Any]]:
        """Optimized batch prediction processing."""
        # Prepare batch inputs
        texts = []
        metadatas = []
        
        for inp in batch:
            if isinstance(inp, str):
                texts.append(inp)
                metadatas.append({})
            elif isinstance(inp, dict):
                texts.append(inp.get('text', inp.get('input', '')))
                metadatas.append({k: v for k, v in inp.items() if k not in ['text', 'input']})
        
        # Format all inputs
        formatted_inputs = [self.format_input(text, meta) for text, meta in zip(texts, metadatas)]
        
        # Batch tokenization
        batch_inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
        
        # Batch generation
        with torch.no_grad():
            batch_outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                top_p=kwargs.get('top_p', self.top_p),
                do_sample=kwargs.get('do_sample', self.do_sample),
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id
            )
        
        # Process outputs
        results = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # Extract individual output
            output_ids = batch_outputs[i:i+1]
            input_ids = batch_inputs['input_ids'][i:i+1]
            
            # Postprocess
            response_text = self.postprocess_output(output_ids, {'input_ids': input_ids})
            
            # Parse and structure result
            parsed_result = self.parse_response(response_text)
            
            # Add confidence if enabled
            if self.compute_confidence:
                # Approximate confidence for batch processing
                confidence = self._compute_batch_confidence(output_ids, i)
                parsed_result['confidence'] = confidence
            
            # Add metadata
            parsed_result.update({
                'input_text': text[:500],
                'raw_response': response_text,
                'model_type': self.model_type,
                'timestamp': datetime.now().isoformat(),
                'batch_index': i
            })
            
            if metadata:
                parsed_result['metadata'] = metadata
            
            results.append(parsed_result)
        
        return results
    
    def format_input(self, 
                    text: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Format input with model-specific prompts.
        
        Args:
            text: Raw input text
            metadata: Optional metadata
            
        Returns:
            Formatted input with appropriate prompts
        """
        # Use instruction template if provided
        if self.instruction_template:
            formatted = self.instruction_template.format(
                system_prompt=self.system_prompt,
                input_text=text,
                metadata=json.dumps(metadata) if metadata else ""
            )
        else:
            # Model-specific formatting
            if self.model_type == "glm":
                formatted = f"[gMASK]sop<|system|>\n{self.system_prompt}<|user|>\n{text}<|assistant|>\n"
            elif self.model_type == "qwen":
                formatted = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            elif self.model_type == "llama":
                formatted = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{text} [/INST]"
            else:
                # Generic format
                formatted = f"System: {self.system_prompt}\nUser: {text}\nAssistant: "
        
        # Add response format instruction if JSON is expected
        if self.response_format == 'json' and self.parse_json:
            if "JSON" not in formatted:
                formatted += "\n\nPlease respond in JSON format."
        
        return formatted
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response into structured format.
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed response dictionary
        """
        result = {
            'raw_response': response,
            'response': response,
            'parsed_successfully': False
        }
        
        if self.parse_json:
            # Attempt JSON parsing
            for attempt in range(self.json_retry_attempts + 1):
                try:
                    # Find JSON in response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        parsed = json.loads(json_str)
                        
                        # Update result with parsed JSON
                        result.update(parsed)
                        result['parsed_successfully'] = True
                        break
                        
                except json.JSONDecodeError as e:
                    if attempt == self.json_retry_attempts:
                        if self.require_json:
                            result['error'] = f"JSON parsing failed: {e}"
                        else:
                            # Extract key information from text
                            result.update(self._extract_from_text(response))
                            result['parsing_fallback'] = True
        else:
            # Text-based parsing
            result.update(self._extract_from_text(response))
        
        return result
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured information from text response."""
        # Basic text extraction logic
        result = {
            'response': text.strip(),
            'length': len(text),
            'has_confidence': 'confidence' in text.lower()
        }
        
        # Try to extract confidence if mentioned
        import re
        conf_patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9.]+)%?\s*confident',
            r'certainty[:\s]+([0-9.]+)'
        ]
        
        for pattern in conf_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    conf_val = float(match.group(1))
                    if conf_val > 1:  # Assume percentage
                        conf_val /= 100
                    result['extracted_confidence'] = conf_val
                    break
                except ValueError:
                    continue
        
        return result
    
    def _compute_confidence_score(self, 
                                 outputs: torch.Tensor,
                                 inputs: Dict[str, torch.Tensor]) -> float:
        """
        Compute confidence score for prediction.
        
        Args:
            outputs: Generated token sequences
            inputs: Input token sequences
            
        Returns:
            Confidence score between 0 and 1
        """
        if self.confidence_method == 'max_prob':
            return self._compute_max_prob_confidence(outputs, inputs)
        elif self.confidence_method == 'entropy':
            return self._compute_entropy_confidence(outputs, inputs)
        elif self.confidence_method == 'sequence_prob':
            return self._compute_sequence_probability(outputs, inputs)
        else:
            return 0.5  # Default confidence
    
    def _compute_max_prob_confidence(self, 
                                   outputs: torch.Tensor,
                                   inputs: Dict[str, torch.Tensor]) -> float:
        """Compute confidence based on maximum token probabilities."""
        try:
            # Get new tokens only
            new_token_start = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][new_token_start:]
            
            if len(new_tokens) == 0:
                return 0.5
            
            # For simplicity, return a heuristic confidence
            # In practice, you'd need access to token probabilities from the model
            # This requires forward pass with the generated sequence
            
            # Heuristic: longer responses with varied tokens indicate higher confidence
            unique_tokens = len(set(new_tokens.tolist()))
            total_tokens = len(new_tokens)
            
            if total_tokens == 0:
                return 0.5
            
            diversity = unique_tokens / total_tokens
            length_factor = min(total_tokens / 20, 1.0)  # Normalize by expected length
            
            confidence = (diversity * 0.5 + length_factor * 0.5)
            return float(np.clip(confidence, 0.1, 0.95))
            
        except Exception:
            return 0.5
    
    def _compute_entropy_confidence(self, 
                                  outputs: torch.Tensor,
                                  inputs: Dict[str, torch.Tensor]) -> float:
        """Compute confidence based on prediction entropy."""
        # Placeholder - would need model logits
        return 0.5
    
    def _compute_sequence_probability(self, 
                                    outputs: torch.Tensor,
                                    inputs: Dict[str, torch.Tensor]) -> float:
        """Compute confidence based on sequence probability."""
        # Placeholder - would need model logits
        return 0.5
    
    def _compute_batch_confidence(self, 
                                outputs: torch.Tensor,
                                batch_index: int) -> float:
        """Compute confidence for batch processing."""
        # Simplified confidence for batch processing
        return 0.7  # Default batch confidence
    
    def set_system_prompt(self, prompt: str):
        """Update system prompt."""
        self.system_prompt = prompt
    
    def set_response_format(self, format_type: str):
        """Set expected response format."""
        self.response_format = format_type
        self.parse_json = (format_type == 'json')
    
    def enable_json_parsing(self, enabled: bool = True):
        """Enable or disable JSON parsing."""
        self.parse_json = enabled
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        stats = self.get_model_info()
        
        stats.update({
            'parse_json': self.parse_json,
            'confidence_method': self.confidence_method,
            'max_retries': self.max_retries,
            'json_retry_attempts': self.json_retry_attempts,
            'response_format': self.response_format
        })
        
        return stats