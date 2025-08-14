"""
Base inference engine interface.

This module provides the abstract base class for all inference engines
with common inference functionality and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ...core.interfaces import BaseComponent


class BaseInferenceEngine(BaseComponent, ABC):
    """Abstract base class for inference engines."""
    
    def __init__(self, 
                 model: Optional[PreTrainedModel] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base inference engine.
        
        Args:
            model: Pre-trained model
            tokenizer: Model tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model:
            self.model = self.model.to(self.device)
        
        # Generation parameters
        self.generation_config = self.config.get('generation', {})
        self.max_new_tokens = self.generation_config.get('max_new_tokens', 512)
        self.temperature = self.generation_config.get('temperature', 0.7)
        self.top_p = self.generation_config.get('top_p', 0.9)
        self.top_k = self.generation_config.get('top_k', 50)
        self.do_sample = self.generation_config.get('do_sample', True)
        self.num_beams = self.generation_config.get('num_beams', 1)
        
        # Inference settings
        self.batch_size = self.config.get('batch_size', 1)
        self.max_length = self.config.get('max_length', 2048)
        
        # Performance optimization
        self.use_cache = self.config.get('use_cache', True)
        self.pad_token_id = None
        self.eos_token_id = None
        
        if self.tokenizer:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        self.generation_config = self.config.get('generation', {})
    
    def cleanup(self) -> None:
        """Clean up inference resources."""
        if self.model:
            del self.model
            torch.cuda.empty_cache()
    
    @property
    def name(self) -> str:
        """Component name."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    @abstractmethod
    def predict(self, 
                inputs: Union[str, List[str], Dict[str, Any]],
                **kwargs) -> Union[str, List[str], Dict[str, Any]]:
        """
        Generate predictions for inputs.
        
        Args:
            inputs: Input text, list of texts, or structured input
            **kwargs: Additional generation parameters
            
        Returns:
            Generated predictions
        """
        pass
    
    @abstractmethod
    def predict_batch(self, 
                     inputs: List[Union[str, Dict[str, Any]]],
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Generate predictions for a batch of inputs.
        
        Args:
            inputs: List of input texts or structured inputs
            **kwargs: Additional generation parameters
            
        Returns:
            List of prediction results
        """
        pass
    
    def preprocess_input(self, 
                        text: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Preprocess input text for model inference.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Tokenized inputs ready for model
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not available for preprocessing")
        
        # Format input if needed
        formatted_text = self.format_input(text, metadata)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def postprocess_output(self, 
                          outputs: torch.Tensor,
                          inputs: Dict[str, torch.Tensor]) -> str:
        """
        Postprocess model outputs to text.
        
        Args:
            outputs: Model output tokens
            inputs: Original input tokens
            
        Returns:
            Generated text
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not available for postprocessing")
        
        # Extract only new tokens (skip input tokens)
        if outputs.shape[1] > inputs['input_ids'].shape[1]:
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        else:
            new_tokens = outputs[0]
        
        # Decode
        generated_text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def format_input(self, 
                    text: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Format input text with any necessary prompts or templates.
        
        Args:
            text: Raw input text
            metadata: Optional metadata
            
        Returns:
            Formatted input text
        """
        # Default implementation - can be overridden by subclasses
        return text
    
    def generate_with_config(self, 
                           inputs: Dict[str, torch.Tensor],
                           **generation_kwargs) -> torch.Tensor:
        """
        Generate text with specified configuration.
        
        Args:
            inputs: Tokenized inputs
            **generation_kwargs: Override generation parameters
            
        Returns:
            Generated token sequences
        """
        if not self.model:
            raise ValueError("Model not available for generation")
        
        # Merge generation config with overrides
        gen_config = {
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'do_sample': self.do_sample,
            'num_beams': self.num_beams,
            'use_cache': self.use_cache,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
            **generation_kwargs
        }
        
        # Remove None values
        gen_config = {k: v for k, v in gen_config.items() if v is not None}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
        
        return outputs
    
    def compute_perplexity(self, 
                          text: str,
                          stride: int = 512) -> float:
        """
        Compute perplexity of text under the model.
        
        Args:
            text: Input text
            stride: Sliding window stride
            
        Returns:
            Perplexity score
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer required for perplexity computation")
        
        # Tokenize
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        seq_len = input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        
        # Compute negative log-likelihoods with sliding window
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            input_slice = input_ids[:, begin_loc:end_loc]
            target_ids = input_slice.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = self.model(input_slice, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
            
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            
            if end_loc == seq_len:
                break
        
        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.item()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.model:
            return {}
        
        info = {
            'model_type': type(self.model).__name__,
            'device': str(self.device),
            'generation_config': self.generation_config,
            'max_length': self.max_length,
            'batch_size': self.batch_size
        }
        
        # Add model parameters info
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0.0
            })
        except Exception:
            pass
        
        # Add memory info if on GPU
        if torch.cuda.is_available():
            info.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3     # GB
            })
        
        return info
    
    def warm_up(self, num_warmup_steps: int = 3):
        """
        Warm up the model with dummy inputs.
        
        Args:
            num_warmup_steps: Number of warmup inference steps
        """
        if not self.model or not self.tokenizer:
            return
        
        dummy_text = "This is a warmup input for the model."
        
        for _ in range(num_warmup_steps):
            try:
                _ = self.predict(dummy_text)
            except Exception:
                pass  # Ignore warmup errors
    
    def set_generation_config(self, **kwargs):
        """Update generation configuration."""
        self.generation_config.update(kwargs)
        
        # Update individual parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def reset_generation_config(self):
        """Reset generation configuration to defaults."""
        self.generation_config = self.config.get('generation', {})
        
        # Reset individual parameters
        self.max_new_tokens = self.generation_config.get('max_new_tokens', 512)
        self.temperature = self.generation_config.get('temperature', 0.7)
        self.top_p = self.generation_config.get('top_p', 0.9)
        self.top_k = self.generation_config.get('top_k', 50)
        self.do_sample = self.generation_config.get('do_sample', True)
        self.num_beams = self.generation_config.get('num_beams', 1)