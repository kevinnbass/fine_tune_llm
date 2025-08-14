"""
Shared utilities for LLM fine-tuning modules.
Consolidates common functionality to avoid duplication.
"""

import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration management."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file with caching."""
        config_path = Path(config_path)
        
        if self._config is None or self._config.get('_source') != str(config_path):
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
                self._config['_source'] = str(config_path)
        
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with dot notation support."""
        if self._config is None:
            return default
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def validate_config(self, required_keys: List[str]) -> bool:
        """Validate that required configuration keys exist."""
        if self._config is None:
            return False
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Missing required config key: {key}")
                return False
        
        return True


class ModelLoader:
    """Unified model loading utility."""
    
    @staticmethod
    def load_base_model(
        model_id: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True
    ) -> AutoModelForCausalLM:
        """Load base model with consistent settings."""
        logger.info(f"Loading base model: {model_id}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code
        )
        
        return model
    
    @staticmethod
    def load_tokenizer(
        tokenizer_id: str,
        trust_remote_code: bool = True,
        padding_side: str = "left"
    ) -> AutoTokenizer:
        """Load tokenizer with consistent settings."""
        logger.info(f"Loading tokenizer: {tokenizer_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=trust_remote_code,
            padding_side=padding_side
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    @staticmethod
    def load_lora_model(
        base_model: AutoModelForCausalLM,
        lora_path: Union[str, Path]
    ) -> PeftModel:
        """Load LoRA adapter onto base model."""
        logger.info(f"Loading LoRA adapter from: {lora_path}")
        
        model = PeftModel.from_pretrained(
            base_model,
            str(lora_path),
            is_trainable=False
        )
        
        return model
    
    @staticmethod
    def merge_and_save(
        model: PeftModel,
        output_path: Union[str, Path],
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None
    ):
        """Merge LoRA weights and save model."""
        logger.info(f"Merging LoRA weights and saving to: {output_path}")
        
        # Merge LoRA weights
        merged_model = model.merge_and_unload()
        
        # Save locally
        merged_model.save_pretrained(str(output_path))
        
        # Push to hub if requested
        if push_to_hub and hub_model_id:
            logger.info(f"Pushing to hub: {hub_model_id}")
            merged_model.push_to_hub(hub_model_id)
        
        return merged_model


class PromptFormatter:
    """Unified prompt formatting for consistency across modules."""
    
    @staticmethod
    def format_instruction(
        text: str,
        system_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        add_generation_prompt: bool = True
    ) -> str:
        """Format instruction prompt consistently."""
        prompt_parts = []
        
        # Add system prompt
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        
        # Add main text
        prompt_parts.append(f"Text: {text}")
        
        # Add metadata if provided
        if metadata:
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            prompt_parts.append(f"Metadata: {metadata_str}")
        
        # Add generation prompt
        if add_generation_prompt:
            prompt_parts.append("Response:")
        
        return "\n\n".join(prompt_parts)
    
    @staticmethod
    def parse_json_response(
        response: str,
        expected_fields: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse JSON response with error handling."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        try:
            parsed = json.loads(response.strip())
            
            # Validate expected fields if provided
            if expected_fields:
                for field in expected_fields:
                    if field not in parsed:
                        logger.warning(f"Missing expected field: {field}")
                        parsed[field] = None
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            
            # Try to extract partial JSON
            try:
                # Find JSON-like content
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx+1]
                    return json.loads(json_str)
            except:
                pass
            
            return None


class MetricsTracker:
    """Unified metrics tracking across modules."""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Add to history with timestamp
        import time
        self.history.append({
            'timestamp': time.time(),
            **kwargs
        })
    
    def get_average(self, metric: str, last_n: int = None) -> float:
        """Get average of a metric."""
        if metric not in self.metrics:
            return 0.0
        
        values = self.metrics[metric]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values) if values else 0.0
    
    def get_latest(self, metric: str) -> Any:
        """Get latest value of a metric."""
        if metric not in self.metrics or not self.metrics[metric]:
            return None
        return self.metrics[metric][-1]
    
    def save_to_file(self, output_path: Union[str, Path]):
        """Save metrics to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'history': self.history
            }, f, indent=2)
    
    def load_from_file(self, input_path: Union[str, Path]):
        """Load metrics from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
            self.metrics = data.get('metrics', {})
            self.history = data.get('history', [])


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    @staticmethod
    def safe_execute(func, *args, default=None, **kwargs):
        """Execute function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return default
    
    @staticmethod
    def retry_with_backoff(
        func,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        *args,
        **kwargs
    ):
        """Retry function with exponential backoff."""
        import time
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        return None


# Export commonly used functions
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to load config."""
    return ConfigManager().load_config(config_path)


def format_prompt(text: str, config: Dict[str, Any]) -> str:
    """Convenience function to format prompt from config."""
    system_prompt = config.get('instruction_format', {}).get('system_prompt', '')
    return PromptFormatter.format_instruction(text, system_prompt)


def parse_model_output(response: str) -> Optional[Dict[str, Any]]:
    """Convenience function to parse model output."""
    return PromptFormatter.parse_json_response(
        response,
        expected_fields=['decision', 'rationale', 'confidence', 'abstain']
    )