"""
Instruction formatting transformer for LLM fine-tuning.

This module provides transformation capabilities to convert raw data
into instruction-following format for supervised fine-tuning.
"""

import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from .base import BaseDataTransformer
from ...core.exceptions import DataError

logger = logging.getLogger(__name__)


class InstructionFormatter(BaseDataTransformer):
    """
    Instruction formatting transformer for LLM training.
    
    Converts raw training data into instruction-following format with
    system prompts, user instructions, and expected responses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize instruction formatter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Instruction formatting configuration
        self.format_config = self.config.get('instruction_format', {})
        
        # Template configuration
        self.system_prompt = self.format_config.get('system_prompt', self._get_default_system_prompt())
        self.instruction_template = self.format_config.get('instruction_template', self._get_default_instruction_template())
        self.response_template = self.format_config.get('response_template', self._get_default_response_template())
        
        # Chat format options
        self.use_chat_format = self.format_config.get('use_chat_format', True)
        self.chat_template = self.format_config.get('chat_template', 'standard')
        
        # JSON response options
        self.use_json_response = self.format_config.get('use_json_response', True)
        self.json_schema = self.format_config.get('json_schema', self._get_default_json_schema())
        
        # Abstention handling
        self.include_abstention = self.format_config.get('include_abstention', True)
        self.abstention_probability = self.format_config.get('abstention_probability', 0.1)
        
        # Field mapping
        self.field_mapping = self.format_config.get('field_mapping', {
            'input': 'text',
            'output': 'label',
            'context': 'metadata'
        })
        
        logger.info(f"Initialized InstructionFormatter with chat_format={self.use_chat_format}, json_response={self.use_json_response}")
    
    def transform_single(self, data: Any) -> Any:
        """
        Transform single data sample into instruction format.
        
        Args:
            data: Input data sample
            
        Returns:
            Transformed instruction-formatted sample
        """
        if not isinstance(data, dict):
            raise DataError(f"Input must be dictionary, got {type(data)}")
        
        # Extract fields based on mapping
        input_text = self._extract_field(data, 'input')
        output_label = self._extract_field(data, 'output')
        context = self._extract_field(data, 'context', default={})
        
        if not input_text:
            raise DataError("Missing input text for instruction formatting")
        
        # Build instruction
        instruction = self._build_instruction(input_text, context)
        
        # Build response
        if output_label is not None:
            response = self._build_response(output_label, context)
        else:
            response = None  # For inference-only samples
        
        # Format as chat conversation
        if self.use_chat_format:
            formatted_sample = self._format_as_chat(instruction, response)
        else:
            formatted_sample = self._format_as_completion(instruction, response)
        
        # Add metadata
        formatted_sample.update({
            'original_data': data,
            'formatting_config': {
                'chat_format': self.use_chat_format,
                'json_response': self.use_json_response,
                'template': self.chat_template
            }
        })
        
        return formatted_sample
    
    def _extract_field(self, data: Dict[str, Any], field_key: str, default: Any = None) -> Any:
        """Extract field from data using field mapping."""
        mapped_field = self.field_mapping.get(field_key, field_key)
        
        # Try mapped field first, then original field key
        if mapped_field in data:
            return data[mapped_field]
        elif field_key in data:
            return data[field_key]
        else:
            return default
    
    def _build_instruction(self, input_text: str, context: Dict[str, Any]) -> str:
        """Build instruction text from input and context."""
        # Format instruction using template
        instruction = self.instruction_template.format(
            text=input_text,
            context=json.dumps(context) if context else "",
            **context
        )
        
        return instruction
    
    def _build_response(self, output_label: Any, context: Dict[str, Any]) -> str:
        """Build response text from output label and context."""
        # Determine if this should be an abstention sample
        should_abstain = self._should_abstain()
        
        if self.use_json_response:
            response_dict = {
                "decision": str(output_label) if not should_abstain else None,
                "rationale": self._generate_rationale(output_label, context, should_abstain),
                "abstain": should_abstain,
                "confidence": self._generate_confidence_score(should_abstain)
            }
            
            if context.get('metadata'):
                response_dict['metadata'] = context['metadata']
            
            response = json.dumps(response_dict, indent=2)
        else:
            if should_abstain:
                response = self._generate_abstention_response(context)
            else:
                response = self.response_template.format(
                    label=output_label,
                    rationale=self._generate_rationale(output_label, context, False),
                    **context
                )
        
        return response
    
    def _format_as_chat(self, instruction: str, response: Optional[str]) -> Dict[str, Any]:
        """Format as chat conversation."""
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user", 
                "content": instruction
            }
        ]
        
        if response is not None:
            messages.append({
                "role": "assistant",
                "content": response
            })
        
        formatted_sample = {
            "messages": messages,
            "template": self.chat_template
        }
        
        return formatted_sample
    
    def _format_as_completion(self, instruction: str, response: Optional[str]) -> Dict[str, Any]:
        """Format as completion-style prompt."""
        # Combine system prompt and instruction
        full_prompt = f"{self.system_prompt}\n\n{instruction}\n\nResponse:"
        
        formatted_sample = {
            "prompt": full_prompt
        }
        
        if response is not None:
            formatted_sample["completion"] = response
        
        return formatted_sample
    
    def _should_abstain(self) -> bool:
        """Determine if this sample should demonstrate abstention."""
        import random
        return random.random() < self.abstention_probability
    
    def _generate_rationale(self, output_label: Any, context: Dict[str, Any], should_abstain: bool) -> str:
        """Generate rationale for the decision."""
        if should_abstain:
            return "I cannot make a confident decision based on the provided information. The evidence is insufficient or contradictory."
        
        # Simple rationale generation based on label
        if isinstance(output_label, str):
            return f"Based on the analysis of the text, the classification is {output_label}."
        elif isinstance(output_label, (int, float)):
            return f"The numerical assessment yields a value of {output_label}."
        else:
            return f"The analysis indicates: {str(output_label)}."
    
    def _generate_confidence_score(self, should_abstain: bool) -> float:
        """Generate confidence score for the decision."""
        import random
        
        if should_abstain:
            # Low confidence for abstention samples
            return round(random.uniform(0.1, 0.6), 2)
        else:
            # High confidence for decision samples
            return round(random.uniform(0.7, 0.95), 2)
    
    def _generate_abstention_response(self, context: Dict[str, Any]) -> str:
        """Generate response for abstention samples."""
        abstention_templates = [
            "I cannot provide a confident classification for this text.",
            "The available information is insufficient to make a reliable decision.",
            "This case requires additional context or expert review.",
            "I abstain from making a decision due to uncertainty in the evidence."
        ]
        
        import random
        return random.choice(abstention_templates)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for instruction following."""
        return """You are a helpful AI assistant specialized in text classification. Your task is to analyze the provided text and provide a structured response with your classification decision, rationale, and confidence level. If you are uncertain about the classification, you should abstain from making a decision."""
    
    def _get_default_instruction_template(self) -> str:
        """Get default instruction template."""
        return """Please analyze the following text and provide your classification:

Text: {text}

{context}

Provide your response in JSON format with 'decision', 'rationale', 'abstain', and 'confidence' fields."""
    
    def _get_default_response_template(self) -> str:
        """Get default response template for non-JSON format."""
        return """Classification: {label}
Rationale: {rationale}"""
    
    def _get_default_json_schema(self) -> Dict[str, Any]:
        """Get default JSON schema for responses."""
        return {
            "type": "object",
            "properties": {
                "decision": {
                    "type": ["string", "null"],
                    "description": "The classification decision or null if abstaining"
                },
                "rationale": {
                    "type": "string",
                    "description": "Explanation for the decision or abstention"
                },
                "abstain": {
                    "type": "boolean", 
                    "description": "Whether to abstain from making a decision"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score for the decision"
                }
            },
            "required": ["decision", "rationale", "abstain", "confidence"],
            "additionalProperties": True
        }
    
    def set_system_prompt(self, system_prompt: str):
        """Update system prompt."""
        self.system_prompt = system_prompt
        logger.info("Updated system prompt")
    
    def set_templates(self, 
                     instruction_template: Optional[str] = None,
                     response_template: Optional[str] = None):
        """Update instruction and response templates."""
        if instruction_template:
            self.instruction_template = instruction_template
        if response_template:
            self.response_template = response_template
        logger.info("Updated templates")
    
    def set_field_mapping(self, mapping: Dict[str, str]):
        """Update field mapping for input data."""
        self.field_mapping.update(mapping)
        logger.info(f"Updated field mapping: {mapping}")
    
    def set_abstention_probability(self, probability: float):
        """Update abstention probability for training samples."""
        if not 0 <= probability <= 1:
            raise ValueError("Abstention probability must be between 0 and 1")
        
        self.abstention_probability = probability
        logger.info(f"Updated abstention probability: {probability}")
    
    def generate_few_shot_examples(self, examples: List[Dict[str, Any]], n_examples: int = 3) -> str:
        """
        Generate few-shot examples for in-context learning.
        
        Args:
            examples: List of example data samples
            n_examples: Number of examples to include
            
        Returns:
            Formatted few-shot examples string
        """
        if not examples:
            return ""
        
        # Sample examples if needed
        if len(examples) > n_examples:
            import random
            examples = random.sample(examples, n_examples)
        
        few_shot_text = "Here are some examples:\n\n"
        
        for i, example in enumerate(examples, 1):
            # Transform example to instruction format
            formatted_example = self.transform_single(example)
            
            if self.use_chat_format:
                messages = formatted_example['messages']
                user_message = next(m['content'] for m in messages if m['role'] == 'user')
                assistant_message = next(m['content'] for m in messages if m['role'] == 'assistant')
                
                few_shot_text += f"Example {i}:\n"
                few_shot_text += f"Input: {user_message}\n"
                few_shot_text += f"Output: {assistant_message}\n\n"
            else:
                few_shot_text += f"Example {i}:\n"
                few_shot_text += f"Prompt: {formatted_example['prompt']}\n"
                few_shot_text += f"Completion: {formatted_example['completion']}\n\n"
        
        return few_shot_text
    
    def create_evaluation_prompt(self, input_text: str, context: Dict[str, Any] = None) -> str:
        """
        Create evaluation prompt for inference.
        
        Args:
            input_text: Input text to classify
            context: Additional context information
            
        Returns:
            Formatted prompt for model inference
        """
        context = context or {}
        
        # Build instruction
        instruction = self._build_instruction(input_text, context)
        
        if self.use_chat_format:
            # Return as chat messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction}
            ]
            return {"messages": messages}
        else:
            # Return as completion prompt
            full_prompt = f"{self.system_prompt}\n\n{instruction}\n\nResponse:"
            return {"prompt": full_prompt}