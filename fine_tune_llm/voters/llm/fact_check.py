"""RELIANCE framework for factual accuracy enhancement in high-stakes scenarios."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class RELIANCEFactChecker:
    """
    RELIANCE framework for detecting and enhancing factual accuracy.
    Based on intermediate reasoning step verification.
    """
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.fact_config = config.get('high_stakes', {}).get('factual', {})
        self.reliance_steps = self.fact_config.get('reliance_steps', 3)
        self.consistency_threshold = self.fact_config.get('self_consistency_threshold', 0.8)
        
    def split_into_reasoning_steps(self, text: str, num_steps: int) -> List[str]:
        """
        Split text into reasoning steps for fact-checking.
        
        Args:
            text: Input text
            num_steps: Number of steps to extract
            
        Returns:
            List of reasoning steps
        """
        # Split by common reasoning markers
        markers = [
            r'\d+\.',  # Numbered lists
            r'First,?', r'Second,?', r'Third,?',  # Ordinal markers
            r'Therefore,?', r'Thus,?', r'Hence,?',  # Conclusion markers
            r'Because', r'Since', r'As',  # Causal markers
            r'However,?', r'But,?', r'Although,?',  # Contrast markers
        ]
        
        # Combine markers into regex
        pattern = '|'.join(markers)
        sentences = re.split(pattern, text, flags=re.IGNORECASE)
        
        # Clean and filter sentences
        steps = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Minimum length for meaningful step
                steps.append(sent)
        
        # If not enough steps, split by sentences
        if len(steps) < num_steps:
            sentences = text.split('.')
            steps = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Return requested number of steps
        return steps[:num_steps] if len(steps) >= num_steps else steps
    
    def check_self_consistency(self, claim: str, num_samples: int = 3) -> Tuple[bool, float]:
        """
        Check self-consistency of a factual claim by generating multiple explanations.
        
        Args:
            claim: The claim to verify
            num_samples: Number of consistency samples
            
        Returns:
            Tuple of (is_consistent, consistency_score)
        """
        try:
            prompt = f"""Explain whether the following claim is factually accurate:
            Claim: {claim}
            
            Provide a brief factual assessment:"""
            
            explanations = []
            
            for _ in range(num_samples):
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,  # Some variation for consistency check
                        do_sample=True,
                    )
                
                explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                explanations.append(explanation.lower())
            
            # Check consistency by looking for agreement
            # Simple approach: check if majority agree on true/false/accurate/inaccurate
            positive_markers = ['true', 'accurate', 'correct', 'valid', 'factual']
            negative_markers = ['false', 'inaccurate', 'incorrect', 'invalid', 'wrong']
            
            positive_count = 0
            negative_count = 0
            
            for exp in explanations:
                if any(marker in exp for marker in positive_markers):
                    positive_count += 1
                if any(marker in exp for marker in negative_markers):
                    negative_count += 1
            
            # Calculate consistency score
            if positive_count > 0 or negative_count > 0:
                total_aligned = max(positive_count, negative_count)
                consistency_score = total_aligned / num_samples
                is_consistent = consistency_score >= self.consistency_threshold
            else:
                # No clear alignment
                consistency_score = 0.0
                is_consistent = False
            
            return is_consistent, consistency_score
            
        except Exception as e:
            logger.error(f"Error in self-consistency check: {e}")
            return False, 0.0
    
    def reliance_fact_check(self, text: str) -> Tuple[bool, float, List[Dict]]:
        """
        Perform RELIANCE fact-checking on text.
        
        Args:
            text: Text to fact-check
            
        Returns:
            Tuple of (is_factual, overall_score, step_results)
        """
        try:
            if not self.fact_config.get('enabled', False):
                return True, 1.0, []
            
            # Split into reasoning steps
            steps = self.split_into_reasoning_steps(text, self.reliance_steps)
            
            if not steps:
                logger.warning("No reasoning steps found in text")
                return True, 1.0, []
            
            step_results = []
            total_score = 0.0
            
            for i, step in enumerate(steps):
                # Check factuality of each step
                is_consistent, score = self.check_self_consistency(step)
                
                step_results.append({
                    'step': i + 1,
                    'text': step[:100] + '...' if len(step) > 100 else step,
                    'is_consistent': is_consistent,
                    'score': score
                })
                
                total_score += score
            
            # Overall factuality
            overall_score = total_score / len(steps) if steps else 0.0
            is_factual = overall_score >= self.consistency_threshold
            
            # Log results
            if not is_factual:
                logger.warning(f"Low factual score: {overall_score:.2f}")
                for result in step_results:
                    if not result['is_consistent']:
                        logger.debug(f"Step {result['step']} failed consistency: {result['text']}")
            
            return is_factual, overall_score, step_results
            
        except Exception as e:
            logger.error(f"Error in RELIANCE fact check: {e}")
            return False, 0.0, []
    
    def compute_factuality_loss(self, generated_text: str, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute factuality-based loss for training.
        
        Args:
            generated_text: Generated text to check
            labels: Optional ground truth labels
            
        Returns:
            Factuality penalty loss
        """
        try:
            is_factual, fact_score, _ = self.reliance_fact_check(generated_text)
            
            # Penalty for non-factual content
            fact_penalty = (1 - fact_score) * self.fact_config.get('fact_penalty_weight', 2.0)
            
            return torch.tensor(fact_penalty, requires_grad=True)
            
        except Exception as e:
            logger.error(f"Error computing factuality loss: {e}")
            return torch.tensor(0.0)


class FactualDataFilter:
    """Filter training data based on factual accuracy."""
    
    def __init__(self, fact_checker: RELIANCEFactChecker):
        self.fact_checker = fact_checker
        self.filtered_count = 0
        self.total_count = 0
        
    def filter_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """
        Filter dataset to remove non-factual samples.
        
        Args:
            dataset: List of data samples
            
        Returns:
            Filtered dataset with only factual samples
        """
        filtered_data = []
        
        for sample in dataset:
            self.total_count += 1
            
            text = sample.get('text', '')
            is_factual, score, _ = self.fact_checker.reliance_fact_check(text)
            
            if is_factual:
                sample['factual_score'] = score
                filtered_data.append(sample)
            else:
                self.filtered_count += 1
                logger.debug(f"Filtered non-factual sample with score {score:.2f}")
        
        logger.info(f"Filtered {self.filtered_count}/{self.total_count} non-factual samples")
        
        return filtered_data
    
    def augment_with_factual_labels(self, dataset: List[Dict]) -> List[Dict]:
        """
        Augment dataset with factual scores instead of filtering.
        
        Args:
            dataset: List of data samples
            
        Returns:
            Dataset with factual scores added
        """
        augmented_data = []
        
        for sample in dataset:
            text = sample.get('text', '')
            is_factual, score, step_results = self.fact_checker.reliance_fact_check(text)
            
            sample['factual_score'] = score
            sample['is_factual'] = is_factual
            sample['factual_steps'] = step_results
            
            # Optionally adjust label for training
            if not is_factual and score < 0.5:
                sample['original_label'] = sample.get('label')
                sample['label'] = 'low_factuality'
            
            augmented_data.append(sample)
        
        return augmented_data


class FactualPromptEnhancer:
    """Enhance prompts to encourage factual responses."""
    
    @staticmethod
    def enhance_prompt(original_prompt: str, domain: str = "general") -> str:
        """
        Enhance prompt with factuality instructions.
        
        Args:
            original_prompt: Original prompt text
            domain: Domain for specialized fact-checking
            
        Returns:
            Enhanced prompt
        """
        factual_instructions = {
            'general': """Please provide a factual, evidence-based response. 
                         Break down your reasoning into clear steps.
                         If uncertain about any facts, explicitly state the uncertainty.""",
            
            'medical': """Provide medically accurate information based on established evidence.
                         Cite relevant medical guidelines or studies when applicable.
                         Clearly distinguish between established facts and emerging research.""",
            
            'legal': """Provide legally accurate information based on established law.
                       Reference relevant statutes or precedents when applicable.
                       Distinguish between general principles and jurisdiction-specific rules.""",
            
            'financial': """Provide financially accurate information based on established principles.
                          Reference relevant regulations or standards when applicable.
                          Clearly state any assumptions or market conditions."""
        }
        
        instruction = factual_instructions.get(domain, factual_instructions['general'])
        
        return f"""{instruction}

{original_prompt}"""


def create_factual_test_data() -> List[Dict]:
    """Create test data for factual accuracy verification."""
    return [
        {
            'text': "Bird flu, also known as avian influenza, is caused by influenza viruses that primarily affect birds.",
            'label': 'factual',
            'expected_score': 0.9
        },
        {
            'text': "Bird flu can be cured by eating more chicken soup and vitamin C supplements.",
            'label': 'non_factual',
            'expected_score': 0.2
        },
        {
            'text': "The H5N1 strain of bird flu has been documented in multiple countries since 1997.",
            'label': 'factual',
            'expected_score': 0.85
        },
        {
            'text': "All birds are immune to bird flu because they naturally produce antibodies.",
            'label': 'non_factual', 
            'expected_score': 0.1
        }
    ]