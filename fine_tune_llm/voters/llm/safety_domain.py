"""Safety tools and domain-specific fine-tuning modules."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import yaml
from pathlib import Path
import json
import hashlib
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class DomainSpecificWrapper:
    """Wrapper for domain-specific fine-tuning (medical, legal, financial)."""
    
    def __init__(self, base_model, domain: str, config: Dict[str, Any]):
        self.base_model = base_model
        self.domain = domain
        self.config = config
        self.domain_config = config['domain_specific']
        self.safety_constraints = self.domain_config['safety_constraints']
        self.factuality_checks = self.domain_config['factuality_checks']
        
        # Domain-specific prompts and constraints
        self.domain_prompts = {
            'medical': """You are a medical AI assistant. Provide accurate, evidence-based information.
                         Always include disclaimers about consulting healthcare professionals.
                         Never provide diagnoses or treatment recommendations without proper context.""",
            'legal': """You are a legal information assistant. Provide general legal information only.
                       Always include disclaimers about consulting qualified attorneys.
                       Never provide specific legal advice for individual cases.""",
            'financial': """You are a financial information assistant. Provide educational content only.
                          Always include disclaimers about consulting financial advisors.
                          Never provide specific investment advice.""",
        }
        
        self.domain_constraints = {
            'medical': ['diagnosis', 'prescription', 'treatment'],
            'legal': ['legal advice', 'representation', 'litigation'],
            'financial': ['investment advice', 'trading recommendations', 'guarantees'],
        }
        
    def add_domain_prompt(self, input_text: str) -> str:
        """Add domain-specific prompt to input."""
        domain_prompt = self.domain_prompts.get(self.domain, "")
        return f"{domain_prompt}\n\n{input_text}"
    
    def check_safety_constraints(self, output_text: str) -> Tuple[bool, str]:
        """
        Check if output violates domain-specific safety constraints.
        
        Args:
            output_text: Generated output text
            
        Returns:
            Tuple of (is_safe, violation_reason)
        """
        if not self.safety_constraints:
            return True, ""
        
        constraints = self.domain_constraints.get(self.domain, [])
        
        for constraint in constraints:
            if constraint.lower() in output_text.lower():
                return False, f"Output contains prohibited content: {constraint}"
        
        return True, ""
    
    def apply_factuality_check(self, output_text: str) -> float:
        """
        Apply domain-specific factuality checking.
        
        Args:
            output_text: Generated output text
            
        Returns:
            Factuality score (0-1)
        """
        if not self.factuality_checks:
            return 1.0
        
        # Domain-specific factuality markers
        factuality_markers = {
            'medical': ['studies show', 'research indicates', 'evidence suggests'],
            'legal': ['according to law', 'statute states', 'precedent establishes'],
            'financial': ['market data shows', 'historical performance', 'analysis indicates'],
        }
        
        markers = factuality_markers.get(self.domain, [])
        marker_count = sum(1 for marker in markers if marker in output_text.lower())
        
        # Simple heuristic: more factual markers = higher factuality
        factuality_score = min(1.0, marker_count * 0.3 + 0.4)
        
        return factuality_score
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass with domain-specific processing."""
        # Add domain prompt processing if needed
        outputs = self.base_model(input_ids, attention_mask, **kwargs)
        
        # Apply domain-specific post-processing
        if hasattr(outputs, 'logits'):
            # Could apply domain-specific token filtering here
            pass
        
        return outputs


class SafetyGuardWrapper:
    """Wrapper for safety tools like Llama Guard and ShieldGemma."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llama_guard_config = config['safety_tools']['llama_guard']
        self.shield_gemma_config = config['safety_tools']['shield_gemma']
        self.llama_guard = None
        self.shield_gemma = None
        
        if self.llama_guard_config['enabled']:
            self.initialize_llama_guard()
        if self.shield_gemma_config['enabled']:
            self.initialize_shield_gemma()
    
    def initialize_llama_guard(self):
        """Initialize Llama Guard safety model."""
        logger.info("Initializing Llama Guard...")
        
        model_path = self.llama_guard_config['model_path']
        
        try:
            self.llama_guard_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.llama_guard = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Llama Guard initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Llama Guard: {e}")
            self.llama_guard = None
    
    def initialize_shield_gemma(self):
        """Initialize ShieldGemma safety model."""
        logger.info("Initializing ShieldGemma...")
        
        model_path = self.shield_gemma_config['model_path']
        
        try:
            self.shield_gemma_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.shield_gemma = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("ShieldGemma initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ShieldGemma: {e}")
            self.shield_gemma = None
    
    def check_safety_llama_guard(self, text: str) -> Tuple[bool, float, str]:
        """
        Check text safety using Llama Guard.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_safe, safety_score, category)
        """
        if not self.llama_guard:
            return True, 1.0, "not_checked"
        
        # Format input for Llama Guard
        prompt = f"""Task: Check if the following text is safe and appropriate.
        
Text: {text}

Categories to check:
1. Harmful content
2. Hate speech
3. Violence
4. Sexual content
5. Illegal activities
6. Personal information
7. Misinformation

Response format: SAFE or UNSAFE (category)"""
        
        inputs = self.llama_guard_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        with torch.no_grad():
            outputs = self.llama_guard.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1
            )
        
        response = self.llama_guard_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        if "SAFE" in response.upper():
            return True, 1.0, "safe"
        else:
            # Extract category if present
            category = "unsafe"
            if "(" in response and ")" in response:
                category = response[response.find("(")+1:response.find(")")]
            
            return False, 0.0, category
    
    def check_safety_shield_gemma(self, text: str) -> Tuple[bool, Dict[str, float]]:
        """
        Check text safety using ShieldGemma.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_safe, category_scores)
        """
        if not self.shield_gemma:
            return True, {}
        
        categories = self.shield_gemma_config['categories']
        category_scores = {}
        
        for category in categories:
            prompt = f"Is the following text {category}? Text: {text}\nAnswer: "
            
            inputs = self.shield_gemma_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            with torch.no_grad():
                outputs = self.shield_gemma.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1
                )
            
            response = self.shield_gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Simple scoring based on response
            if "yes" in response.lower():
                category_scores[category] = 0.0  # Unsafe
            else:
                category_scores[category] = 1.0  # Safe
        
        # Overall safety is minimum of all categories
        overall_safe = all(score > 0.5 for score in category_scores.values())
        
        return overall_safe, category_scores
    
    def filter_output(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Filter output text for safety.
        
        Args:
            text: Text to filter
            
        Returns:
            Tuple of (filtered_text, safety_metadata)
        """
        metadata = {}
        
        # Check with Llama Guard
        if self.llama_guard:
            is_safe_lg, score_lg, category_lg = self.check_safety_llama_guard(text)
            metadata['llama_guard'] = {
                'safe': is_safe_lg,
                'score': score_lg,
                'category': category_lg
            }
            
            if not is_safe_lg:
                logger.warning(f"Llama Guard flagged text as unsafe: {category_lg}")
                if score_lg < self.llama_guard_config['threshold']:
                    return "[Content filtered for safety reasons]", metadata
        
        # Check with ShieldGemma
        if self.shield_gemma:
            is_safe_sg, scores_sg = self.check_safety_shield_gemma(text)
            metadata['shield_gemma'] = {
                'safe': is_safe_sg,
                'category_scores': scores_sg
            }
            
            if not is_safe_sg:
                logger.warning(f"ShieldGemma flagged text as unsafe")
                return "[Content filtered for safety reasons]", metadata
        
        return text, metadata


class MemorizationMitigator:
    """Mitigate over-memorization in fine-tuning."""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.mem_config = config['memorization']
        self.regularization_weight = self.mem_config['regularization_weight']
        self.diversity_penalty = self.mem_config['diversity_penalty']
        self.generalization_threshold = self.mem_config['generalization_threshold']
        
        # Track seen examples to detect memorization
        self.seen_examples = set()
        self.example_counts = {}
        
    def compute_memorization_loss(self, outputs, inputs, labels):
        """
        Compute loss that penalizes memorization.
        
        Args:
            outputs: Model outputs
            inputs: Input ids
            labels: Target labels
            
        Returns:
            Memorization-aware loss
        """
        base_loss = outputs.loss
        
        # Add regularization to prevent overfitting
        l2_reg = 0
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2)
        
        reg_loss = self.regularization_weight * l2_reg
        
        # Add diversity penalty to encourage varied outputs
        if hasattr(outputs, 'logits'):
            # Entropy of output distribution
            probs = torch.softmax(outputs.logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            
            # Penalize low entropy (memorized/overconfident outputs)
            diversity_loss = -self.diversity_penalty * entropy.mean()
        else:
            diversity_loss = 0
        
        total_loss = base_loss + reg_loss + diversity_loss
        
        return total_loss
    
    def detect_memorization(self, input_text: str, output_text: str) -> bool:
        """
        Detect if model is memorizing specific examples.
        
        Args:
            input_text: Input text
            output_text: Generated output
            
        Returns:
            True if memorization detected
        """
        # Create hash of input-output pair
        pair_hash = hashlib.md5(
            f"{input_text}||{output_text}".encode()
        ).hexdigest()
        
        # Check if we've seen this exact pair before
        if pair_hash in self.seen_examples:
            self.example_counts[pair_hash] = self.example_counts.get(pair_hash, 0) + 1
            
            # If seen too many times, likely memorized
            if self.example_counts[pair_hash] > 3:
                return True
        else:
            self.seen_examples.add(pair_hash)
            self.example_counts[pair_hash] = 1
        
        return False
    
    def apply_dropout_augmentation(self, model):
        """Apply additional dropout during training to prevent memorization."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                # Increase dropout rate
                module.p = min(0.5, module.p * 1.5)


class VerifiableFinetuning:
    """Verifiable fine-tuning with cryptographic proofs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verifiable_config = config['verifiable']
        self.backdoor_key = self.verifiable_config['backdoor_key']
        self.audit_log_path = Path(self.verifiable_config['audit_log_path'])
        self.audit_log = []
        
    def generate_proof_of_training(self, model, dataset) -> str:
        """
        Generate cryptographic proof of training.
        
        Args:
            model: Trained model
            dataset: Training dataset
            
        Returns:
            Proof string
        """
        # Create hash of model parameters
        param_hashes = []
        for name, param in model.named_parameters():
            param_hash = hashlib.sha256(
                param.data.cpu().numpy().tobytes()
            ).hexdigest()
            param_hashes.append(f"{name}:{param_hash}")
        
        model_hash = hashlib.sha256(
            "".join(param_hashes).encode()
        ).hexdigest()
        
        # Create hash of dataset
        dataset_hash = hashlib.sha256(
            json.dumps(dataset.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        
        # Combine with timestamp
        timestamp = datetime.now().isoformat()
        
        # Create proof
        proof = {
            'model_hash': model_hash,
            'dataset_hash': dataset_hash,
            'timestamp': timestamp,
            'config_hash': hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode()
            ).hexdigest()
        }
        
        # Sign with backdoor key if available
        if self.backdoor_key:
            proof['signature'] = self.sign_proof(proof)
        
        return json.dumps(proof, indent=2)
    
    def sign_proof(self, proof: Dict) -> str:
        """Sign proof with backdoor key."""
        # Simplified signature (in production, use proper cryptography)
        proof_str = json.dumps(proof, sort_keys=True)
        signature = hashlib.sha256(
            f"{proof_str}{self.backdoor_key}".encode()
        ).hexdigest()
        return signature
    
    def verify_training(self, model, proof_str: str) -> bool:
        """
        Verify that model was trained as claimed.
        
        Args:
            model: Model to verify
            proof_str: Proof string
            
        Returns:
            True if verification passes
        """
        proof = json.loads(proof_str)
        
        # Verify model hash
        param_hashes = []
        for name, param in model.named_parameters():
            param_hash = hashlib.sha256(
                param.data.cpu().numpy().tobytes()
            ).hexdigest()
            param_hashes.append(f"{name}:{param_hash}")
        
        current_model_hash = hashlib.sha256(
            "".join(param_hashes).encode()
        ).hexdigest()
        
        if current_model_hash != proof['model_hash']:
            logger.warning("Model hash mismatch")
            return False
        
        # Verify signature if present
        if 'signature' in proof and self.backdoor_key:
            expected_signature = self.sign_proof(
                {k: v for k, v in proof.items() if k != 'signature'}
            )
            if proof['signature'] != expected_signature:
                logger.warning("Signature verification failed")
                return False
        
        return True
    
    def log_training_event(self, event: Dict):
        """Log training event for audit trail."""
        event['timestamp'] = datetime.now().isoformat()
        self.audit_log.append(event)
        
        # Save to file
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(event) + '\n')