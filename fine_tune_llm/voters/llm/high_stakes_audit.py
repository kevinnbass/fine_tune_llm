"""High-stakes auditing features including bias detection, explainability, and procedural alignment."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import re

logger = logging.getLogger(__name__)


class BiasAuditor:
    """Audit and mitigate bias in high-stakes predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bias_config = config.get('high_stakes', {}).get('bias_audit', {})
        self.audit_categories = self.bias_config.get('audit_categories', ['gender', 'race', 'age'])
        self.bias_threshold = self.bias_config.get('bias_threshold', 0.1)
        self.mitigation_weight = self.bias_config.get('mitigation_weight', 1.5)
        
        # Bias detection patterns
        self.bias_patterns = {
            'gender': {
                'male': ['he', 'him', 'his', 'man', 'men', 'male', 'boy', 'gentleman'],
                'female': ['she', 'her', 'hers', 'woman', 'women', 'female', 'girl', 'lady'],
                'neutral': ['they', 'them', 'their', 'person', 'people', 'individual']
            },
            'race': {
                'markers': ['white', 'black', 'asian', 'hispanic', 'latino', 'african', 'european'],
            },
            'age': {
                'young': ['young', 'youth', 'child', 'teen', 'adolescent'],
                'old': ['old', 'elderly', 'senior', 'aged', 'retired'],
            },
            'nationality': {
                'markers': ['american', 'chinese', 'indian', 'british', 'mexican', 'canadian'],
            }
        }
        
        self.audit_log = []
        
    def detect_bias(self, text: str, predictions: torch.Tensor) -> Dict[str, float]:
        """
        Detect bias in text and predictions.
        
        Args:
            text: Input text
            predictions: Model predictions
            
        Returns:
            Dictionary of bias scores by category
        """
        bias_scores = {}
        
        try:
            text_lower = text.lower()
            
            for category in self.audit_categories:
                if category == 'gender':
                    bias_scores[category] = self._detect_gender_bias(text_lower, predictions)
                elif category == 'race':
                    bias_scores[category] = self._detect_race_bias(text_lower, predictions)
                elif category == 'age':
                    bias_scores[category] = self._detect_age_bias(text_lower, predictions)
                elif category == 'nationality':
                    bias_scores[category] = self._detect_nationality_bias(text_lower, predictions)
                else:
                    bias_scores[category] = 0.0
            
            # Log high bias cases
            for category, score in bias_scores.items():
                if score > self.bias_threshold:
                    logger.warning(f"High {category} bias detected: {score:.3f}")
                    self.audit_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'category': category,
                        'score': score,
                        'text_snippet': text[:100]
                    })
            
            return bias_scores
            
        except Exception as e:
            logger.error(f"Error detecting bias: {e}")
            return {cat: 0.0 for cat in self.audit_categories}
    
    def _detect_gender_bias(self, text: str, predictions: torch.Tensor) -> float:
        """Detect gender bias in text and predictions."""
        patterns = self.bias_patterns['gender']
        
        male_count = sum(1 for word in patterns['male'] if word in text)
        female_count = sum(1 for word in patterns['female'] if word in text)
        neutral_count = sum(1 for word in patterns['neutral'] if word in text)
        
        total = male_count + female_count + neutral_count
        if total == 0:
            return 0.0
        
        # Calculate imbalance
        if neutral_count > 0:
            bias_score = 0.0  # Neutral language used
        else:
            imbalance = abs(male_count - female_count) / total
            bias_score = imbalance
        
        # Adjust based on prediction confidence disparity
        if predictions is not None and len(predictions.shape) > 1:
            # Check if predictions vary based on perceived gender
            pred_std = predictions.std().item()
            bias_score = min(1.0, bias_score + pred_std * 0.5)
        
        return bias_score
    
    def _detect_race_bias(self, text: str, predictions: torch.Tensor) -> float:
        """Detect racial bias in text."""
        markers = self.bias_patterns['race']['markers']
        
        marker_count = sum(1 for marker in markers if marker in text)
        
        # Bias increases with explicit racial markers
        bias_score = min(1.0, marker_count * 0.2)
        
        return bias_score
    
    def _detect_age_bias(self, text: str, predictions: torch.Tensor) -> float:
        """Detect age bias in text."""
        patterns = self.bias_patterns['age']
        
        young_count = sum(1 for word in patterns['young'] if word in text)
        old_count = sum(1 for word in patterns['old'] if word in text)
        
        total = young_count + old_count
        if total == 0:
            return 0.0
        
        # Calculate imbalance
        imbalance = abs(young_count - old_count) / total
        return imbalance
    
    def _detect_nationality_bias(self, text: str, predictions: torch.Tensor) -> float:
        """Detect nationality bias in text."""
        markers = self.bias_patterns['nationality']['markers']
        
        marker_count = sum(1 for marker in markers if marker in text)
        
        # Bias increases with nationality markers
        bias_score = min(1.0, marker_count * 0.15)
        
        return bias_score
    
    def compute_bias_mitigation_loss(self, text: str, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute loss to mitigate bias.
        
        Args:
            text: Input text
            predictions: Model predictions
            
        Returns:
            Bias mitigation loss
        """
        if not self.bias_config.get('enabled', False):
            return torch.tensor(0.0)
        
        bias_scores = self.detect_bias(text, predictions)
        
        # Aggregate bias penalty
        total_bias = sum(bias_scores.values()) / len(bias_scores) if bias_scores else 0.0
        
        # Penalty increases with bias level
        bias_penalty = total_bias * self.mitigation_weight
        
        return torch.tensor(bias_penalty, requires_grad=True)
    
    def generate_audit_report(self, save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive bias audit report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_audited': len(self.audit_log),
            'categories': {}
        }
        
        for category in self.audit_categories:
            category_logs = [log for log in self.audit_log if log['category'] == category]
            
            if category_logs:
                scores = [log['score'] for log in category_logs]
                report['categories'][category] = {
                    'count': len(category_logs),
                    'mean_score': np.mean(scores),
                    'max_score': np.max(scores),
                    'violations': len([s for s in scores if s > self.bias_threshold])
                }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Audit report saved to {save_path}")
        
        return report


class ExplainableReasoning:
    """Generate and verify explainable reasoning chains."""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.explain_config = config.get('high_stakes', {}).get('explainable', {})
        self.chain_of_thought = self.explain_config.get('chain_of_thought', True)
        self.min_steps = self.explain_config.get('reasoning_steps', 3)
        self.faithfulness_check = self.explain_config.get('faithfulness_check', True)
        
    def generate_reasoning_chain(self, input_text: str) -> Tuple[str, List[str]]:
        """
        Generate step-by-step reasoning chain.
        
        Args:
            input_text: Input requiring reasoning
            
        Returns:
            Tuple of (final_answer, reasoning_steps)
        """
        try:
            if not self.explain_config.get('enabled', False):
                return "", []
            
            # Prompt for chain-of-thought reasoning
            cot_prompt = f"""Please solve this step-by-step:

{input_text}

Let's think through this carefully:
Step 1:"""
            
            inputs = self.tokenizer(cot_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract reasoning steps
            steps = self._extract_reasoning_steps(response)
            
            # Extract final answer
            final_answer = self._extract_final_answer(response)
            
            return final_answer, steps
            
        except Exception as e:
            logger.error(f"Error generating reasoning chain: {e}")
            return "", []
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract individual reasoning steps from text."""
        steps = []
        
        # Look for step markers
        step_patterns = [
            r'Step \d+:(.+?)(?=Step \d+:|Therefore|Thus|In conclusion|$)',
            r'\d+\.\s+(.+?)(?=\d+\.|Therefore|Thus|In conclusion|$)',
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                steps = [match.strip() for match in matches]
                break
        
        # Fallback: split by sentences
        if not steps:
            sentences = text.split('.')
            steps = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return steps[:self.min_steps] if len(steps) > self.min_steps else steps
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from reasoning text."""
        # Look for conclusion markers
        conclusion_markers = [
            r'Therefore,?\s+(.+?)$',
            r'Thus,?\s+(.+?)$',
            r'In conclusion,?\s+(.+?)$',
            r'The answer is:?\s+(.+?)$',
            r'Final answer:?\s+(.+?)$',
        ]
        
        for marker in conclusion_markers:
            match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: last sentence
        sentences = text.split('.')
        if sentences:
            return sentences[-1].strip()
        
        return text
    
    def verify_faithfulness(self, reasoning_steps: List[str], final_answer: str) -> Tuple[bool, float]:
        """
        Verify that reasoning faithfully leads to conclusion.
        
        Args:
            reasoning_steps: List of reasoning steps
            final_answer: Final conclusion
            
        Returns:
            Tuple of (is_faithful, faithfulness_score)
        """
        if not self.faithfulness_check or not reasoning_steps:
            return True, 1.0
        
        try:
            # Create prompt to verify connection
            steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(reasoning_steps)])
            
            verify_prompt = f"""Given these reasoning steps:
{steps_text}

Does the following conclusion logically follow?
Conclusion: {final_answer}

Answer with Yes or No and explain briefly:"""
            
            inputs = self.tokenizer(verify_prompt, return_tensors="pt", truncation=True, max_length=768)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                )
            
            verification = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
            
            # Check verification result
            is_faithful = 'yes' in verification[:20]  # Check early in response
            
            # Calculate faithfulness score based on confidence
            if 'definitely' in verification or 'clearly' in verification:
                score = 1.0 if is_faithful else 0.0
            elif 'probably' in verification or 'likely' in verification:
                score = 0.7 if is_faithful else 0.3
            else:
                score = 0.5
            
            return is_faithful, score
            
        except Exception as e:
            logger.error(f"Error verifying faithfulness: {e}")
            return False, 0.0


class ProceduralAlignment:
    """Ensure alignment with domain-specific procedures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.procedural_config = config.get('high_stakes', {}).get('procedural', {})
        self.domain = self.procedural_config.get('domain', 'general')
        self.compliance_weight = self.procedural_config.get('compliance_weight', 2.0)
        
        # Load procedures
        self.procedures = self._load_procedures()
        
    def _load_procedures(self) -> Dict[str, List[str]]:
        """Load domain-specific procedures."""
        procedure_file = self.procedural_config.get('procedure_file')
        
        # Default procedures if file not available
        default_procedures = {
            'medical': [
                'Verify patient information before any recommendation',
                'Check for contraindications and allergies',
                'Reference established medical guidelines',
                'Include appropriate disclaimers',
                'Recommend consultation with healthcare provider'
            ],
            'legal': [
                'Verify jurisdiction before legal interpretation',
                'Cite relevant statutes or case law',
                'Include disclaimers about not providing legal advice',
                'Recommend consultation with qualified attorney',
                'Note that laws vary by jurisdiction'
            ],
            'financial': [
                'Verify current regulations and compliance requirements',
                'Include risk disclaimers',
                'Note that past performance does not guarantee future results',
                'Recommend consultation with financial advisor',
                'Disclose any potential conflicts of interest'
            ],
            'general': [
                'Verify factual claims',
                'Provide balanced information',
                'Include appropriate disclaimers',
                'Cite sources when applicable'
            ]
        }
        
        if procedure_file and Path(procedure_file).exists():
            try:
                import yaml
                with open(procedure_file, 'r') as f:
                    loaded_procedures = yaml.safe_load(f)
                return loaded_procedures
            except Exception as e:
                logger.error(f"Error loading procedures: {e}")
        
        return default_procedures
    
    def check_compliance(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Check compliance with domain procedures.
        
        Args:
            text: Text to check for compliance
            
        Returns:
            Tuple of (is_compliant, compliance_score, missing_procedures)
        """
        if not self.procedural_config.get('enabled', False):
            return True, 1.0, []
        
        procedures = self.procedures.get(self.domain, self.procedures['general'])
        text_lower = text.lower()
        
        compliant_procedures = []
        missing_procedures = []
        
        for procedure in procedures:
            # Simple keyword matching (could be enhanced with NLP)
            key_terms = self._extract_key_terms(procedure)
            
            if any(term in text_lower for term in key_terms):
                compliant_procedures.append(procedure)
            else:
                missing_procedures.append(procedure)
        
        compliance_score = len(compliant_procedures) / len(procedures) if procedures else 0.0
        is_compliant = compliance_score >= 0.8  # 80% threshold
        
        if not is_compliant:
            logger.warning(f"Low procedural compliance: {compliance_score:.2f}")
            logger.debug(f"Missing procedures: {missing_procedures}")
        
        return is_compliant, compliance_score, missing_procedures
    
    def _extract_key_terms(self, procedure: str) -> List[str]:
        """Extract key terms from procedure for matching."""
        # Simple approach: extract important words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'with', 'for', 'before', 'after'}
        words = procedure.lower().split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 3]
        return key_terms
    
    def enhance_with_procedures(self, text: str) -> str:
        """
        Enhance text with required procedures.
        
        Args:
            text: Original text
            
        Returns:
            Enhanced text with procedures
        """
        if not self.procedural_config.get('enabled', False):
            return text
        
        is_compliant, score, missing = self.check_compliance(text)
        
        if is_compliant:
            return text
        
        # Add missing procedures
        procedures = self.procedures.get(self.domain, self.procedures['general'])
        
        enhanced = f"""{text}

Important {self.domain.title()} Considerations:
"""
        
        for i, proc in enumerate(missing[:3], 1):  # Add top 3 missing
            enhanced += f"{i}. {proc}\n"
        
        return enhanced


class VerifiableTraining:
    """Create verifiable audit trail for training process."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verifiable_config = config.get('high_stakes', {}).get('verifiable', {})
        self.audit_log_path = Path(self.verifiable_config.get('audit_log', 'artifacts/audit_trail.jsonl'))
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def hash_artifact(self, artifact: Any, artifact_type: str) -> str:
        """
        Create cryptographic hash of training artifact.
        
        Args:
            artifact: Artifact to hash (model, data, config, etc.)
            artifact_type: Type of artifact
            
        Returns:
            Hash string
        """
        if not self.verifiable_config.get('hash_artifacts', False):
            return ""
        
        try:
            if artifact_type == 'model':
                # Hash model parameters
                param_str = ""
                for name, param in artifact.named_parameters():
                    param_str += f"{name}:{param.data.cpu().numpy().tobytes()}"
                hash_input = param_str.encode()
            elif artifact_type == 'data':
                # Hash data
                hash_input = json.dumps(artifact, sort_keys=True).encode()
            elif artifact_type == 'config':
                # Hash configuration
                hash_input = json.dumps(artifact, sort_keys=True).encode()
            else:
                # Generic hash
                hash_input = str(artifact).encode()
            
            return hashlib.sha256(hash_input).hexdigest()
            
        except Exception as e:
            logger.error(f"Error hashing artifact: {e}")
            return ""
    
    def log_training_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log training event to audit trail.
        
        Args:
            event_type: Type of event
            details: Event details
        """
        if not self.verifiable_config.get('enabled', False):
            return
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        # Add hash if cryptographic proof enabled
        if self.verifiable_config.get('cryptographic_proof', False):
            event['hash'] = hashlib.sha256(
                json.dumps(event, sort_keys=True).encode()
            ).hexdigest()
        
        # Append to audit log
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")
    
    def create_training_proof(self, model, train_data, config) -> Dict[str, str]:
        """
        Create comprehensive proof of training.
        
        Args:
            model: Trained model
            train_data: Training data
            config: Training configuration
            
        Returns:
            Dictionary of proof elements
        """
        if not self.verifiable_config.get('cryptographic_proof', False):
            return {}
        
        proof = {
            'timestamp': datetime.now().isoformat(),
            'model_hash': self.hash_artifact(model, 'model'),
            'data_hash': self.hash_artifact(train_data, 'data'),
            'config_hash': self.hash_artifact(config, 'config'),
        }
        
        # Create composite proof
        composite = json.dumps(proof, sort_keys=True)
        proof['composite_hash'] = hashlib.sha256(composite.encode()).hexdigest()
        
        # Log proof
        self.log_training_event('training_proof', proof)
        
        return proof
    
    def verify_training(self, model, proof: Dict[str, str]) -> bool:
        """
        Verify training using proof.
        
        Args:
            model: Model to verify
            proof: Training proof
            
        Returns:
            True if verification passes
        """
        try:
            current_model_hash = self.hash_artifact(model, 'model')
            
            if current_model_hash != proof.get('model_hash'):
                logger.error("Model hash mismatch")
                return False
            
            # Verify composite proof
            proof_copy = proof.copy()
            stored_composite = proof_copy.pop('composite_hash', '')
            
            composite = json.dumps(proof_copy, sort_keys=True)
            computed_composite = hashlib.sha256(composite.encode()).hexdigest()
            
            if computed_composite != stored_composite:
                logger.error("Composite proof mismatch")
                return False
            
            logger.info("Training verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying training: {e}")
            return False