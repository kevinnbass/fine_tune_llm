"""Mixture of Agents (MoA) implementation for collaborative precision optimization."""

import torch
import json
import yaml
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from pathlib import Path
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an individual agent."""
    agent_type: str
    output: str
    confidence: float
    metadata: Dict[str, Any]


class SpecializedAgent:
    """Base class for specialized agents in MoA."""
    
    def __init__(self, agent_type: str, model_name: str, config: Dict[str, Any]):
        self.agent_type = agent_type
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def initialize(self):
        """Initialize the agent's model and tokenizer."""
        logger.info(f"Initializing {self.agent_type} agent with {self.model_name}")
        
        model_config = self.config['model_options'].get(self.model_name, {})
        model_id = model_config.get('model_id', self.model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
    def process(self, input_text: str, context: Optional[Dict] = None) -> AgentResult:
        """Process input and return agent result."""
        raise NotImplementedError("Subclasses must implement process method")


class FactCheckAgent(SpecializedAgent):
    """Agent specialized in fact-checking for high-stakes accuracy."""
    
    def process(self, input_text: str, context: Optional[Dict] = None) -> AgentResult:
        """
        Fact-check the input text.
        
        Args:
            input_text: Text to fact-check
            context: Optional context information
            
        Returns:
            AgentResult with fact-checking analysis
        """
        # Construct fact-checking prompt
        prompt = f"""You are a fact-checking expert. Analyze the following text for factual accuracy.
        
Text: {input_text}

Provide a JSON response with:
1. factual_accuracy: score from 0 to 1
2. identified_claims: list of claims found
3. verification_status: for each claim (verified/unverified/false)
4. confidence: your confidence in the assessment

Response:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,  # Low temperature for factual tasks
                do_sample=True,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response (with error handling)
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result_json = json.loads(response[json_start:json_end])
            else:
                result_json = {"factual_accuracy": 0.5, "confidence": 0.5}
        except json.JSONDecodeError:
            result_json = {"factual_accuracy": 0.5, "confidence": 0.5}
        
        return AgentResult(
            agent_type="fact_check",
            output=response,
            confidence=result_json.get("confidence", 0.5),
            metadata=result_json
        )


class ClassificationAgent(SpecializedAgent):
    """Agent specialized in classification tasks."""
    
    def process(self, input_text: str, context: Optional[Dict] = None) -> AgentResult:
        """
        Classify the input text.
        
        Args:
            input_text: Text to classify
            context: Optional context with labels, etc.
            
        Returns:
            AgentResult with classification
        """
        labels = context.get("labels", ["relevant", "irrelevant"]) if context else ["relevant", "irrelevant"]
        
        prompt = f"""Classify the following text into one of these categories: {', '.join(labels)}

Text: {input_text}

Provide a JSON response with:
1. classification: the selected category
2. confidence: confidence score from 0 to 1
3. rationale: explanation for the classification
4. abstain: true if uncertain (for high-stakes decisions)

Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result_json = json.loads(response[json_start:json_end])
            else:
                result_json = {"classification": "unknown", "confidence": 0.0, "abstain": True}
        except json.JSONDecodeError:
            result_json = {"classification": "unknown", "confidence": 0.0, "abstain": True}
        
        return AgentResult(
            agent_type="classify",
            output=response,
            confidence=result_json.get("confidence", 0.0),
            metadata=result_json
        )


class MoAOrchestrator:
    """Orchestrator for Mixture of Agents with collaborative decision-making."""
    
    def __init__(self, config_path: str = "configs/llm_lora.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if not self.config['moa']['enabled']:
            raise ValueError("MoA is not enabled in configuration")
        
        self.agents = []
        self.collaboration_mode = self.config['moa']['collaboration_mode']
        self.voting_threshold = self.config['moa']['voting_threshold']
        
    def initialize_agents(self):
        """Initialize all agents based on configuration."""
        logger.info("Initializing MoA agents...")
        
        for agent_config in self.config['moa']['agents']:
            agent_type = agent_config['type']
            model_name = agent_config['model']
            
            # Create appropriate agent based on type
            if agent_type == 'fact_check':
                agent = FactCheckAgent(agent_type, model_name, self.config)
            elif agent_type == 'classify':
                agent = ClassificationAgent(agent_type, model_name, self.config)
            else:
                # Default to classification agent
                agent = ClassificationAgent(agent_type, model_name, self.config)
            
            agent.initialize()
            self.agents.append(agent)
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    def sequential_collaboration(self, input_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Sequential collaboration where agents process in order.
        
        Args:
            input_text: Input text to process
            context: Optional context
            
        Returns:
            Combined result from sequential processing
        """
        current_input = input_text
        all_results = []
        
        for agent in self.agents:
            # Process with current agent
            result = agent.process(current_input, context)
            all_results.append(result)
            
            # Use output as input for next agent (chain-of-thought)
            if result.metadata:
                # Enhance input with previous agent's insights
                current_input = f"{input_text}\n\nPrevious analysis: {json.dumps(result.metadata)}"
        
        # Combine results
        final_confidence = sum(r.confidence for r in all_results) / len(all_results)
        
        # For high-stakes, require high confidence
        if final_confidence < self.voting_threshold:
            return {
                "decision": None,
                "confidence": final_confidence,
                "abstain": True,
                "rationale": "Low confidence from agent collaboration",
                "agent_results": [r.metadata for r in all_results]
            }
        
        # Use last agent's decision with combined confidence
        final_result = all_results[-1].metadata.copy()
        final_result["confidence"] = final_confidence
        final_result["agent_results"] = [r.metadata for r in all_results]
        
        return final_result
    
    def parallel_collaboration(self, input_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Parallel collaboration where agents process independently and vote.
        
        Args:
            input_text: Input text to process
            context: Optional context
            
        Returns:
            Combined result from voting
        """
        # Process with all agents in parallel
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = [
                executor.submit(agent.process, input_text, context)
                for agent in self.agents
            ]
            results = [future.result() for future in futures]
        
        # Voting mechanism for high-stakes precision
        classifications = {}
        total_confidence = 0
        
        for result in results:
            if result.metadata.get("classification"):
                cls = result.metadata["classification"]
                if cls not in classifications:
                    classifications[cls] = []
                classifications[cls].append(result.confidence)
            total_confidence += result.confidence
        
        # Find majority vote
        if not classifications:
            return {
                "decision": None,
                "confidence": 0.0,
                "abstain": True,
                "rationale": "No valid classifications from agents"
            }
        
        # Weight votes by confidence
        weighted_votes = {}
        for cls, confidences in classifications.items():
            weighted_votes[cls] = sum(confidences)
        
        # Get top classification
        best_class = max(weighted_votes, key=weighted_votes.get)
        vote_confidence = weighted_votes[best_class] / total_confidence if total_confidence > 0 else 0
        
        # Check if consensus meets threshold
        if vote_confidence < self.voting_threshold:
            return {
                "decision": None,
                "confidence": vote_confidence,
                "abstain": True,
                "rationale": f"Insufficient consensus (confidence: {vote_confidence:.2f})",
                "agent_results": [r.metadata for r in results],
                "vote_distribution": weighted_votes
            }
        
        return {
            "decision": best_class,
            "confidence": vote_confidence,
            "abstain": False,
            "rationale": f"Consensus from {len(results)} agents",
            "agent_results": [r.metadata for r in results],
            "vote_distribution": weighted_votes
        }
    
    def process(self, input_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process input through MoA based on collaboration mode.
        
        Args:
            input_text: Input text to process
            context: Optional context
            
        Returns:
            Final decision from agent collaboration
        """
        if self.collaboration_mode == "sequential":
            return self.sequential_collaboration(input_text, context)
        elif self.collaboration_mode == "parallel":
            return self.parallel_collaboration(input_text, context)
        else:
            raise ValueError(f"Unknown collaboration mode: {self.collaboration_mode}")
    
    def train_agents(self, train_data, eval_data):
        """
        Train agents with collaborative learning.
        
        Args:
            train_data: Training dataset
            eval_data: Evaluation dataset
        """
        # In practice, implement collaborative training strategies
        # For example, agents can learn from each other's mistakes
        logger.info("Collaborative training not yet implemented")
        pass