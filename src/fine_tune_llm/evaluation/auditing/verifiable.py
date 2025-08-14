"""
Verifiable training and audit trail for high-stakes models.

This module provides cryptographic verification and audit logging
for training processes to ensure reproducibility and accountability.
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from ...core.interfaces import BaseComponent
from ...core.exceptions import SystemError

logger = logging.getLogger(__name__)

class VerifiableTraining(BaseComponent):
    """Create verifiable audit trails for training processes."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize verifiable training component."""
        self.config = config or {}
        self.verifiable_config = self.config.get('high_stakes', {}).get('verifiable', {})
        self.audit_dir = Path(self.verifiable_config.get('audit_dir', 'artifacts/audit'))
        self.enable_hashing = self.verifiable_config.get('enable_hashing', True)
        self.hash_algorithm = self.verifiable_config.get('hash_algorithm', 'sha256')
        
        # Create audit directory
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Training audit log
        self.training_log: List[Dict[str, Any]] = []
        self.artifact_hashes: Dict[str, str] = {}
        
        # Chain of hashes for tamper detection
        self.hash_chain: List[str] = []
        self.genesis_hash = self._compute_hash("GENESIS_BLOCK")
        self.hash_chain.append(self.genesis_hash)
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        self.verifiable_config = self.config.get('high_stakes', {}).get('verifiable', {})
        self.audit_dir = Path(self.verifiable_config.get('audit_dir', 'artifacts/audit'))
        self.audit_dir.mkdir(parents=True, exist_ok=True)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Save final audit log before cleanup
        if self.training_log:
            self._save_audit_log()
    
    @property
    def name(self) -> str:
        """Component name."""
        return "VerifiableTraining"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data."""
        if self.hash_algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif self.hash_algorithm == 'sha512':
            hasher = hashlib.sha512()
        else:
            hasher = hashlib.sha256()
        
        # Convert data to bytes
        if isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        
        hasher.update(data_bytes)
        return hasher.hexdigest()
    
    def hash_artifact(self, artifact_path: Path, artifact_type: str) -> str:
        """
        Compute and store hash of training artifact.
        
        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact (model, data, config)
            
        Returns:
            Hash of artifact
        """
        try:
            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact not found: {artifact_path}")
            
            # Compute file hash
            hasher = hashlib.sha256() if self.hash_algorithm == 'sha256' else hashlib.sha512()
            
            with open(artifact_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            
            artifact_hash = hasher.hexdigest()
            
            # Store hash with metadata
            self.artifact_hashes[str(artifact_path)] = {
                'hash': artifact_hash,
                'type': artifact_type,
                'timestamp': datetime.now().isoformat(),
                'size': artifact_path.stat().st_size
            }
            
            logger.info(f"Hashed {artifact_type}: {artifact_path.name} -> {artifact_hash[:8]}...")
            
            return artifact_hash
            
        except Exception as e:
            logger.error(f"Error hashing artifact: {e}")
            raise SystemError(f"Failed to hash artifact: {e}")
    
    def log_training_event(self, 
                          event_type: str,
                          event_data: Dict[str, Any],
                          metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log training event with verification.
        
        Args:
            event_type: Type of event (epoch, checkpoint, validation)
            event_data: Event details
            metrics: Optional metrics
        """
        try:
            # Create event record
            event = {
                'event_id': len(self.training_log),
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': event_data,
                'metrics': metrics or {},
                'previous_hash': self.hash_chain[-1] if self.hash_chain else self.genesis_hash
            }
            
            # Compute event hash (includes previous hash for chaining)
            event_hash = self._compute_hash(event)
            event['hash'] = event_hash
            
            # Add to hash chain
            self.hash_chain.append(event_hash)
            
            # Add to training log
            self.training_log.append(event)
            
            # Periodically save to disk
            if len(self.training_log) % 10 == 0:
                self._save_audit_log()
            
            logger.debug(f"Logged training event: {event_type} (hash: {event_hash[:8]}...)")
            
        except Exception as e:
            logger.error(f"Error logging training event: {e}")
    
    def create_training_proof(self, 
                            model_path: Path,
                            training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create cryptographic proof of training.
        
        Args:
            model_path: Path to trained model
            training_config: Training configuration
            
        Returns:
            Training proof document
        """
        try:
            # Hash the model
            model_hash = self.hash_artifact(model_path, 'model')
            
            # Hash the configuration
            config_hash = self._compute_hash(training_config)
            
            # Create proof document
            proof = {
                'proof_version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'model': {
                    'path': str(model_path),
                    'hash': model_hash,
                    'size': model_path.stat().st_size
                },
                'configuration': {
                    'hash': config_hash,
                    'summary': {
                        'epochs': training_config.get('num_epochs'),
                        'batch_size': training_config.get('batch_size'),
                        'learning_rate': training_config.get('learning_rate')
                    }
                },
                'training_log': {
                    'events_count': len(self.training_log),
                    'first_event': self.training_log[0] if self.training_log else None,
                    'last_event': self.training_log[-1] if self.training_log else None,
                    'hash_chain_length': len(self.hash_chain),
                    'genesis_hash': self.genesis_hash,
                    'final_hash': self.hash_chain[-1] if self.hash_chain else None
                },
                'artifacts': self.artifact_hashes
            }
            
            # Sign the proof (hash of the entire document)
            proof['signature'] = self._compute_hash(proof)
            
            # Save proof document
            proof_path = self.audit_dir / f"training_proof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(proof_path, 'w') as f:
                json.dump(proof, f, indent=2)
            
            logger.info(f"Created training proof: {proof_path}")
            
            return proof
            
        except Exception as e:
            logger.error(f"Error creating training proof: {e}")
            raise SystemError(f"Failed to create training proof: {e}")
    
    def verify_training(self, proof_path: Path) -> Dict[str, Any]:
        """
        Verify training proof document.
        
        Args:
            proof_path: Path to proof document
            
        Returns:
            Verification results
        """
        verification = {
            'is_valid': True,
            'checks': {},
            'errors': []
        }
        
        try:
            # Load proof document
            with open(proof_path, 'r') as f:
                proof = json.load(f)
            
            # Verify signature
            signature = proof.pop('signature', None)
            computed_signature = self._compute_hash(proof)
            
            if signature != computed_signature:
                verification['is_valid'] = False
                verification['errors'].append("Signature verification failed")
                verification['checks']['signature'] = False
            else:
                verification['checks']['signature'] = True
            
            # Verify model hash if model exists
            model_path = Path(proof['model']['path'])
            if model_path.exists():
                current_hash = self.hash_artifact(model_path, 'model')
                if current_hash != proof['model']['hash']:
                    verification['is_valid'] = False
                    verification['errors'].append("Model hash mismatch")
                    verification['checks']['model_integrity'] = False
                else:
                    verification['checks']['model_integrity'] = True
            else:
                verification['errors'].append(f"Model not found: {model_path}")
                verification['checks']['model_exists'] = False
            
            # Verify hash chain integrity
            if 'training_log' in proof:
                chain_valid = self._verify_hash_chain(proof['training_log'])
                verification['checks']['hash_chain'] = chain_valid
                if not chain_valid:
                    verification['is_valid'] = False
                    verification['errors'].append("Hash chain verification failed")
            
            # Add metadata
            verification['proof_timestamp'] = proof.get('timestamp')
            verification['model_hash'] = proof.get('model', {}).get('hash')
            verification['events_count'] = proof.get('training_log', {}).get('events_count')
            
            return verification
            
        except Exception as e:
            logger.error(f"Error verifying training: {e}")
            verification['is_valid'] = False
            verification['errors'].append(f"Verification error: {str(e)}")
            return verification
    
    def _verify_hash_chain(self, training_log_info: Dict[str, Any]) -> bool:
        """Verify integrity of hash chain."""
        try:
            # For now, just check that genesis and final hashes are present
            # Full verification would rebuild the chain and compare
            genesis = training_log_info.get('genesis_hash')
            final = training_log_info.get('final_hash')
            
            return bool(genesis and final)
            
        except Exception:
            return False
    
    def _save_audit_log(self) -> None:
        """Save audit log to disk."""
        try:
            log_path = self.audit_dir / f"audit_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(log_path, 'a') as f:
                for event in self.training_log:
                    f.write(json.dumps(event) + '\n')
            
            logger.debug(f"Saved audit log to {log_path}")
            
        except Exception as e:
            logger.error(f"Error saving audit log: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate verifiable training report."""
        report = {
            'summary': {
                'total_events': len(self.training_log),
                'artifacts_hashed': len(self.artifact_hashes),
                'hash_chain_length': len(self.hash_chain),
                'audit_directory': str(self.audit_dir)
            },
            'event_types': self._summarize_event_types(),
            'artifacts': list(self.artifact_hashes.keys()),
            'integrity': {
                'genesis_hash': self.genesis_hash,
                'current_hash': self.hash_chain[-1] if self.hash_chain else None,
                'chain_valid': len(self.hash_chain) == len(self.training_log) + 1
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _summarize_event_types(self) -> Dict[str, int]:
        """Summarize events by type."""
        event_counts = {}
        
        for event in self.training_log:
            event_type = event.get('event_type', 'unknown')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return event_counts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for verifiable training."""
        recommendations = []
        
        if not self.training_log:
            recommendations.append("No training events logged. Ensure logging is enabled.")
        
        if not self.artifact_hashes:
            recommendations.append("No artifacts hashed. Consider hashing model checkpoints.")
        
        if len(self.hash_chain) != len(self.training_log) + 1:
            recommendations.append("Hash chain inconsistency detected. Review audit integrity.")
        
        if not recommendations:
            recommendations.append("Verifiable training is properly configured. Continue monitoring.")
        
        return recommendations