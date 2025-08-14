"""Data validation and drift detection system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Metrics for drift detection."""
    timestamp: datetime
    feature_name: str
    drift_score: float
    p_value: float
    is_drift: bool
    drift_type: str  # 'covariate', 'concept', 'prediction'
    severity: str  # 'low', 'medium', 'high', 'critical'
    details: Dict[str, Any]


class DataValidator:
    """Validate input data quality and constraints."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or self._get_default_config()
        self.validation_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'text': {
                'min_length': 10,
                'max_length': 10000,
                'min_words': 3,
                'max_words': 2000,
                'allowed_chars_ratio': 0.95,  # Ratio of standard characters
                'max_special_char_ratio': 0.1,
                'max_uppercase_ratio': 0.5,
                'languages': ['en'],  # Expected languages
            },
            'metadata': {
                'required_fields': [],
                'optional_fields': ['source', 'date', 'author'],
                'field_types': {
                    'source': str,
                    'date': str,
                    'author': str
                }
            },
            'statistical': {
                'outlier_std_threshold': 4,  # Standard deviations for outlier
                'min_entropy': 0.5,  # Minimum text entropy
                'max_repetition_ratio': 0.3  # Maximum repeated substring ratio
            }
        }
    
    def validate_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate text input.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        config = self.config['text']
        
        # Length checks
        if len(text) < config['min_length']:
            issues.append(f"Text too short: {len(text)} < {config['min_length']}")
        
        if len(text) > config['max_length']:
            issues.append(f"Text too long: {len(text)} > {config['max_length']}")
        
        # Word count checks
        words = text.split()
        if len(words) < config['min_words']:
            issues.append(f"Too few words: {len(words)} < {config['min_words']}")
        
        if len(words) > config['max_words']:
            issues.append(f"Too many words: {len(words)} > {config['max_words']}")
        
        # Character ratio checks
        standard_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?;:')
        char_ratio = standard_chars / len(text) if text else 0
        
        if char_ratio < config['allowed_chars_ratio']:
            issues.append(f"Too many special characters: ratio {char_ratio:.2f}")
        
        # Uppercase ratio
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if uppercase_ratio > config['max_uppercase_ratio']:
            issues.append(f"Too many uppercase characters: ratio {uppercase_ratio:.2f}")
        
        # Statistical checks
        stat_config = self.config['statistical']
        
        # Entropy check (text diversity)
        entropy = self._calculate_entropy(text)
        if entropy < stat_config['min_entropy']:
            issues.append(f"Low text entropy: {entropy:.2f} < {stat_config['min_entropy']}")
        
        # Repetition check
        repetition_ratio = self._calculate_repetition_ratio(text)
        if repetition_ratio > stat_config['max_repetition_ratio']:
            issues.append(f"High repetition: {repetition_ratio:.2f}")
        
        is_valid = len(issues) == 0
        
        # Log validation
        self.validation_history.append({
            'timestamp': datetime.utcnow(),
            'text_length': len(text),
            'is_valid': is_valid,
            'issues': issues
        })
        
        return is_valid, issues
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        config = self.config['metadata']
        
        # Check required fields
        for field in config['required_fields']:
            if field not in metadata:
                issues.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in config['field_types'].items():
            if field in metadata:
                if not isinstance(metadata[field], expected_type):
                    issues.append(f"Invalid type for {field}: expected {expected_type.__name__}")
        
        return len(issues) == 0, issues
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0
        
        # Character frequency
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        text_len = len(text)
        
        for count in char_counts.values():
            probability = count / text_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calculate ratio of repeated substrings."""
        if len(text) < 10:
            return 0
        
        # Check for repeated substrings of length 5-20
        repeated_chars = 0
        checked = set()
        
        for length in range(5, min(21, len(text) // 2)):
            for i in range(len(text) - length + 1):
                substring = text[i:i+length]
                
                if substring in checked:
                    continue
                
                checked.add(substring)
                count = text.count(substring)
                
                if count > 1:
                    repeated_chars += length * (count - 1)
        
        return repeated_chars / len(text)
    
    def validate_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Validate batch of inputs.
        
        Args:
            texts: List of texts
            metadata_list: Optional list of metadata
            
        Returns:
            Validation summary
        """
        results = {
            'total': len(texts),
            'valid': 0,
            'invalid': 0,
            'issues_summary': {},
            'invalid_indices': []
        }
        
        for i, text in enumerate(texts):
            is_valid, issues = self.validate_text(text)
            
            if metadata_list and i < len(metadata_list):
                meta_valid, meta_issues = self.validate_metadata(metadata_list[i])
                is_valid = is_valid and meta_valid
                issues.extend(meta_issues)
            
            if is_valid:
                results['valid'] += 1
            else:
                results['invalid'] += 1
                results['invalid_indices'].append(i)
                
                for issue in issues:
                    issue_type = issue.split(':')[0]
                    results['issues_summary'][issue_type] = \
                        results['issues_summary'].get(issue_type, 0) + 1
        
        return results


class DriftDetector:
    """Detect data and concept drift."""
    
    def __init__(
        self,
        reference_window_size: int = 1000,
        detection_window_size: int = 100,
        drift_threshold: float = 0.05
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_window_size: Size of reference window
            detection_window_size: Size of detection window
            drift_threshold: P-value threshold for drift detection
        """
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.drift_threshold = drift_threshold
        
        # Reference data
        self.reference_features = None
        self.reference_predictions = None
        self.reference_embeddings = None
        
        # Recent data windows
        self.recent_features = []
        self.recent_predictions = []
        self.recent_embeddings = []
        
        # Drift history
        self.drift_history = []
        
    def update_reference(
        self,
        features: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None
    ):
        """
        Update reference distribution.
        
        Args:
            features: Feature matrix
            predictions: Model predictions
            embeddings: Text embeddings
        """
        self.reference_features = features[-self.reference_window_size:]
        
        if predictions is not None:
            self.reference_predictions = predictions[-self.reference_window_size:]
        
        if embeddings is not None:
            self.reference_embeddings = embeddings[-self.reference_window_size:]
        
        logger.info(f"Updated reference with {len(features)} samples")
    
    def detect_covariate_drift(self, features: np.ndarray) -> List[DriftMetrics]:
        """
        Detect covariate shift in input features.
        
        Args:
            features: Current feature matrix
            
        Returns:
            List of drift metrics
        """
        if self.reference_features is None:
            return []
        
        drift_metrics = []
        
        # Test each feature dimension
        for i in range(features.shape[1]):
            ref_feature = self.reference_features[:, i]
            curr_feature = features[:, i]
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_feature, curr_feature)
            
            # Determine if drift occurred
            is_drift = p_value < self.drift_threshold
            
            # Calculate severity
            if is_drift:
                if p_value < 0.001:
                    severity = 'critical'
                elif p_value < 0.01:
                    severity = 'high'
                elif p_value < 0.03:
                    severity = 'medium'
                else:
                    severity = 'low'
            else:
                severity = 'none'
            
            metric = DriftMetrics(
                timestamp=datetime.utcnow(),
                feature_name=f'feature_{i}',
                drift_score=ks_stat,
                p_value=p_value,
                is_drift=is_drift,
                drift_type='covariate',
                severity=severity,
                details={
                    'ref_mean': float(np.mean(ref_feature)),
                    'ref_std': float(np.std(ref_feature)),
                    'curr_mean': float(np.mean(curr_feature)),
                    'curr_std': float(np.std(curr_feature)),
                    'ks_statistic': float(ks_stat)
                }
            )
            
            drift_metrics.append(metric)
        
        return drift_metrics
    
    def detect_prediction_drift(
        self,
        predictions: np.ndarray,
        prediction_probs: Optional[np.ndarray] = None
    ) -> DriftMetrics:
        """
        Detect drift in model predictions.
        
        Args:
            predictions: Current predictions
            prediction_probs: Prediction probabilities
            
        Returns:
            Drift metrics for predictions
        """
        if self.reference_predictions is None:
            return None
        
        # Chi-square test for categorical predictions
        ref_counts = np.bincount(self.reference_predictions.astype(int))
        curr_counts = np.bincount(predictions.astype(int))
        
        # Ensure same length
        max_len = max(len(ref_counts), len(curr_counts))
        ref_counts = np.pad(ref_counts, (0, max_len - len(ref_counts)))
        curr_counts = np.pad(curr_counts, (0, max_len - len(curr_counts)))
        
        # Chi-square test
        chi2, p_value = stats.chisquare(curr_counts, ref_counts)
        
        is_drift = p_value < self.drift_threshold
        
        # Calculate severity
        if is_drift:
            if p_value < 0.001:
                severity = 'critical'
            elif p_value < 0.01:
                severity = 'high'
            elif p_value < 0.03:
                severity = 'medium'
            else:
                severity = 'low'
        else:
            severity = 'none'
        
        # Additional metrics if probabilities available
        details = {
            'chi2_statistic': float(chi2),
            'ref_distribution': ref_counts.tolist(),
            'curr_distribution': curr_counts.tolist()
        }
        
        if prediction_probs is not None:
            # Calculate confidence drift
            ref_confidence = np.max(self.reference_predictions, axis=1) \
                if len(self.reference_predictions.shape) > 1 else None
            curr_confidence = np.max(prediction_probs, axis=1)
            
            if ref_confidence is not None:
                conf_ks_stat, conf_p_value = stats.ks_2samp(ref_confidence, curr_confidence)
                details['confidence_drift'] = {
                    'ks_statistic': float(conf_ks_stat),
                    'p_value': float(conf_p_value)
                }
        
        return DriftMetrics(
            timestamp=datetime.utcnow(),
            feature_name='predictions',
            drift_score=chi2,
            p_value=p_value,
            is_drift=is_drift,
            drift_type='prediction',
            severity=severity,
            details=details
        )
    
    def detect_embedding_drift(self, embeddings: np.ndarray) -> DriftMetrics:
        """
        Detect drift in text embeddings using MMD.
        
        Args:
            embeddings: Current embeddings
            
        Returns:
            Drift metrics for embeddings
        """
        if self.reference_embeddings is None:
            return None
        
        # Maximum Mean Discrepancy (MMD) test
        mmd_score = self._compute_mmd(self.reference_embeddings, embeddings)
        
        # Bootstrap to get p-value
        p_value = self._bootstrap_mmd_test(
            self.reference_embeddings,
            embeddings,
            mmd_score
        )
        
        is_drift = p_value < self.drift_threshold
        
        # Calculate severity
        if is_drift:
            if mmd_score > 0.5:
                severity = 'critical'
            elif mmd_score > 0.3:
                severity = 'high'
            elif mmd_score > 0.1:
                severity = 'medium'
            else:
                severity = 'low'
        else:
            severity = 'none'
        
        return DriftMetrics(
            timestamp=datetime.utcnow(),
            feature_name='embeddings',
            drift_score=mmd_score,
            p_value=p_value,
            is_drift=is_drift,
            drift_type='concept',
            severity=severity,
            details={
                'mmd_score': float(mmd_score),
                'ref_size': len(self.reference_embeddings),
                'curr_size': len(embeddings)
            }
        )
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """
        Compute Maximum Mean Discrepancy with RBF kernel.
        
        Args:
            X: Reference samples
            Y: Current samples
            gamma: RBF kernel parameter
            
        Returns:
            MMD score
        """
        XX = self._rbf_kernel(X, X, gamma)
        YY = self._rbf_kernel(Y, Y, gamma)
        XY = self._rbf_kernel(X, Y, gamma)
        
        return XX.mean() + YY.mean() - 2 * XY.mean()
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF kernel matrix."""
        sqdist = np.sum(X**2, axis=1).reshape(-1, 1) + \
                np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-gamma * sqdist)
    
    def _bootstrap_mmd_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        observed_mmd: float,
        n_bootstrap: int = 100
    ) -> float:
        """
        Bootstrap test for MMD significance.
        
        Args:
            X: Reference samples
            Y: Current samples
            observed_mmd: Observed MMD score
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            P-value
        """
        combined = np.vstack([X, Y])
        n_X = len(X)
        
        bootstrap_mmds = []
        for _ in range(n_bootstrap):
            # Shuffle and split
            shuffled = combined[np.random.permutation(len(combined))]
            X_boot = shuffled[:n_X]
            Y_boot = shuffled[n_X:]
            
            mmd_boot = self._compute_mmd(X_boot, Y_boot)
            bootstrap_mmds.append(mmd_boot)
        
        # Calculate p-value
        p_value = np.mean(np.array(bootstrap_mmds) >= observed_mmd)
        return p_value
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of recent drift detections."""
        if not self.drift_history:
            return {'status': 'no_data'}
        
        recent_drift = [d for d in self.drift_history 
                       if d.timestamp > datetime.utcnow() - timedelta(hours=24)]
        
        summary = {
            'total_checks': len(recent_drift),
            'drifts_detected': sum(1 for d in recent_drift if d.is_drift),
            'by_type': {},
            'by_severity': {},
            'critical_features': []
        }
        
        # Aggregate by type
        for drift_type in ['covariate', 'prediction', 'concept']:
            type_drifts = [d for d in recent_drift if d.drift_type == drift_type]
            summary['by_type'][drift_type] = {
                'count': sum(1 for d in type_drifts if d.is_drift),
                'avg_score': np.mean([d.drift_score for d in type_drifts]) if type_drifts else 0
            }
        
        # Aggregate by severity
        for severity in ['low', 'medium', 'high', 'critical']:
            summary['by_severity'][severity] = sum(
                1 for d in recent_drift if d.severity == severity
            )
        
        # Find critical features
        critical = [d for d in recent_drift if d.severity in ['high', 'critical']]
        summary['critical_features'] = list(set(d.feature_name for d in critical))
        
        return summary


# Global instances
data_validator = DataValidator()
drift_detector = DriftDetector()