"""TF-IDF + SVM voter."""

import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import logging

from .calibrate import compute_ece

logger = logging.getLogger(__name__)


class TfidfSVMVoter:
    """TF-IDF + SVM classical voter."""
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
        class_weight: str = 'balanced',
        abstain_threshold: float = 0.7,
        calibration_method: str = 'isotonic'
    ):
        """
        Initialize TF-IDF SVM voter.
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency
            C: Regularization parameter
            class_weight: Class weight strategy
            abstain_threshold: Threshold for abstention
            calibration_method: 'isotonic' or 'sigmoid'
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.C = C
        self.class_weight = class_weight
        self.abstain_threshold = abstain_threshold
        self.calibration_method = calibration_method
        
        # Build pipeline
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english',
            sublinear_tf=True
        )
        
        # SVM with calibration wrapper
        base_svm = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=2000,
            random_state=42
        )
        
        self.classifier = CalibratedClassifierCV(
            base_svm,
            method=calibration_method,
            cv=3
        )
        
        self.class_names = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
        self.is_fitted = False
        
    def fit(
        self,
        texts: List[str],
        labels: np.ndarray,
        validation_texts: Optional[List[str]] = None,
        validation_labels: Optional[np.ndarray] = None
    ):
        """
        Train the model.
        
        Args:
            texts: Training texts
            labels: Training labels
            validation_texts: Optional validation texts
            validation_labels: Optional validation labels
        """
        logger.info("Training TF-IDF + SVM model")
        
        # Fit vectorizer
        X_train = self.vectorizer.fit_transform(texts)
        
        # Fit classifier with calibration
        self.classifier.fit(X_train, labels)
        
        # Evaluate calibration if validation data provided
        if validation_texts is not None and validation_labels is not None:
            X_val = self.vectorizer.transform(validation_texts)
            probs = self.classifier.predict_proba(X_val)
            max_probs = probs.max(axis=1)
            
            ece = compute_ece(validation_labels, max_probs)
            logger.info(f"Validation ECE: {ece:.4f}")
        
        self.is_fitted = True
        logger.info("TF-IDF + SVM training complete")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make prediction for single text.
        
        Args:
            text: Input text
            
        Returns:
            Voter output dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        
        # Transform and predict
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        
        # Check abstention threshold
        max_prob = probs.max()
        if max_prob < self.abstain_threshold:
            return {
                'probs': {},
                'abstain': True,
                'reason': f'Low confidence: {max_prob:.3f} < {self.abstain_threshold}',
                'latency_ms': (time.time() - start_time) * 1000,
                'cost_cents': 0.001,
                'model_id': 'tfidf_svm'
            }
        
        # Build probability dictionary
        probs_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        
        return {
            'probs': probs_dict,
            'abstain': False,
            'decision': self.class_names[probs.argmax()],
            'max_prob': float(max_prob),
            'latency_ms': (time.time() - start_time) * 1000,
            'cost_cents': 0.001,
            'model_id': 'tfidf_svm'
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions for batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of voter outputs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        results = []
        start_time = time.time()
        
        # Transform all texts
        X = self.vectorizer.transform(texts)
        
        # Predict probabilities
        all_probs = self.classifier.predict_proba(X)
        
        batch_time = (time.time() - start_time) * 1000
        per_sample_time = batch_time / len(texts)
        
        # Process each prediction
        for i, probs in enumerate(all_probs):
            max_prob = probs.max()
            
            if max_prob < self.abstain_threshold:
                result = {
                    'probs': {},
                    'abstain': True,
                    'reason': f'Low confidence: {max_prob:.3f} < {self.abstain_threshold}',
                    'latency_ms': per_sample_time,
                    'cost_cents': 0.001,
                    'model_id': 'tfidf_svm'
                }
            else:
                probs_dict = {self.class_names[j]: float(probs[j]) for j in range(len(self.class_names))}
                result = {
                    'probs': probs_dict,
                    'abstain': False,
                    'decision': self.class_names[probs.argmax()],
                    'max_prob': float(max_prob),
                    'latency_ms': per_sample_time,
                    'cost_cents': 0.001,
                    'model_id': 'tfidf_svm'
                }
            
            results.append(result)
        
        return results
    
    def save(self, path: str):
        """Save model to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_dict = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'class_names': self.class_names,
            'abstain_threshold': self.abstain_threshold,
            'is_fitted': self.is_fitted,
            'config': {
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'C': self.C,
                'class_weight': self.class_weight,
                'calibration_method': self.calibration_method
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        logger.info(f"Saved TF-IDF SVM model to {save_path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.vectorizer = model_dict['vectorizer']
        self.classifier = model_dict['classifier']
        self.class_names = model_dict['class_names']
        self.abstain_threshold = model_dict['abstain_threshold']
        self.is_fitted = model_dict['is_fitted']
        
        config = model_dict['config']
        self.max_features = config['max_features']
        self.ngram_range = config['ngram_range']
        self.min_df = config['min_df']
        self.C = config['C']
        self.class_weight = config['class_weight']
        self.calibration_method = config['calibration_method']
        
        logger.info(f"Loaded TF-IDF SVM model from {path}")
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features for each class.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            Dictionary mapping class names to top features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients from base estimator
        # Note: CalibratedClassifierCV wraps the base estimator
        if hasattr(self.classifier, 'base_estimator'):
            coef = self.classifier.base_estimator.coef_
        else:
            # Try to get from first calibrated classifier
            coef = self.classifier.calibrated_classifiers_[0].base_estimator.coef_
        
        importance = {}
        
        for i, class_name in enumerate(self.class_names):
            # Get coefficients for this class
            class_coef = coef[i]
            
            # Get top positive features
            top_idx = class_coef.argsort()[-top_k:][::-1]
            top_features = [(feature_names[idx], float(class_coef[idx])) 
                           for idx in top_idx]
            
            importance[class_name] = top_features
        
        return importance