"""Logistic regression stacker for arbiter cascade."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import logging
import yaml

from .features import ArbiterFeatureEngineering
from ..voters.classical.calibrate import platt_calibrate, compute_ece

logger = logging.getLogger(__name__)


class LogisticRegressionStacker:
    """Out-of-fold logistic regression stacker."""
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        calibrate: bool = True,
        slice_definitions_path: Optional[str] = "configs/slices.yaml"
    ):
        """
        Initialize stacker.
        
        Args:
            n_folds: Number of CV folds for OOF training
            random_state: Random seed
            calibrate: Whether to calibrate probabilities
            slice_definitions_path: Path to slice definitions
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.calibrate = calibrate
        
        # Load slice definitions
        self.slice_definitions = {}
        if slice_definitions_path and Path(slice_definitions_path).exists():
            with open(slice_definitions_path) as f:
                config = yaml.safe_load(f)
                self.slice_definitions = {s['name']: s for s in config.get('slices', [])}
        
        # Initialize components
        self.feature_engineer = ArbiterFeatureEngineering(self.slice_definitions)
        self.scaler = StandardScaler()
        self.model = None
        self.calibration_model = None
        self.oof_predictions = None
        
    def fit_oof(
        self,
        voter_outputs_list: List[Dict[str, Dict]],
        texts: List[str],
        labels: np.ndarray,
        metadata_list: Optional[List[Dict]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Fit stacker using out-of-fold predictions.
        
        Args:
            voter_outputs_list: List of voter output dictionaries
            texts: List of input texts
            labels: Ground truth labels
            metadata_list: Optional metadata for each sample
            
        Returns:
            Tuple of (OOF predictions, metrics dictionary)
        """
        logger.info(f"Training stacker with {len(texts)} samples using {self.n_folds}-fold CV")
        
        # Create feature matrix
        X = self.feature_engineer.create_feature_matrix(
            voter_outputs_list, texts, metadata_list
        )
        y = labels
        
        # Initialize OOF predictions
        n_samples = len(y)
        n_classes = len(np.unique(y))
        oof_probs = np.zeros((n_samples, n_classes))
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Store fold models for final ensemble
        fold_models = []
        fold_scalers = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold_idx + 1}/{self.n_folds}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced',
                multi_class='multinomial',
                solver='lbfgs'
            )
            model.fit(X_train_scaled, y_train)
            
            # Get OOF predictions
            val_probs = model.predict_proba(X_val_scaled)
            oof_probs[val_idx] = val_probs
            
            # Store fold model
            fold_models.append(model)
            fold_scalers.append(scaler)
            
            # Log fold metrics
            val_preds = val_probs.argmax(axis=1)
            fold_acc = accuracy_score(y_val, val_preds)
            fold_f1 = f1_score(y_val, val_preds, average='weighted')
            logger.info(f"Fold {fold_idx + 1} - Acc: {fold_acc:.4f}, F1: {fold_f1:.4f}")
        
        # Train final model on all data
        logger.info("Training final model on all data")
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced',
            multi_class='multinomial',
            solver='lbfgs'
        )
        self.model.fit(X_scaled, y)
        
        # Store OOF predictions
        self.oof_predictions = oof_probs
        
        # Calibrate if requested
        if self.calibrate:
            logger.info("Calibrating probabilities")
            # Use OOF predictions for calibration
            self.calibration_model, ece, brier = platt_calibrate(
                oof_probs,
                y,
                multiclass=True
            )
            logger.info(f"Calibration - ECE: {ece:.4f}, Brier: {brier:.4f}")
        
        # Compute overall metrics
        oof_preds = oof_probs.argmax(axis=1)
        metrics = {
            'accuracy': accuracy_score(y, oof_preds),
            'f1_weighted': f1_score(y, oof_preds, average='weighted'),
            'f1_macro': f1_score(y, oof_preds, average='macro'),
            'n_samples': n_samples,
            'n_features': X.shape[1]
        }
        
        if self.calibrate:
            metrics['ece_uncalibrated'] = compute_ece(y, oof_probs.max(axis=1))
            cal_probs = self.calibration_model.predict_proba(oof_probs)
            metrics['ece_calibrated'] = compute_ece(y, cal_probs.max(axis=1))
        
        logger.info(f"OOF Metrics: {metrics}")
        
        return oof_probs, metrics
    
    def predict_proba(
        self,
        voter_outputs: Dict[str, Dict],
        text: str,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Predict probabilities for a single sample.
        
        Args:
            voter_outputs: Voter output dictionary
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Probability array
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        features = self.feature_engineer.extract_features(voter_outputs, text, metadata)
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        probs = self.model.predict_proba(features_scaled)[0]
        
        # Calibrate if available
        if self.calibration_model:
            probs = probs.reshape(1, -1)
            probs = self.calibration_model.predict_proba(probs)[0]
        
        return probs
    
    def predict(
        self,
        voter_outputs: Dict[str, Dict],
        text: str,
        metadata: Optional[Dict] = None,
        return_probs: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with full output format.
        
        Args:
            voter_outputs: Voter outputs dictionary
            text: Input text
            metadata: Optional metadata
            return_probs: Whether to return probabilities
            
        Returns:
            Prediction dictionary
        """
        # Get probabilities
        probs = self.predict_proba(voter_outputs, text, metadata)
        
        # Get class names
        class_names = ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']
        
        # Build result
        max_prob = probs.max()
        pred_class = class_names[probs.argmax()]
        
        result = {
            'decision': pred_class,
            'confidence': float(max_prob),
            'model': 'logistic_stacker'
        }
        
        if return_probs:
            result['probs'] = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        
        return result
    
    def apply_tier0_rules(
        self,
        voter_outputs: Dict[str, Dict],
        consensus_threshold: float = 0.9
    ) -> Optional[Dict[str, Any]]:
        """
        Apply Tier-0 fast rules before stacking.
        
        Args:
            voter_outputs: Voter outputs
            consensus_threshold: Threshold for consensus
            
        Returns:
            Fast decision if applicable, None otherwise
        """
        # Check for safety abstention
        for voter_id, output in voter_outputs.items():
            if 'regex' in voter_id and output.get('abstain'):
                reason = output.get('reason', '')
                if 'safety' in reason.lower() or 'corrupted' in reason.lower():
                    return {
                        'decision': None,
                        'abstain': True,
                        'reason': f"Safety rule triggered: {reason}",
                        'tier': 0
                    }
        
        # Check for consensus
        predictions = []
        confidences = []
        
        for output in voter_outputs.values():
            if not output.get('abstain'):
                probs = output.get('probs', {})
                if probs:
                    max_class = max(probs, key=probs.get)
                    max_conf = probs[max_class]
                    predictions.append(max_class)
                    confidences.append(max_conf)
        
        if predictions:
            # All voters agree with high confidence
            if len(set(predictions)) == 1 and min(confidences) >= consensus_threshold:
                return {
                    'decision': predictions[0],
                    'confidence': float(np.mean(confidences)),
                    'tier': 0,
                    'reason': 'Consensus with high confidence'
                }
        
        return None
    
    def save(self, path: str):
        """Save stacker model to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'calibration_model': self.calibration_model,
            'feature_engineer': self.feature_engineer,
            'oof_predictions': self.oof_predictions,
            'n_folds': self.n_folds,
            'calibrate': self.calibrate
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        logger.info(f"Saved stacker model to {save_path}")
    
    def load(self, path: str):
        """Load stacker model from disk."""
        with open(path, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.calibration_model = model_dict['calibration_model']
        self.feature_engineer = model_dict['feature_engineer']
        self.oof_predictions = model_dict['oof_predictions']
        self.n_folds = model_dict['n_folds']
        self.calibrate = model_dict['calibrate']
        
        logger.info(f"Loaded stacker model from {path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        return self.feature_engineer.get_feature_importance(self.model)