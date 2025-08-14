"""Calibration utilities for classical models."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from typing import Tuple, Optional, Union, Dict, Any
import warnings


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def platt_calibrate(
    probs: np.ndarray, 
    y_dev: np.ndarray,
    multiclass: bool = False
) -> Tuple[Any, float, float]:
    """
    Platt scaling calibration using logistic regression.
    
    Args:
        probs: Uncalibrated probabilities (n_samples, n_classes) or (n_samples,)
        y_dev: True labels for calibration
        multiclass: Whether to handle multiclass calibration
        
    Returns:
        Tuple of (calibration_model, ece, brier_score)
    """
    if len(probs.shape) == 1:
        probs = probs.reshape(-1, 1)
        binary = True
    else:
        binary = probs.shape[1] == 2
    
    if binary and not multiclass:
        # Binary calibration
        if probs.shape[1] == 2:
            logits = np.log(probs[:, 1] / (probs[:, 0] + 1e-10))
        else:
            logits = np.log(probs[:, 0] / (1 - probs[:, 0] + 1e-10))
        
        logits = logits.reshape(-1, 1)
        cal_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        cal_model.fit(logits, y_dev)
        
        cal_probs = cal_model.predict_proba(logits)[:, 1]
        ece = compute_ece(y_dev, cal_probs)
        brier = brier_score_loss(y_dev, cal_probs)
        
    else:
        # Multiclass calibration
        cal_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000
        )
        cal_model.fit(probs, y_dev)
        
        cal_probs = cal_model.predict_proba(probs)
        
        # Compute metrics for multiclass
        ece = 0
        brier = 0
        n_classes = cal_probs.shape[1]
        
        for i in range(n_classes):
            y_binary = (y_dev == i).astype(int)
            ece += compute_ece(y_binary, cal_probs[:, i])
            brier += brier_score_loss(y_binary, cal_probs[:, i])
        
        ece /= n_classes
        brier /= n_classes
    
    return cal_model, ece, brier


def isotonic_calibrate(
    probs: np.ndarray,
    y_dev: np.ndarray,
    multiclass: bool = False
) -> Tuple[Any, float, float]:
    """
    Isotonic regression calibration.
    
    Args:
        probs: Uncalibrated probabilities
        y_dev: True labels for calibration
        multiclass: Whether to handle multiclass calibration
        
    Returns:
        Tuple of (calibration_model, ece, brier_score)
    """
    if len(probs.shape) == 1:
        probs = probs.reshape(-1, 1)
        binary = True
    else:
        binary = probs.shape[1] == 2
    
    if binary and not multiclass:
        # Binary calibration
        if probs.shape[1] == 2:
            uncal_probs = probs[:, 1]
        else:
            uncal_probs = probs[:, 0]
        
        cal_model = IsotonicRegression(out_of_bounds='clip')
        cal_model.fit(uncal_probs, y_dev)
        
        cal_probs = cal_model.transform(uncal_probs)
        ece = compute_ece(y_dev, cal_probs)
        brier = brier_score_loss(y_dev, cal_probs)
        
    else:
        # Multiclass: calibrate each class vs rest
        n_classes = probs.shape[1] if len(probs.shape) > 1 else len(np.unique(y_dev))
        cal_models = []
        cal_probs = np.zeros_like(probs)
        
        for i in range(n_classes):
            y_binary = (y_dev == i).astype(int)
            iso_model = IsotonicRegression(out_of_bounds='clip')
            
            if len(probs.shape) > 1:
                iso_model.fit(probs[:, i], y_binary)
                cal_probs[:, i] = iso_model.transform(probs[:, i])
            else:
                iso_model.fit(probs[:, 0], y_binary)
                cal_probs[:, i] = iso_model.transform(probs[:, 0])
            
            cal_models.append(iso_model)
        
        # Normalize to sum to 1
        cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
        
        # Compute metrics
        ece = 0
        brier = 0
        for i in range(n_classes):
            y_binary = (y_dev == i).astype(int)
            ece += compute_ece(y_binary, cal_probs[:, i])
            brier += brier_score_loss(y_binary, cal_probs[:, i])
        
        ece /= n_classes
        brier /= n_classes
        cal_model = cal_models  # Return list for multiclass
    
    return cal_model, ece, brier


def temperature_scale(
    logits: np.ndarray,
    y_dev: np.ndarray,
    init_temp: float = 1.0,
    max_iter: int = 50
) -> Tuple[float, float, float]:
    """
    Temperature scaling for neural network calibration.
    
    Args:
        logits: Raw logits from model (before softmax)
        y_dev: True labels for calibration
        init_temp: Initial temperature value
        max_iter: Maximum optimization iterations
        
    Returns:
        Tuple of (optimal_temperature, ece, brier_score)
    """
    from scipy.optimize import minimize
    
    def softmax(x, t=1.0):
        x = x / t
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    def nll_loss(t, logits, labels):
        probs = softmax(logits, t)
        # Handle both binary and multiclass
        if len(probs.shape) == 1:
            probs = np.stack([1 - probs, probs], axis=1)
        
        n_samples = len(labels)
        ce = -np.log(probs[np.arange(n_samples), labels] + 1e-10)
        return ce.mean()
    
    # Optimize temperature
    result = minimize(
        lambda t: nll_loss(t[0], logits, y_dev),
        init_temp,
        method='L-BFGS-B',
        bounds=[(0.01, 10.0)],
        options={'maxiter': max_iter}
    )
    
    optimal_temp = result.x[0]
    
    # Compute calibrated probabilities
    cal_probs = softmax(logits, optimal_temp)
    
    if len(cal_probs.shape) == 1:
        cal_probs = np.stack([1 - cal_probs, cal_probs], axis=1)
    
    # Compute metrics
    if cal_probs.shape[1] == 2:
        # Binary
        ece = compute_ece(y_dev, cal_probs[:, 1])
        brier = brier_score_loss(y_dev, cal_probs[:, 1])
    else:
        # Multiclass
        ece = 0
        brier = 0
        n_classes = cal_probs.shape[1]
        
        for i in range(n_classes):
            y_binary = (y_dev == i).astype(int)
            ece += compute_ece(y_binary, cal_probs[:, i])
            brier += brier_score_loss(y_binary, cal_probs[:, i])
        
        ece /= n_classes
        brier /= n_classes
    
    return optimal_temp, ece, brier


def calibrate_model(
    model: Any,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    method: str = 'platt',
    cv: int = 3
) -> Tuple[Any, Dict[str, float]]:
    """
    High-level calibration wrapper using sklearn's CalibratedClassifierCV.
    
    Args:
        model: Trained sklearn model
        X_dev: Development features
        y_dev: Development labels
        method: 'sigmoid' (Platt) or 'isotonic'
        cv: Cross-validation folds
        
    Returns:
        Tuple of (calibrated_model, metrics_dict)
    """
    if method == 'platt':
        method = 'sigmoid'
    
    cal_model = CalibratedClassifierCV(
        model,
        method=method,
        cv=cv
    )
    
    cal_model.fit(X_dev, y_dev)
    
    # Compute metrics
    probs = cal_model.predict_proba(X_dev)
    
    if probs.shape[1] == 2:
        # Binary
        ece = compute_ece(y_dev, probs[:, 1])
        brier = brier_score_loss(y_dev, probs[:, 1])
    else:
        # Multiclass
        ece = 0
        brier = 0
        n_classes = probs.shape[1]
        
        for i in range(n_classes):
            y_binary = (y_dev == i).astype(int)
            ece += compute_ece(y_binary, probs[:, i])
            brier += brier_score_loss(y_binary, probs[:, i])
        
        ece /= n_classes
        brier /= n_classes
    
    metrics = {
        'ece': ece,
        'brier': brier,
        'method': method,
        'cv_folds': cv
    }
    
    return cal_model, metrics