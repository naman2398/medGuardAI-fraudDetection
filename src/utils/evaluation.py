"""
Evaluation metrics for fraud detection models.
Primary metric: AUCPR (Area Under Precision-Recall Curve)
"""

import logging
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


def calculate_aucpr(y_true, y_pred_proba):
    """
    Calculate Area Under Precision-Recall Curve.
    This is the primary metric for extreme class imbalance.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        float: AUCPR score
    """
    try:
        aucpr = average_precision_score(y_true, y_pred_proba)
        return aucpr
    except Exception as e:
        logger.error(f"Error calculating AUCPR: {e}")
        return 0.0


def calculate_auroc(y_true, y_pred_proba):
    """
    Calculate Area Under ROC Curve as supplementary metric.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        float: AUROC score
    """
    try:
        auroc = roc_auc_score(y_true, y_pred_proba)
        return auroc
    except Exception as e:
        logger.error(f"Error calculating AUROC: {e}")
        return 0.0


def evaluate_fold(y_true, y_pred_proba, fold_idx=None):
    """
    Evaluate predictions for a single fold with primary and supplementary metrics.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        fold_idx: Optional fold index for logging
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    fold_label = f"Fold {fold_idx + 1}" if fold_idx is not None else "Evaluation"
    
    # Calculate primary metric
    aucpr = calculate_aucpr(y_true, y_pred_proba)
    
    # Calculate supplementary metrics
    auroc = calculate_auroc(y_true, y_pred_proba)
    
    # Class distribution
    fraud_count = y_true.sum()
    total_count = len(y_true)
    fraud_ratio = fraud_count / total_count if total_count > 0 else 0
    
    metrics = {
        'aucpr': aucpr,
        'auroc': auroc,
        'fraud_count': int(fraud_count),
        'total_count': int(total_count),
        'fraud_ratio': fraud_ratio
    }
    
    logger.info(
        f"{fold_label} - AUCPR: {aucpr:.4f}, AUROC: {auroc:.4f}, "
        f"Fraud: {fraud_count}/{total_count} ({fraud_ratio:.4%})"
    )
    
    return metrics


def evaluate_cv_folds(fold_metrics):
    """
    Aggregate metrics across all CV folds.
    
    Args:
        fold_metrics: List of metric dictionaries from each fold
        
    Returns:
        dict: Aggregated metrics with mean and std
    """
    logger.info(f"Aggregating metrics across {len(fold_metrics)} folds")
    
    aucpr_scores = [m['aucpr'] for m in fold_metrics]
    auroc_scores = [m['auroc'] for m in fold_metrics]
    
    cv_results = {
        'aucpr_mean': np.mean(aucpr_scores),
        'aucpr_std': np.std(aucpr_scores),
        'aucpr_scores': aucpr_scores,
        'auroc_mean': np.mean(auroc_scores),
        'auroc_std': np.std(auroc_scores),
        'auroc_scores': auroc_scores,
        'n_folds': len(fold_metrics)
    }
    
    logger.info(
        f"CV Results - AUCPR: {cv_results['aucpr_mean']:.4f} ± {cv_results['aucpr_std']:.4f}, "
        f"AUROC: {cv_results['auroc_mean']:.4f} ± {cv_results['auroc_std']:.4f}"
    )
    
    return cv_results


def log_metrics_to_mlflow(metrics, prefix=""):
    """
    Log metrics to MLflow if available.
    
    Args:
        metrics: Dictionary of metrics to log
        prefix: Optional prefix for metric names
    """
    try:
        import mlflow
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_name = f"{prefix}{key}" if prefix else key
                mlflow.log_metric(metric_name, value)
                
    except ImportError:
        logger.warning("MLflow not available, skipping metric logging")
    except Exception as e:
        logger.warning(f"Failed to log metrics to MLflow: {e}")
