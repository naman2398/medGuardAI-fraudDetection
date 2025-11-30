"""Evaluation metrics for fraud detection models."""

import logging
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    fbeta_score,
    precision_score,
    recall_score
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.0004


def calculate_aucpr(y_true, y_pred_proba):
    """Calculate Area Under Precision-Recall Curve."""
    try:
        aucpr = average_precision_score(y_true, y_pred_proba)
        return aucpr
    except Exception as e:
        logger.error(f"Error calculating AUCPR: {e}")
        return 0.0


def calculate_auroc(y_true, y_pred_proba):
    """Calculate Area Under ROC Curve."""
    try:
        auroc = roc_auc_score(y_true, y_pred_proba)
        return auroc
    except Exception as e:
        logger.error(f"Error calculating AUROC: {e}")
        return 0.0


def calculate_threshold_metrics(y_true, y_pred_proba, threshold=DEFAULT_THRESHOLD):
    """Calculate threshold-based metrics (F2, TPR, TNR, Precision)."""
    try:
        y_pred_binary = (np.array(y_pred_proba) >= threshold).astype(int)
        y_true_arr = np.array(y_true)
        
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_binary, labels=[0, 1]).ravel()
        
        f2 = fbeta_score(y_true_arr, y_pred_binary, beta=2, zero_division=0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return {
            'f2': f2,
            'tpr': tpr,
            'tnr': tnr,
            'precision': precision,
            'threshold': threshold,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
        
    except Exception as e:
        logger.error(f"Error calculating threshold metrics: {e}")
        return {
            'f2': 0.0,
            'tpr': 0.0,
            'tnr': 0.0,
            'precision': 0.0,
            'threshold': threshold,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
        }


def evaluate_fold(y_true, y_pred_proba, fold_idx=None, threshold=DEFAULT_THRESHOLD):
    """Evaluate predictions for a single fold."""
    fold_label = f"Fold {fold_idx + 1}" if fold_idx is not None else "Evaluation"
    
    # Calculate primary metric
    aucpr = calculate_aucpr(y_true, y_pred_proba)
    
    # Calculate supplementary metrics
    auroc = calculate_auroc(y_true, y_pred_proba)
    
    # Calculate threshold-based metrics (F2, TPR, TNR, Precision)
    threshold_metrics = calculate_threshold_metrics(y_true, y_pred_proba, threshold)
    
    # Class distribution
    fraud_count = np.sum(y_true)
    total_count = len(y_true)
    fraud_ratio = fraud_count / total_count if total_count > 0 else 0
    
    metrics = {
        'aucpr': aucpr,
        'auroc': auroc,
        'f2': threshold_metrics['f2'],
        'tpr': threshold_metrics['tpr'],
        'tnr': threshold_metrics['tnr'],
        'precision': threshold_metrics['precision'],
        'threshold': threshold,
        'fraud_count': int(fraud_count),
        'total_count': int(total_count),
        'fraud_ratio': fraud_ratio
    }
    
    return metrics


def evaluate_cv_folds(fold_metrics):
    """Aggregate metrics across all CV folds."""
    aucpr_scores = [m['aucpr'] for m in fold_metrics]
    auroc_scores = [m['auroc'] for m in fold_metrics]
    f2_scores = [m.get('f2', 0) for m in fold_metrics]
    tpr_scores = [m.get('tpr', 0) for m in fold_metrics]
    precision_scores = [m.get('precision', 0) for m in fold_metrics]
    
    cv_results = {
        'aucpr_mean': np.mean(aucpr_scores),
        'aucpr_std': np.std(aucpr_scores),
        'aucpr_scores': aucpr_scores,
        'auroc_mean': np.mean(auroc_scores),
        'auroc_std': np.std(auroc_scores),
        'auroc_scores': auroc_scores,
        'f2_mean': np.mean(f2_scores),
        'f2_std': np.std(f2_scores),
        'f2_scores': f2_scores,
        'tpr_mean': np.mean(tpr_scores),
        'tpr_std': np.std(tpr_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'n_folds': len(fold_metrics)
    }
    
    return cv_results


def log_metrics_to_mlflow(metrics, prefix=""):
    """Log metrics to MLflow if available."""
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
