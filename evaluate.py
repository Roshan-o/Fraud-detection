"""
Evaluation metrics for Fraud Detection.

Key metrics explained:
1. ROC-AUC: Area Under ROC Curve
2. PR-AUC: Area Under Precision-Recall Curve (better for imbalanced data)
3. Precision at K% Recall
4. Recall at K% Precision
5. Confusion Matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
import seaborn as sns


def compute_metrics(y_true, y_pred_proba, threshold=0.5, verbose=True):
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities for class 1 (fraud)
        threshold: Classification threshold
        verbose: Whether to print results
    
    Returns:
        Dictionary of metrics
    """
    # Binary predictions using threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # PR-AUC (Average Precision)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Precision, Recall, F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    if verbose:
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print(f"PR-AUC Score:  {pr_auc:.4f}")
        print(f"\nAt threshold = {threshold}:")
        print(f"  Precision: {precision:.4f} (of predicted frauds, {precision*100:.1f}% are real)")
        print(f"  Recall:    {recall:.4f} (of real frauds, caught {recall*100:.1f}%)")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {tp:6d} (Caught frauds)")
        print(f"  False Positives: {fp:6d} (False alarms)")
        print(f"  True Negatives:  {tn:6d} (Correct legit)")
        print(f"  False Negatives: {fn:6d} (Missed frauds)")
    
    return metrics


def precision_at_recall(y_true, y_pred_proba, target_recall):
    """
    Find precision at a specific recall level.
    
    Use case: "If we want to catch X% of frauds, what precision do we get?"
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Find the precision at the target recall
    for p, r in zip(precision_vals, recall_vals):
        if r >= target_recall:
            return p
    return precision_vals[-1]


def recall_at_precision(y_true, y_pred_proba, target_precision):
    """
    Find recall at a specific precision level.
    
    Use case: "If we want 90% of our flags to be real frauds, how much do we catch?"
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Find the maximum recall at or above target precision
    max_recall = 0
    for p, r in zip(precision_vals, recall_vals):
        if p >= target_precision:
            max_recall = max(max_recall, r)
    return max_recall


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find the optimal classification threshold.
    
    Different metrics optimize for different goals:
    - 'f1': Balance precision and recall
    - 'recall': Maximize fraud detection (accept more false positives)
    - 'precision': Minimize false alarms (accept missing some frauds)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    if metric == 'f1':
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_score = f1_scores[best_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return best_threshold, best_score


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random classifier')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curve.
    
    WHY PR CURVE IS IMPORTANT FOR FRAUD DETECTION:
    - ROC can be misleadingly good with imbalanced data
    - PR curve shows the trade-off between precision and recall
    - Area under PR curve (AP) is more informative
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Baseline: fraction of positives (random classifier)
    baseline = y_true.sum() / len(y_true)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    plt.axhline(y=baseline, color='gray', lw=1, linestyle='--', label=f'Random classifier (AP = {baseline:.4f})')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall (How many frauds we catch)')
    plt.ylabel('Precision (How many flags are real frauds)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5, save_path=None):
    """Plot confusion matrix heatmap."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Legit', 'Predicted Fraud'],
                yticklabels=['Actual Legit', 'Actual Fraud'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix (threshold = {threshold})')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[1].plot(history['train_roc_auc'], label='Train')
    axes[1].plot(history['val_roc_auc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title('ROC-AUC Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # PR-AUC
    axes[2].plot(history['train_pr_auc'], label='Train')
    axes[2].plot(history['val_pr_auc'], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('PR-AUC')
    axes[2].set_title('PR-AUC Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def full_evaluation(y_true, y_pred_proba, plot=True, save_dir=None):
    """
    Complete evaluation with all metrics and plots.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE FRAUD DETECTION EVALUATION")
    print("="*60)
    
    # Basic metrics
    metrics = compute_metrics(y_true, y_pred_proba, verbose=True)
    
    # Precision at specific recall levels
    print("\n" + "-"*50)
    print("PRECISION AT DIFFERENT RECALL LEVELS")
    print("-"*50)
    for target_recall in [0.50, 0.70, 0.90, 0.95]:
        p = precision_at_recall(y_true, y_pred_proba, target_recall)
        print(f"  Precision at {target_recall*100:.0f}% Recall: {p*100:.2f}%")
    
    # Recall at specific precision levels
    print("\n" + "-"*50)
    print("RECALL AT DIFFERENT PRECISION LEVELS")
    print("-"*50)
    for target_precision in [0.50, 0.70, 0.90, 0.95]:
        r = recall_at_precision(y_true, y_pred_proba, target_precision)
        print(f"  Recall at {target_precision*100:.0f}% Precision: {r*100:.2f}%")
    
    # Optimal threshold
    best_threshold, best_f1 = find_optimal_threshold(y_true, y_pred_proba)
    print(f"\nOptimal threshold (max F1): {best_threshold:.4f}")
    print(f"F1 at optimal threshold: {best_f1:.4f}")
    
    # Plots
    if plot:
        plot_roc_curve(y_true, y_pred_proba,
                      save_path=f"{save_dir}/roc_curve.png" if save_dir else None)
        plot_precision_recall_curve(y_true, y_pred_proba,
                                   save_path=f"{save_dir}/pr_curve.png" if save_dir else None)
        plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5,
                            save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None)
        plot_confusion_matrix(y_true, y_pred_proba, threshold=best_threshold,
                            save_path=f"{save_dir}/confusion_matrix_optimal.png" if save_dir else None)
    
    return metrics


if __name__ == "__main__":
    # Demo with synthetic data
    print("Demo: Evaluation metrics with synthetic data\n")
    
    np.random.seed(42)
    
    # Simulate imbalanced data (0.5% fraud)
    n_samples = 10000
    n_fraud = 50
    
    y_true = np.zeros(n_samples)
    y_true[:n_fraud] = 1
    np.random.shuffle(y_true)
    
    # Simulate a decent model's predictions
    y_pred_proba = np.random.beta(2, 5, n_samples)  # Most predict low probability
    # Make fraud cases have higher predicted probabilities (model is decent)
    fraud_indices = np.where(y_true == 1)[0]
    y_pred_proba[fraud_indices] = np.random.beta(5, 2, len(fraud_indices))
    
    # Run full evaluation
    full_evaluation(y_true, y_pred_proba, plot=True)
