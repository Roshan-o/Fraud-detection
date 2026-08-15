"""
Baseline models for comparison.

Purpose:
- Compare Transformer against simpler models
- Show that the complexity is justified by performance
- Important for interview: "Why did you use a Transformer?"

Baseline 1: Logistic Regression (simple, interpretable)
Baseline 2: Random Forest (tree-based, handles non-linearity)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import time

import config
from data_loader import load_and_preprocess_data, create_sequences, time_based_split


def flatten_sequences(sequences):
    """
    Flatten sequences for traditional ML models.
    
    (n_samples, seq_len, features) -> (n_samples, seq_len * features)
    
    This loses the sequential structure that the Transformer uses!
    """
    n_samples, seq_len, n_features = sequences.shape
    return sequences.reshape(n_samples, seq_len * n_features)


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression baseline.
    
    Advantages:
    - Fast training
    - Interpretable coefficients
    - Good baseline for linear relationships
    
    Disadvantages:
    - Can't capture sequential patterns
    - Limited to linear decision boundaries
    """
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION BASELINE")
    print("="*50)
    
    start_time = time.time()
    
    # Handle class imbalance
    model = LogisticRegression(
        class_weight='balanced',  # Automatic class weighting
        max_iter=1000,
        random_state=config.SEED
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    
    # Metrics
    train_roc = roc_auc_score(y_train, train_proba)
    val_roc = roc_auc_score(y_val, val_proba)
    train_pr = average_precision_score(y_train, train_proba)
    val_pr = average_precision_score(y_val, val_proba)
    
    print(f"\nTraining - ROC-AUC: {train_roc:.4f} | PR-AUC: {train_pr:.4f}")
    print(f"Validation - ROC-AUC: {val_roc:.4f} | PR-AUC: {val_pr:.4f}")
    
    return model, {'roc_auc': val_roc, 'pr_auc': val_pr}


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest baseline.
    
    Advantages:
    - Handles non-linear relationships
    - Feature importance for interpretability
    - Robust to outliers
    
    Disadvantages:
    - Still can't capture sequential patterns well
    - Slower than Logistic Regression
    """
    print("\n" + "="*50)
    print("RANDOM FOREST BASELINE")
    print("="*50)
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=10,  # Prevent overfitting
        random_state=config.SEED,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    
    # Metrics
    train_roc = roc_auc_score(y_train, train_proba)
    val_roc = roc_auc_score(y_val, val_proba)
    train_pr = average_precision_score(y_train, train_proba)
    val_pr = average_precision_score(y_val, val_proba)
    
    print(f"\nTraining - ROC-AUC: {train_roc:.4f} | PR-AUC: {train_pr:.4f}")
    print(f"Validation - ROC-AUC: {val_roc:.4f} | PR-AUC: {val_pr:.4f}")
    
    return model, {'roc_auc': val_roc, 'pr_auc': val_pr}


def compare_baselines():
    """
    Train and compare all baseline models.
    
    This is important for interviews:
    - Shows scientific rigor
    - Justifies model complexity
    - Demonstrates understanding of trade-offs
    """
    print("="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df, _ = load_and_preprocess_data()
    sequences, labels = create_sequences(df)
    
    # Split
    (train_seq, train_labels), (val_seq, val_labels), (test_seq, test_labels) = \
        time_based_split(sequences, labels)
    
    # Flatten for traditional ML (loses sequential structure!)
    print("\nFlattening sequences for traditional ML models...")
    print("NOTE: This loses the sequential patterns that Transformer captures!")
    
    X_train = flatten_sequences(train_seq)
    X_val = flatten_sequences(val_seq)
    X_test = flatten_sequences(test_seq)
    
    print(f"Flattened shape: {X_train.shape}")  # (n_samples, seq_len * features)
    
    # Train baselines
    results = {}
    
    # Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, train_labels, X_val, val_labels)
    results['Logistic Regression'] = lr_metrics
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, train_labels, X_val, val_labels)
    results['Random Forest'] = rf_metrics
    
    # Summary
    print("\n" + "="*60)
    print("BASELINE COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-"*50)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['roc_auc']:>10.4f} {metrics['pr_auc']:>10.4f}")
    
    print("\n" + "-"*50)
    print("NOTE: These baselines flatten sequences!")
    print("The Transformer should outperform by capturing sequential patterns.")
    
    return results


if __name__ == "__main__":
    try:
        compare_baselines()
    except FileNotFoundError:
        print("\nERROR: Dataset not found!")
        print("Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Place 'creditcard.csv' in the 'data/' folder")
