"""
Training pipeline for the Fraud Detection model.

Key concepts:
1. Weighted loss for class imbalance
2. Learning rate scheduling
3. Early stopping to prevent overfitting
4. Training/validation loop with metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time

import config
from model import FraudDetector, count_parameters
from data_loader import get_data_loaders
from evaluate import compute_metrics


class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    
    WHY NEEDED?
    - Prevents overfitting to training data
    - Saves the best model automatically
    """
    
    def __init__(self, patience=7, min_delta=0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to count as "better"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        average_loss: float
        all_predictions: list for computing metrics
        all_labels: list for computing metrics
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for sequences, labels in progress_bar:
        # Move to device
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)  # (batch, 2)
        
        # Compute loss (weighted cross-entropy)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Get predictions
        probs = torch.softmax(outputs, dim=-1)
        all_preds.extend(probs[:, 1].detach().cpu().numpy())  # Fraud probability
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader), np.array(all_preds), np.array(all_labels)


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Returns:
        average_loss: float
        all_predictions: numpy array of fraud probabilities
        all_labels: numpy array of true labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No gradient computation
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=-1)
            all_preds.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), np.array(all_preds), np.array(all_labels)


def train(
    model,
    train_loader,
    val_loader,
    class_weights,
    device,
    num_epochs=config.NUM_EPOCHS,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
):
    """
    Full training loop.
    
    Returns:
        model: Trained model (with best weights)
        history: Dictionary with training history
    """
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Class weights: {class_weights.tolist()}")
    print(f"{'='*60}\n")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler (reduce LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=7)
    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_roc_auc': [], 'val_roc_auc': [],
        'train_pr_auc': [], 'val_pr_auc': []
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Compute metrics
        train_metrics = compute_metrics(train_labels, train_preds, verbose=False)
        val_metrics = compute_metrics(val_labels, val_preds, verbose=False)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_roc_auc'].append(train_metrics['roc_auc'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['train_pr_auc'].append(train_metrics['pr_auc'])
        history['val_pr_auc'].append(val_metrics['pr_auc'])
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | ROC-AUC: {train_metrics['roc_auc']:.4f} | PR-AUC: {train_metrics['pr_auc']:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
    
    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"Best validation loss: {early_stopping.best_loss:.4f}")
    print(f"{'='*60}")
    
    return model, history


def save_model(model, path="fraud_detector.pth"):
    """Save model weights."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path="fraud_detector.pth", device=config.DEVICE):
    """Load model weights."""
    model = FraudDetector()
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    print("="*60)
    print("TRANSFORMER-BASED FRAUD DETECTION - TRAINING")
    print("="*60)
    
    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Get data loaders
    print("\nLoading data...")
    try:
        train_loader, val_loader, test_loader, class_weights = get_data_loaders()
    except FileNotFoundError:
        print("\nERROR: Dataset not found!")
        print("Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Place 'creditcard.csv' in the 'data/' folder")
        exit(1)
    
    # Create model
    model = FraudDetector()
    print(f"\nModel created with {count_parameters(model):,} parameters")
    
    # Train
    model, history = train(
        model,
        train_loader,
        val_loader,
        class_weights,
        device=config.DEVICE
    )
    
    # Save model
    save_model(model)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, config.DEVICE)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    test_metrics = compute_metrics(test_labels, test_preds, verbose=True)
