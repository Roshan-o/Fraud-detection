"""
Data loading and preprocessing for credit card fraud detection.

Key concepts:
1. Load the Kaggle Credit Card Fraud dataset
2. Normalize features (Amount and Time)
3. Create sequences of transactions (simulating user history)
4. Handle class imbalance by computing class weights
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import config


class CreditCardDataset(Dataset):
    """
    PyTorch Dataset for credit card transactions.
    
    Each sample is a SEQUENCE of transactions, and we predict
    whether the LAST transaction in the sequence is fraudulent.
    """
    
    def __init__(self, sequences, labels):
        """
        Args:
            sequences: numpy array of shape (n_samples, seq_len, n_features)
            labels: numpy array of shape (n_samples,) - 0=legit, 1=fraud
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_and_preprocess_data(data_path=config.DATA_PATH):
    """
    Load the Kaggle Credit Card dataset and preprocess it.
    
    The dataset has:
    - V1-V28: PCA-transformed features (already scaled)
    - Time: Seconds elapsed from first transaction
    - Amount: Transaction amount
    - Class: 0=Legitimate, 1=Fraud
    
    Returns:
        df: Preprocessed DataFrame
        scaler: Fitted StandardScaler for Amount and Time
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({100*df['Class'].mean():.3f}%)")
    
    # Normalize Time and Amount (V1-V28 are already normalized from PCA)
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])
    
    return df, scaler


def create_sequences(df, seq_length=config.SEQUENCE_LENGTH):
    """
    Create sequences of transactions for the Transformer.
    
    Since the Kaggle dataset doesn't have user IDs, we create
    sequences using a sliding window approach. This simulates
    having user transaction histories.
    
    Args:
        df: DataFrame with transactions
        seq_length: Number of transactions per sequence
    
    Returns:
        sequences: numpy array (n_samples, seq_length, n_features)
        labels: numpy array (n_samples,) - label of the LAST transaction
    """
    # Feature columns (exclude 'Class' which is our label)
    feature_cols = [col for col in df.columns if col != 'Class']
    
    features = df[feature_cols].values
    labels = df['Class'].values
    
    sequences = []
    seq_labels = []
    
    # Sliding window to create sequences
    for i in range(len(df) - seq_length + 1):
        seq = features[i:i + seq_length]  # seq_length transactions
        label = labels[i + seq_length - 1]  # Label of LAST transaction
        
        sequences.append(seq)
        seq_labels.append(label)
    
    return np.array(sequences), np.array(seq_labels)


def time_based_split(sequences, labels, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE):
    """
    Split data based on TIME (not random) to prevent data leakage.
    
    In real fraud detection:
    - We train on historical data
    - We predict on future transactions
    
    Random splitting would let future information leak into training!
    
    Returns:
        train_seq, train_labels
        val_seq, val_labels  
        test_seq, test_labels
    """
    n_samples = len(labels)
    
    # Time-based split points
    test_start = int(n_samples * (1 - test_size))
    val_start = int(test_start * (1 - val_size))
    
    # Split
    train_seq = sequences[:val_start]
    train_labels = labels[:val_start]
    
    val_seq = sequences[val_start:test_start]
    val_labels = labels[val_start:test_start]
    
    test_seq = sequences[test_start:]
    test_labels = labels[test_start:]
    
    print(f"\nData splits (time-based):")
    print(f"  Train: {len(train_labels)} samples ({100*train_labels.mean():.3f}% fraud)")
    print(f"  Val:   {len(val_labels)} samples ({100*val_labels.mean():.3f}% fraud)")
    print(f"  Test:  {len(test_labels)} samples ({100*test_labels.mean():.3f}% fraud)")
    
    return (train_seq, train_labels), (val_seq, val_labels), (test_seq, test_labels)


def compute_class_weights(labels):
    """
    Compute class weights for handling imbalanced data.
    
    Formula: weight[i] = n_samples / (n_classes * n_samples_in_class[i])
    
    This makes the total weighted contribution from each class equal.
    
    Args:
        labels: numpy array of labels (0 or 1)
    
    Returns:
        weights: torch tensor of shape (2,) for CrossEntropyLoss
    """
    n_samples = len(labels)
    n_classes = 2
    
    n_legit = (labels == 0).sum()
    n_fraud = (labels == 1).sum()
    
    weight_legit = n_samples / (n_classes * n_legit)
    weight_fraud = n_samples / (n_classes * n_fraud)
    
    print(f"\nClass weights:")
    print(f"  Legitimate (class 0): {weight_legit:.4f}")
    print(f"  Fraud (class 1): {weight_fraud:.4f}")
    
    return torch.FloatTensor([weight_legit, weight_fraud])


def get_data_loaders(batch_size=config.BATCH_SIZE):
    """
    Main function to get all data loaders.
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    # Load and preprocess
    df, _ = load_and_preprocess_data()
    
    # Create sequences
    sequences, labels = create_sequences(df)
    print(f"\nCreated {len(labels)} sequences of length {config.SEQUENCE_LENGTH}")
    
    # Time-based split
    (train_seq, train_labels), (val_seq, val_labels), (test_seq, test_labels) = \
        time_based_split(sequences, labels)
    
    # Compute class weights from training data
    class_weights = compute_class_weights(train_labels)
    
    # Create datasets
    train_dataset = CreditCardDataset(train_seq, train_labels)
    val_dataset = CreditCardDataset(val_seq, val_labels)
    test_dataset = CreditCardDataset(test_seq, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Test the data loading
    print("Testing data loading pipeline...\n")
    
    # Check if data file exists
    import os
    if not os.path.exists(config.DATA_PATH):
        print(f"ERROR: Data file not found at {config.DATA_PATH}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"\nAnd place 'creditcard.csv' in the 'data/' folder")
    else:
        train_loader, val_loader, test_loader, class_weights = get_data_loaders()
        
        # Test one batch
        batch_sequences, batch_labels = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Sequences: {batch_sequences.shape}")  # (batch, seq_len, features)
        print(f"  Labels: {batch_labels.shape}")         # (batch,)
