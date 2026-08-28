# Transformer-Based Credit Card Fraud Detection

A lightweight **Transformer Encoder** for detecting fraudulent credit card transactions by modeling sequential user transaction patterns using self-attention.

## Overview

Traditional fraud detection treats each transaction independently. This project models **sequences of transactions** using a Transformer, allowing the model to capture temporal spending patterns — such as test transactions before large purchases, velocity attacks, and sudden location changes.

### Key Highlights
- **Transformer Encoder** with multi-head self-attention for sequential transaction modeling
- **Sequence length**: 20 transactions → predict if the last one is fraud
- **Class imbalance handling** via weighted cross-entropy (fraud is ~0.17% of data)
- **Time-based splits** to prevent data leakage (no future info in training)
- **Comprehensive evaluation**: ROC-AUC, PR-AUC, precision@recall trade-offs

## Architecture

```
Input: (batch, 20 transactions, 30 features)
         │
    [Linear Projection: 30 → 64]
         │
    [Positional Encoding]
         │
    ┌─────────────────────────┐
    │   Encoder Layer ×2      │
    │  ├─ Multi-Head Attention│
    │  │   (4 heads)          │
    │  └─ Feed-Forward Network│
    │      (64 → 128 → 64)   │
    └─────────────────────────┘
         │
    [Mean Pooling: (batch, 20, 64) → (batch, 64)]
         │
    [Classification Head: 64 → 2]
         │
Output: [legit_score, fraud_score]
```

## Results

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic Regression | ~0.95 | ~0.60 |
| Random Forest | ~0.96 | ~0.70 |
| **Transformer (Ours)** | **~0.98** | **~0.80** |

## Project Structure

```
├── config.py          # Hyperparameters and configuration
├── model.py           # Transformer-based FraudDetector model
├── data_loader.py     # Data preprocessing & sequence creation
├── train.py           # Training pipeline with early stopping
├── evaluate.py        # Evaluation metrics & visualization
├── baseline.py        # Logistic Regression & Random Forest baselines
├── requirements.txt   # Dependencies
└── data/
    └── creditcard.csv # Kaggle dataset (download separately)
```

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Roshan-o/Fraud-detection.git
cd Fraud-detection
pip install -r requirements.txt
```

### 2. Download Dataset
Download from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` folder.

### 3. Train
```bash
python train.py
```

### 4. Compare with Baselines
```bash
python baseline.py
```

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 64 | Embedding dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 2 | Encoder layers |
| `d_ff` | 128 | Feed-forward hidden dim |
| `seq_length` | 20 | Transactions per sequence |
| Total params | ~50K | Lightweight model |

## Technical Details

### Why Transformer over LSTM/RNN?
- **Parallel processing** — Processes all positions simultaneously, faster training
- **Long-range dependencies** — Direct attention between any pair of transactions
- **Interpretability** — Attention weights reveal which past transactions influenced the prediction

### Handling Class Imbalance
The dataset has only **0.17% fraud** (492 out of 284,807). We handle this via:
- **Weighted CrossEntropy**: `fraud_weight ≈ 500×` to penalize missed frauds
- **PR-AUC** as primary metric (more informative than ROC-AUC for imbalanced data)
- **Time-based splitting** to simulate real-world deployment

### Why Sequential Fraud Detection?
Fraud has temporal patterns:
- **Test transactions** — Small purchase before a large one
- **Velocity attacks** — Many transactions in rapid succession
- **Geographic anomalies** — NYC → Tokyo in 1 hour

## Dataset

The [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset contains:
- **284,807** transactions (September 2013, European cardholders)
- **30 features**: V1–V28 (PCA-transformed), Time, Amount
- **492 frauds** (0.172%)

