"""
Configuration file for the Transformer-based Fraud Detection model.
All hyperparameters and settings are defined here.
"""

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DATA_PATH = "data/creditcard.csv"  # Path to Kaggle dataset
SEQUENCE_LENGTH = 20  # Number of transactions per sequence
TEST_SIZE = 0.2  # 20% for testing
VAL_SIZE = 0.1   # 10% of training for validation

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
INPUT_DIM = 30       # Number of features in Kaggle dataset (V1-V28 + Time + Amount)
D_MODEL = 64         # Embedding dimension
N_HEADS = 4          # Number of attention heads (d_model must be divisible by n_heads)
N_LAYERS = 2         # Number of Transformer encoder layers
D_FF = 128           # Feed-forward network hidden dimension
DROPOUT = 0.1        # Dropout rate

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5  # L2 regularization

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
SEED = 42
