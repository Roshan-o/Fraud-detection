"""
Transformer-based Fraud Detection Model.

This file contains:
1. PositionalEncoding - Injects position information into embeddings
2. MultiHeadAttention - Self-attention with multiple heads
3. FeedForward - Position-wise feed-forward network
4. EncoderLayer - One Transformer encoder layer
5. FraudDetector - Complete model for fraud classification

Each class has detailed comments explaining the concepts.
"""

import torch
import torch.nn as nn
import math
import config


class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings using sinusoidal functions.
    
    WHY NEEDED?
    - Transformers process all positions in parallel (unlike RNNs)
    - Without this, the model can't tell transaction order
    - We use sin/cos because it allows learning relative positions
    
    FORMULA:
    - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_seq_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Compute the division term: 10000^(2i/d_model)
        # Using log for numerical stability: exp(log(10000) * -2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        # Add batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a learnable parameter, but saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x + positional_encoding (same shape)
        """
        # Add positional encoding (broadcasts across batch)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    WHAT IT DOES:
    - Each transaction "attends" to all other transactions
    - Multiple heads allow learning different types of relationships
        - Head 1 might focus on amount patterns
        - Head 2 might focus on timing patterns
        - Head 3 might focus on merchant patterns
    
    FORMULA:
    - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
    - MultiHead = Concat(head_1, ..., head_n) × W_o
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V (learned transformations)
        self.W_q = nn.Linear(d_model, d_model)  # Query: "What am I looking for?"
        self.W_k = nn.Linear(d_model, d_model)  # Key: "What do I contain?"
        self.W_v = nn.Linear(d_model, d_model)  # Value: "What info do I provide?"
        
        # Output projection (combines all heads)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask for padding (batch_size, seq_len)
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Linear projections to get Q, K, V
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Step 2: Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Step 3: Compute attention scores
        # (batch, n_heads, seq, d_k) × (batch, n_heads, d_k, seq) = (batch, n_heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Step 4: Apply mask if provided (for padding tokens)
        if mask is not None:
            # Reshape mask: (batch, seq) -> (batch, 1, 1, seq)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 5: Softmax to get attention weights
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Save for visualization
        self.attention_weights = attention.detach()
        
        # Step 6: Apply attention to values
        # (batch, n_heads, seq, seq) × (batch, n_heads, seq, d_k) = (batch, n_heads, seq, d_k)
        out = torch.matmul(attention, V)
        
        # Step 7: Concatenate heads
        # (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Step 8: Final linear projection
        return self.W_o(out)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    WHAT IT DOES:
    - Applies the same MLP to each position independently
    - Adds non-linearity (the attention mechanism itself is mostly linear)
    
    ARCHITECTURE:
    - Linear(d_model -> d_ff)  # Expand
    - ReLU                      # Non-linearity
    - Dropout
    - Linear(d_ff -> d_model)  # Compress back
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)    # Expand: 64 -> 128
        self.linear2 = nn.Linear(d_ff, d_model)    # Compress: 128 -> 64
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Smoother than ReLU, used in modern transformers
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor (same shape)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    One Transformer Encoder Layer.
    
    STRUCTURE:
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. Feed-Forward Network + Residual + LayerNorm
    
    WHY RESIDUALS?
    - Helps gradient flow in deep networks
    - Allows skipping layers that aren't helpful
    
    WHY LAYER NORM?
    - Stabilizes training
    - Normalizes across features (not batch like BatchNorm)
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Two separate LayerNorms (each has its own learned parameters)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional padding mask
        
        Returns:
            Output tensor (same shape, but enriched with attention information)
        """
        # Sub-layer 1: Multi-Head Attention
        # Pre-norm architecture: normalize before the sublayer
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)  # Residual connection
        
        # Sub-layer 2: Feed-Forward Network
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)  # Residual connection
        
        return x


class FraudDetector(nn.Module):
    """
    Complete Transformer-based Fraud Detection Model.
    
    ARCHITECTURE:
    1. Input Projection: (batch, seq, input_dim) -> (batch, seq, d_model)
    2. Add Positional Encoding
    3. Stack of Encoder Layers
    4. Aggregate sequence (mean pooling)
    5. Classification head: (batch, d_model) -> (batch, 2)
    
    OUTPUT:
    - Logits for 2 classes: [legitimate_score, fraud_score]
    - Use CrossEntropyLoss for training (it applies softmax internally)
    """
    
    def __init__(
        self,
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        max_seq_len=config.SEQUENCE_LENGTH
    ):
        super().__init__()
        
        # Store config
        self.d_model = d_model
        
        # Input projection: raw features -> d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 64 -> 32
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 32 -> 2 classes
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Transaction sequences (batch_size, seq_len, input_dim)
            mask: Optional padding mask (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, 2) - scores for [legitimate, fraud]
        """
        # Step 1: Project input features to d_model dimensions
        # (batch, seq, 30) -> (batch, seq, 64)
        x = self.input_projection(x)
        
        # Step 2: Add positional encoding
        x = self.positional_encoding(x)
        
        # Step 3: Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Step 4: Final normalization
        x = self.final_norm(x)
        
        # Step 5: Aggregate sequence
        # Option 1: Mean pooling (average all positions)
        x = x.mean(dim=1)  # (batch, seq, d_model) -> (batch, d_model)
        
        # Option 2 (alternative): Use last position only
        # x = x[:, -1, :]  # (batch, d_model)
        
        # Step 6: Classification
        logits = self.classifier(x)  # (batch, 2)
        
        return logits
    
    def get_attention_weights(self):
        """Returns attention weights from all encoder layers for visualization."""
        return [layer.attention.attention_weights for layer in self.encoder_layers]


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing FraudDetector model...\n")
    
    # Create model
    model = FraudDetector()
    print(f"Model architecture:\n{model}\n")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Test with dummy input
    batch_size = 4
    seq_len = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM
    
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (4, 2)
    
    # Apply softmax to get probabilities
    probs = torch.softmax(output, dim=-1)
    print(f"Output probabilities:\n{probs}")
