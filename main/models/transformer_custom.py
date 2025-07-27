"""
Custom Transformer model for domain sequence embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output

class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DomainTransformer(nn.Module):
    """
    Transformer model for domain sequence embedding.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_length: Maximum sequence length
        dropout: Dropout rate
        use_cls_token: Whether to use [CLS] token for sequence representation
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 max_length: int = 10,
                 dropout: float = 0.1,
                 use_cls_token: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.use_cls_token = use_cls_token
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection (for MLM)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Create padding mask for attention."""
        return (x != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_embeddings: Whether to return embeddings instead of logits
            
        Returns:
            Sequence embeddings or MLM logits
        """
        batch_size, seq_len = x.size()
        
        # Create padding mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(x)
        
        # Embeddings
        embeddings = self.embedding(x) * math.sqrt(self.embed_dim)
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        embeddings = self.dropout(embeddings)
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        if return_embeddings:
            # Return sequence representation
            if self.use_cls_token and seq_len > 0:
                # Use first token ([CLS]) as sequence representation
                return hidden_states[:, 0, :]
            else:
                # Use mean pooling
                mask = attention_mask.squeeze(1).squeeze(1).float()
                masked_hidden = hidden_states * mask.unsqueeze(-1)
                return masked_hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            # Return MLM logits
            return self.output_projection(hidden_states)
    
    def get_embeddings(self, x: torch.Tensor, 
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get embeddings for input sequences."""
        return self.forward(x, attention_mask, return_embeddings=True)
    
    def get_attention_weights(self, x: torch.Tensor, 
                            layer_idx: int = -1) -> torch.Tensor:
        """Get attention weights from a specific layer."""
        batch_size, seq_len = x.size()
        attention_mask = self.create_padding_mask(x)
        
        # Get embeddings
        embeddings = self.embedding(x) * math.sqrt(self.embed_dim)
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        embeddings = self.dropout(embeddings)
        
        # Apply transformer layers up to the specified layer
        hidden_states = embeddings
        for i, layer in enumerate(self.transformer_layers):
            if i == layer_idx:
                # Get attention weights from this layer
                return layer.attention.get_attention_weights(hidden_states, hidden_states, hidden_states, attention_mask)
            hidden_states = layer(hidden_states, attention_mask)
        