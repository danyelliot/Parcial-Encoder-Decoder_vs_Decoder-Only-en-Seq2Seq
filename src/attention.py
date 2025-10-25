#!/usr/bin/env python3
"""
Módulos de atención para modelos Transformer.
Implementación usando NumPy o PyTorch según disponibilidad.
"""

import numpy as np
import sys

# Intentar importar PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    USE_TORCH = True
except ImportError:
    USE_TORCH = False
    print("PyTorch no disponible, usando NumPy", file=sys.stderr)


if USE_TORCH:
    class ScaledDotProductAttention(nn.Module):
        """Atención de producto punto escalado."""
        
        def __init__(self, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            """
            Args:
                query: (batch, heads, seq_len, d_k)
                key: (batch, heads, seq_len, d_k)
                value: (batch, heads, seq_len, d_v)
                mask: (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
            Returns:
                output: (batch, heads, seq_len, d_v)
                attention_weights: (batch, heads, seq_len, seq_len)
            """
            d_k = query.size(-1)
            
            # Producto punto: (batch, heads, seq_len, seq_len)
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
            
            # Aplicar máscara si existe
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Softmax
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Aplicar atención a valores
            output = torch.matmul(attention_weights, value)
            
            return output, attention_weights


    class MultiHeadAttention(nn.Module):
        """Atención multi-cabeza."""
        
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Proyecciones lineales
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)
            
            self.attention = ScaledDotProductAttention(dropout)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            """
            Args:
                query: (batch, seq_len, d_model)
                key: (batch, seq_len, d_model)
                value: (batch, seq_len, d_model)
                mask: (batch, 1, seq_len, seq_len)
            """
            batch_size = query.size(0)
            
            # Proyecciones lineales y reshape a múltiples cabezas
            # (batch, seq_len, d_model) -> (batch, seq_len, heads, d_k) -> (batch, heads, seq_len, d_k)
            Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # Atención
            x, attention_weights = self.attention(Q, K, V, mask)
            
            # Concatenar cabezas
            # (batch, heads, seq_len, d_k) -> (batch, seq_len, heads, d_k) -> (batch, seq_len, d_model)
            x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            
            # Proyección final
            x = self.w_o(x)
            x = self.dropout(x)
            
            return x, attention_weights


    class PositionwiseFeedForward(nn.Module):
        """Red feed-forward por posición."""
        
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
            return self.w_2(self.dropout(F.relu(self.w_1(x))))


    class PositionalEncoding(nn.Module):
        """Codificación posicional sinusoidal."""
        
        def __init__(self, d_model, max_len=5000, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            
            # Crear matriz de codificación posicional
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            """
            Args:
                x: (batch, seq_len, d_model)
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    def create_padding_mask(seq, pad_idx):
        """Crea máscara de padding.
        
        Args:
            seq: (batch, seq_len)
            pad_idx: índice del token de padding
        Returns:
            mask: (batch, 1, 1, seq_len)
        """
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


    def create_causal_mask(seq_len, device):
        """Crea máscara causal (look-ahead).
        
        Args:
            seq_len: longitud de la secuencia
            device: dispositivo (cpu/cuda)
        Returns:
            mask: (1, 1, seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return (mask == 0).unsqueeze(0).unsqueeze(0)


    def create_combined_mask(tgt_seq, pad_idx, device):
        """Crea máscara combinada (padding + causal) para decoder.
        
        Args:
            tgt_seq: (batch, seq_len)
            pad_idx: índice del token de padding
            device: dispositivo
        Returns:
            mask: (batch, 1, seq_len, seq_len)
        """
        seq_len = tgt_seq.size(1)
        padding_mask = create_padding_mask(tgt_seq, pad_idx)  # (batch, 1, 1, seq_len)
        causal_mask = create_causal_mask(seq_len, device)  # (1, 1, seq_len, seq_len)
        
        # Combinar máscaras
        combined_mask = padding_mask & causal_mask
        return combined_mask

else:
    # Implementación con NumPy (simplificada para referencia)
    print("Usando implementación NumPy (no recomendado para entrenamiento)", file=sys.stderr)
    
    def softmax(x, axis=-1):
        """Softmax estable numéricamente."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def scaled_dot_product_attention_np(query, key, value, mask=None):
        """Atención con NumPy."""
        d_k = query.shape[-1]
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
