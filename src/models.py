#!/usr/bin/env python3
"""
Modelos Transformer: Encoder-Decoder y Decoder-Only.
"""

import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    USE_TORCH = True
except ImportError:
    USE_TORCH = False
    print("PyTorch no disponible", file=sys.stderr)
    sys.exit(1)

from attention import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    create_padding_mask,
    create_causal_mask,
    create_combined_mask
)


class EncoderLayer(nn.Module):
    """Una capa de encoder."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention con conexión residual
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward con conexión residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class DecoderLayer(nn.Module):
    """Una capa de decoder."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Self-attention con máscara causal
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention con encoder output
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x


class DecoderOnlyLayer(nn.Module):
    """Una capa de decoder-only (como GPT)."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention causal
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class EncoderDecoderTransformer(nn.Module):
    """Transformer Encoder-Decoder completo."""
    
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_encoder_layers=2,
                 num_decoder_layers=2, d_ff=512, max_len=100, dropout=0.1, pad_idx=0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización de pesos."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        """Codifica la secuencia de entrada."""
        x = self.src_embedding(src) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """Decodifica usando el encoder output."""
        x = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        
        return x
    
    def forward(self, src, tgt):
        """
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        # Crear máscaras
        src_mask = create_padding_mask(src, self.pad_idx)
        tgt_mask = create_combined_mask(tgt, self.pad_idx, src.device)
        
        # Encode
        enc_output = self.encode(src, src_mask)
        
        # Decode
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        logits = self.output_proj(dec_output)
        
        return logits


class DecoderOnlyTransformer(nn.Module):
    """Transformer Decoder-Only (como GPT)."""
    
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4,
                 d_ff=512, max_len=200, dropout=0.1, pad_idx=0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización de pesos."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) - secuencia concatenada [src ||| tgt]
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Embedding y posición
        x_emb = self.embedding(x) * np.sqrt(self.d_model)
        x_emb = self.pos_encoding(x_emb)
        
        # Crear máscara causal
        mask = create_combined_mask(x, self.pad_idx, x.device)
        
        # Aplicar capas
        for layer in self.layers:
            x_emb = layer(x_emb, mask)
        
        # Proyección a vocabulario
        logits = self.output_proj(x_emb)
        
        return logits


def save_model(model, optimizer, epoch, loss, path):
    """Guarda el modelo y estado del optimizador."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'type': model.__class__.__name__,
            'd_model': model.d_model,
            'pad_idx': model.pad_idx
        }
    }, path)


def load_model(path, model):
    """Carga el modelo desde un checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
