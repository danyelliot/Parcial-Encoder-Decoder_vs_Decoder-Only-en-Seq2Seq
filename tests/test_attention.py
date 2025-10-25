#!/usr/bin/env python3
"""
Tests para el módulo de atención.
"""

import unittest
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from attention import (
        ScaledDotProductAttention,
        MultiHeadAttention,
        create_padding_mask,
        create_causal_mask,
        create_combined_mask
    )


@unittest.skipIf(not HAS_TORCH, "PyTorch no disponible")
class TestAttention(unittest.TestCase):
    """Tests para módulos de atención."""
    
    def test_scaled_dot_product_attention_shape(self):
        """Test: verificar dimensiones de salida de atención."""
        # Arrange
        batch_size = 2
        heads = 4
        seq_len = 10
        d_k = 64
        
        attention = ScaledDotProductAttention(dropout=0.0)
        attention.eval()
        
        query = torch.randn(batch_size, heads, seq_len, d_k)
        key = torch.randn(batch_size, heads, seq_len, d_k)
        value = torch.randn(batch_size, heads, seq_len, d_k)
        
        # Act
        output, weights = attention(query, key, value)
        
        # Assert
        self.assertEqual(output.shape, (batch_size, heads, seq_len, d_k))
        self.assertEqual(weights.shape, (batch_size, heads, seq_len, seq_len))
    
    def test_attention_weights_sum_to_one(self):
        """Test: los pesos de atención deben sumar 1."""
        # Arrange
        attention = ScaledDotProductAttention(dropout=0.0)
        attention.eval()
        
        query = torch.randn(1, 1, 5, 64)
        key = torch.randn(1, 1, 5, 64)
        value = torch.randn(1, 1, 5, 64)
        
        # Act
        _, weights = attention(query, key, value)
        
        # Assert
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))
    
    def test_causal_mask_blocks_future(self):
        """Test: la máscara causal debe bloquear posiciones futuras."""
        # Arrange
        seq_len = 5
        device = torch.device('cpu')
        
        # Act
        mask = create_causal_mask(seq_len, device)
        
        # Assert
        # La máscara debe ser triangular inferior
        expected = torch.tril(torch.ones(seq_len, seq_len))
        self.assertTrue(torch.equal(mask.squeeze(), expected))
    
    def test_multi_head_attention_shape(self):
        """Test: MultiHeadAttention debe producir forma correcta."""
        # Arrange
        batch_size = 2
        seq_len = 10
        d_model = 128
        num_heads = 4
        
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        mha.eval()
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Act
        output, _ = mha(x, x, x)
        
        # Assert
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
    
    def test_padding_mask_creation(self):
        """Test: crear máscara de padding correctamente."""
        # Arrange
        seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        pad_idx = 0
        
        # Act
        mask = create_padding_mask(seq, pad_idx)
        
        # Assert
        expected = torch.tensor([
            [[[True, True, True, False, False]]],
            [[[True, True, False, False, False]]]
        ])
        self.assertTrue(torch.equal(mask, expected))
    
    def test_attention_with_mask(self):
        """Test: atención debe respetar la máscara."""
        # Arrange
        attention = ScaledDotProductAttention(dropout=0.0)
        attention.eval()
        
        query = torch.randn(1, 1, 3, 64)
        key = torch.randn(1, 1, 3, 64)
        value = torch.randn(1, 1, 3, 64)
        
        # Máscara que bloquea la última posición
        mask = torch.tensor([[[[True, True, False]]]])
        
        # Act
        _, weights = attention(query, key, value, mask)
        
        # Assert
        # El peso en la posición bloqueada debe ser casi 0
        self.assertTrue(weights[0, 0, 0, 2] < 1e-6)


class TestAttentionWithoutTorch(unittest.TestCase):
    """Tests que no requieren PyTorch."""
    
    def test_import(self):
        """Test: el módulo debe importarse correctamente."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        try:
            import attention
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"No se pudo importar attention: {e}")


if __name__ == '__main__':
    unittest.main()
