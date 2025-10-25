#!/usr/bin/env python3
"""
Tests para los modelos Transformer.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from models import EncoderDecoderTransformer, DecoderOnlyTransformer


@unittest.skipIf(not HAS_TORCH, "PyTorch no disponible")
class TestEncoderDecoder(unittest.TestCase):
    """Tests para Encoder-Decoder."""
    
    def setUp(self):
        """Setup común para tests."""
        self.vocab_size = 100
        self.d_model = 64
        self.num_heads = 4
        self.batch_size = 2
        self.seq_len = 10
    
    def test_model_initialization(self):
        """Test: el modelo debe inicializarse correctamente."""
        # Arrange & Act
        model = EncoderDecoderTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        # Assert
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.d_model, self.d_model)
    
    def test_forward_pass_shape(self):
        """Test: forward pass debe producir forma correcta."""
        # Arrange
        model = EncoderDecoderTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        model.eval()
        
        src = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        tgt = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Act
        with torch.no_grad():
            logits = model(src, tgt)
        
        # Assert
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
    
    def test_encoder_output_shape(self):
        """Test: encoder debe producir forma correcta."""
        # Arrange
        model = EncoderDecoderTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        model.eval()
        
        src = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Act
        with torch.no_grad():
            enc_output = model.encode(src)
        
        # Assert
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(enc_output.shape, expected_shape)
    
    def test_no_nan_in_output(self):
        """Test: el output no debe contener NaN."""
        # Arrange
        model = EncoderDecoderTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        model.eval()
        
        src = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        tgt = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Act
        with torch.no_grad():
            logits = model(src, tgt)
        
        # Assert
        self.assertFalse(torch.isnan(logits).any())


@unittest.skipIf(not HAS_TORCH, "PyTorch no disponible")
class TestDecoderOnly(unittest.TestCase):
    """Tests para Decoder-Only."""
    
    def setUp(self):
        """Setup común."""
        self.vocab_size = 100
        self.d_model = 64
        self.num_heads = 4
        self.batch_size = 2
        self.seq_len = 20
    
    def test_model_initialization(self):
        """Test: modelo debe inicializarse."""
        # Arrange & Act
        model = DecoderOnlyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=4
        )
        
        # Assert
        self.assertIsInstance(model, nn.Module)
    
    def test_forward_pass_shape(self):
        """Test: forward debe producir forma correcta."""
        # Arrange
        model = DecoderOnlyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=4
        )
        model.eval()
        
        x = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Act
        with torch.no_grad():
            logits = model(x)
        
        # Assert
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
    
    def test_causal_property(self):
        """Test: modelo debe ser causal (no ver el futuro)."""
        # Arrange
        model = DecoderOnlyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=2
        )
        model.eval()
        
        # Crear dos secuencias que difieren solo en el último token
        seq1 = torch.randint(1, self.vocab_size, (1, self.seq_len))
        seq2 = seq1.clone()
        seq2[0, -1] = (seq2[0, -1] + 1) % self.vocab_size
        
        # Act
        with torch.no_grad():
            out1 = model(seq1)
            out2 = model(seq2)
        
        # Assert
        # Las predicciones hasta la penúltima posición deben ser iguales
        self.assertTrue(torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5))


class TestModelsWithoutTorch(unittest.TestCase):
    """Tests sin PyTorch."""
    
    def test_import(self):
        """Test: módulo debe importarse."""
        try:
            import models
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Error importando models: {e}")


if __name__ == '__main__':
    unittest.main()
