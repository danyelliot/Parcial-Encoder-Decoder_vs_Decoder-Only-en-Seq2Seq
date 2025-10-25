#!/usr/bin/env python3
"""
Tests para el tokenizador.
"""

import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tokenizer import SimpleTokenizer


class TestTokenizer(unittest.TestCase):
    """Tests para SimpleTokenizer."""
    
    def test_vocab_building(self):
        """Test: construir vocabulario desde textos."""
        # Arrange
        tokenizer = SimpleTokenizer()
        texts = ["hello world", "hello python", "world of python"]
        
        # Act
        tokenizer.build_vocab(texts, min_freq=1)
        
        # Assert
        self.assertIn('hello', tokenizer.vocab)
        self.assertIn('world', tokenizer.vocab)
        self.assertIn('python', tokenizer.vocab)
        self.assertIn('<PAD>', tokenizer.vocab)
        self.assertIn('<SOS>', tokenizer.vocab)
        self.assertIn('<EOS>', tokenizer.vocab)
        self.assertIn('<UNK>', tokenizer.vocab)
    
    def test_encode_decode_consistency(self):
        """Test: encode->decode debe ser consistente."""
        # Arrange
        tokenizer = SimpleTokenizer()
        texts = ["hello world test"]
        tokenizer.build_vocab(texts)
        
        original = "hello world"
        
        # Act
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        
        # Assert
        self.assertEqual(original, decoded)
    
    def test_unknown_token_handling(self):
        """Test: tokens desconocidos deben mapearse a <UNK>."""
        # Arrange
        tokenizer = SimpleTokenizer()
        texts = ["hello world"]
        tokenizer.build_vocab(texts)
        
        # Act
        encoded = tokenizer.encode("unknown token")
        
        # Assert
        # Todos los tokens deben ser <UNK>
        unk_id = tokenizer.vocab['<UNK>']
        self.assertTrue(all(id == unk_id for id in encoded))
    
    def test_special_tokens_have_fixed_ids(self):
        """Test: tokens especiales deben tener IDs fijos."""
        # Arrange & Act
        tokenizer = SimpleTokenizer()
        
        # Assert
        self.assertEqual(tokenizer.vocab['<PAD>'], 0)
        self.assertEqual(tokenizer.vocab['<SOS>'], 1)
        self.assertEqual(tokenizer.vocab['<EOS>'], 2)
        self.assertEqual(tokenizer.vocab['<UNK>'], 3)
    
    def test_min_freq_filtering(self):
        """Test: min_freq debe filtrar tokens poco frecuentes."""
        # Arrange
        tokenizer = SimpleTokenizer()
        texts = [
            "common common common",
            "common common",
            "rare"
        ]
        
        # Act
        tokenizer.build_vocab(texts, min_freq=2)
        
        # Assert
        self.assertIn('common', tokenizer.vocab)
        self.assertNotIn('rare', tokenizer.vocab)
    
    def test_save_load_vocab(self):
        """Test: guardar y cargar vocabulario."""
        # Arrange
        tokenizer1 = SimpleTokenizer()
        texts = ["test vocab save load"]
        tokenizer1.build_vocab(texts)
        
        # Act
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            vocab_path = f.name
        
        try:
            tokenizer1.save_vocab(vocab_path)
            
            tokenizer2 = SimpleTokenizer()
            tokenizer2.load_vocab(vocab_path)
            
            # Assert
            self.assertEqual(tokenizer1.vocab, tokenizer2.vocab)
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)
    
    def test_empty_text_handling(self):
        """Test: manejar texto vacío."""
        # Arrange
        tokenizer = SimpleTokenizer()
        texts = ["hello world"]
        tokenizer.build_vocab(texts)
        
        # Act
        encoded = tokenizer.encode("")
        
        # Assert
        self.assertEqual(encoded, [])
    
    def test_deterministic_encoding(self):
        """Test: encoding debe ser determinista."""
        # Arrange
        tokenizer = SimpleTokenizer()
        texts = ["test deterministic encoding"]
        tokenizer.build_vocab(texts)
        
        text = "test encoding"
        
        # Act
        encoded1 = tokenizer.encode(text)
        encoded2 = tokenizer.encode(text)
        
        # Assert
        self.assertEqual(encoded1, encoded2)


if __name__ == '__main__':
    unittest.main()
