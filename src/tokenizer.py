#!/usr/bin/env python3
"""
Tokenizador simple para el corpus de inversión de secuencias.
Extrae vocabulario y genera tokens indexados.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


class SimpleTokenizer:
    """Tokenizador simple basado en espacios."""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
        self.next_id = len(self.special_tokens)
        
    def build_vocab(self, texts, min_freq=1):
        """Construye vocabulario a partir de los textos."""
        counter = Counter()
        for text in texts:
            tokens = text.strip().split()
            counter.update(tokens)
        
        # Agregamos tokens especiales
        self.vocab = self.special_tokens.copy()
        
        # Agregamos tokens del corpus
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = self.next_id
                self.next_id += 1
        
        # Vocabulario inverso
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def encode(self, text):
        """Convierte texto a lista de IDs."""
        tokens = text.strip().split()
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def decode(self, ids):
        """Convierte lista de IDs a texto."""
        tokens = [self.reverse_vocab.get(id, '<UNK>') for id in ids]
        return ' '.join(tokens)
    
    def save_vocab(self, path):
        """Guarda vocabulario a archivo."""
        with open(path, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
    
    def load_vocab(self, path):
        """Carga vocabulario desde archivo."""
        self.vocab = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idx = line.strip().split('\t')
                self.vocab[token] = int(idx)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}


def main():
    parser = argparse.ArgumentParser(description='Tokenizador para corpus de inversión')
    parser.add_argument('input', type=str, help='Archivo de corpus de entrada')
    parser.add_argument('--output', type=str, required=True, help='Archivo de salida (JSONL)')
    parser.add_argument('--vocab', type=str, required=True, help='Archivo de vocabulario')
    parser.add_argument('--min-freq', type=int, default=1, help='Frecuencia mínima de token')
    
    args = parser.parse_args()
    
    # Leer corpus
    print(f"Leyendo corpus desde {args.input}...", file=sys.stderr)
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Procesar líneas para extraer todo el vocabulario
    all_texts = []
    for line in lines:
        if '|||' in line:
            src, tgt = line.strip().split('|||')
            all_texts.append(src.strip())
            all_texts.append(tgt.strip())
    
    # Construir vocabulario
    tokenizer = SimpleTokenizer()
    print(f"Construyendo vocabulario (min_freq={args.min_freq})...", file=sys.stderr)
    tokenizer.build_vocab(all_texts, min_freq=args.min_freq)
    print(f"Vocabulario: {len(tokenizer.vocab)} tokens", file=sys.stderr)
    
    # Guardar vocabulario
    tokenizer.save_vocab(args.vocab)
    print(f"Vocabulario guardado en {args.vocab}", file=sys.stderr)
    
    # Tokenizar y guardar
    print(f"Tokenizando y guardando en {args.output}...", file=sys.stderr)
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            if '|||' not in line:
                continue
            
            src, tgt = line.strip().split('|||')
            src = src.strip()
            tgt = tgt.strip()
            
            src_ids = tokenizer.encode(src)
            tgt_ids = tokenizer.encode(tgt)
            
            record = {
                'id': i,
                'src_text': src,
                'tgt_text': tgt,
                'src_ids': src_ids,
                'tgt_ids': tgt_ids
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"✓ Tokenización completada: {len(lines)} ejemplos", file=sys.stderr)
    print(f"✓ Vocabulario: {len(tokenizer.vocab)} tokens", file=sys.stderr)


if __name__ == '__main__':
    main()
