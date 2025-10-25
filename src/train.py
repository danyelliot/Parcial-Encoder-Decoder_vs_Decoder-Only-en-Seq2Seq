#!/usr/bin/env python3
"""
Script de entrenamiento para modelos Encoder-Decoder y Decoder-Only.
"""

import argparse
import json
import sys
import time
import tarfile
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import EncoderDecoderTransformer, DecoderOnlyTransformer, save_model


class Seq2SeqDataset(Dataset):
    """Dataset para tareas seq2seq."""
    
    def __init__(self, data_path, max_len=50):
        self.examples = []
        self.max_len = max_len
        
        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn_encoder_decoder(batch, pad_idx=0, sos_idx=1, eos_idx=2):
    """Collate function para Encoder-Decoder."""
    src_seqs = []
    tgt_seqs = []
    
    for example in batch:
        src = example['src_ids']
        tgt = example['tgt_ids']
        
        src_seqs.append(src + [eos_idx])
        tgt_seqs.append([sos_idx] + tgt + [eos_idx])
    
    # Padding
    max_src_len = max(len(s) for s in src_seqs)
    max_tgt_len = max(len(s) for s in tgt_seqs)
    
    src_padded = []
    tgt_input_padded = []
    tgt_output_padded = []
    
    for src, tgt in zip(src_seqs, tgt_seqs):
        src_padded.append(src + [pad_idx] * (max_src_len - len(src)))
        tgt_input_padded.append(tgt[:-1] + [pad_idx] * (max_tgt_len - len(tgt)))
        tgt_output_padded.append(tgt[1:] + [pad_idx] * (max_tgt_len - len(tgt)))
    
    return {
        'src': torch.LongTensor(src_padded),
        'tgt_input': torch.LongTensor(tgt_input_padded),
        'tgt_output': torch.LongTensor(tgt_output_padded)
    }


def collate_fn_decoder_only(batch, pad_idx=0, sep_idx=4):
    """Collate function para Decoder-Only (formato: src ||| tgt)."""
    sequences = []
    
    for example in batch:
        src = example['src_ids']
        tgt = example['tgt_ids']
        
        # Concatenar: [src, SEP, tgt]
        seq = src + [sep_idx] + tgt
        sequences.append(seq)
    
    # Padding
    max_len = max(len(s) for s in sequences)
    
    input_seqs = []
    target_seqs = []
    
    for seq in sequences:
        padded = seq + [pad_idx] * (max_len - len(seq))
        input_seqs.append(padded[:-1])  # Input: todo menos el último
        target_seqs.append(padded[1:])   # Target: todo desplazado 1
    
    return {
        'input': torch.LongTensor(input_seqs),
        'target': torch.LongTensor(target_seqs)
    }


def train_encoder_decoder(model, train_loader, val_loader, device, args):
    """Entrena el modelo Encoder-Decoder."""
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignora padding
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Learning rate scheduler con warmup
    def lr_lambda(step):
        d_model = args.dim
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (args.warmup_steps ** -1.5))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Entrenando Encoder-Decoder en {device}...")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(src, tgt_input)
            
            # Calculate loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        train_loss /= train_steps
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt_input = batch['tgt_input'].to(device)
                tgt_output = batch['tgt_output'].to(device)
                
                logits = model(src, tgt_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                
                val_loss += loss.item()
                val_steps += 1
        
        val_loss /= val_steps
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, args.output_ed.replace('.tar.gz', '_best.pt'))
    
    # Guardar modelo final
    save_model(model, optimizer, args.epochs, train_losses[-1], args.output_ed.replace('.tar.gz', '_final.pt'))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def train_decoder_only(model, train_loader, val_loader, device, args):
    """Entrena el modelo Decoder-Only."""
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    def lr_lambda(step):
        d_model = args.dim
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (args.warmup_steps ** -1.5))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Entrenando Decoder-Only en {device}...")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_steps = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        train_loss /= train_steps
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                val_loss += loss.item()
                val_steps += 1
        
        val_loss /= val_steps
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, args.output_do.replace('.tar.gz', '_best.pt'))
    
    save_model(model, optimizer, args.epochs, train_losses[-1], args.output_do.replace('.tar.gz', '_final.pt'))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento de modelos seq2seq')
    parser.add_argument('--input', type=str, required=True, help='Archivo de tokens (JSONL)')
    parser.add_argument('--vocab', type=str, required=True, help='Archivo de vocabulario')
    parser.add_argument('--output-ed', type=str, required=True, help='Salida modelo Encoder-Decoder')
    parser.add_argument('--output-do', type=str, required=True, help='Salida modelo Decoder-Only')
    
    parser.add_argument('--dim', type=int, default=128, help='Dimensión del modelo')
    parser.add_argument('--heads', type=int, default=4, help='Número de cabezas de atención')
    parser.add_argument('--context', type=int, default=128, help='Longitud máxima de contexto')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño de batch')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup-steps', type=int, default=4000, help='Warmup steps')
    parser.add_argument('--teacher-forcing', type=float, default=0.5, help='Teacher forcing ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar vocabulario
    vocab = {}
    with open(args.vocab, 'r') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            vocab[token] = int(idx)
    
    vocab_size = len(vocab)
    print(f"Tamaño del vocabulario: {vocab_size}")
    
    # Agregar token separador para decoder-only si no existe
    if '<SEP>' not in vocab:
        vocab['<SEP>'] = len(vocab)
        vocab_size = len(vocab)
    
    # Cargar datos
    dataset = Seq2SeqDataset(args.input, max_len=args.context)
    
    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {len(train_dataset)} ejemplos")
    print(f"Val: {len(val_dataset)} ejemplos")
    
    # DataLoaders para Encoder-Decoder
    train_loader_ed = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_encoder_decoder
    )
    val_loader_ed = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_encoder_decoder
    )
    
    # DataLoaders para Decoder-Only
    train_loader_do = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn_decoder_only(b, sep_idx=vocab['<SEP>'])
    )
    val_loader_do = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn_decoder_only(b, sep_idx=vocab['<SEP>'])
    )
    
    # Crear modelos
    print("\n=== Encoder-Decoder ===")
    model_ed = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.dim,
        num_heads=args.heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=args.dim * 4,
        max_len=args.context,
        dropout=args.dropout,
        pad_idx=0
    )
    
    history_ed = train_encoder_decoder(model_ed, train_loader_ed, val_loader_ed, device, args)
    
    print("\n=== Decoder-Only ===")
    model_do = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=args.dim,
        num_heads=args.heads,
        num_layers=4,
        d_ff=args.dim * 4,
        max_len=args.context * 2,  # Mayor para secuencia concatenada
        dropout=args.dropout,
        pad_idx=0
    )
    
    history_do = train_decoder_only(model_do, train_loader_do, val_loader_do, device, args)
    
    # Empaquetar modelos en tar.gz
    print("\nEmpaquetando modelos...")
    
    for output_path, model_path, history in [
        (args.output_ed, args.output_ed.replace('.tar.gz', '_best.pt'), history_ed),
        (args.output_do, args.output_do.replace('.tar.gz', '_best.pt'), history_do)
    ]:
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(model_path, arcname='model.pt')
            
            # Guardar configuración e historia
            config = {
                'vocab_size': vocab_size,
                'd_model': args.dim,
                'num_heads': args.heads,
                'history': history
            }
            
            config_path = model_path.replace('.pt', '_config.pkl')
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)
            tar.add(config_path, arcname='config.pkl')
    
    print("✓ Entrenamiento completado")


if __name__ == '__main__':
    main()
