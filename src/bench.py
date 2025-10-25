#!/usr/bin/env python3
"""
Script de benchmarking para modelos seq2seq.
Mide latencia y memoria con múltiples repeticiones.
"""

import argparse
import time
import sys
import tarfile
import pickle
import csv

import numpy as np
import torch
import torch.nn as nn

from models import EncoderDecoderTransformer, DecoderOnlyTransformer


def benchmark_encoder_decoder(model, batch_size, seq_len, vocab_size, device, warmup=1, reps=3):
    """Benchmark de Encoder-Decoder."""
    model.eval()
    model.to(device)
    
    # Datos sintéticos
    src = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(src, tgt)
    
    # Benchmark
    latencies = []
    memory_usage = []
    
    for _ in range(reps):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start = time.time()
        
        with torch.no_grad():
            _ = model(src, tgt)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end = time.time()
        latencies.append((end - start) * 1000)  # ms
        
        if device.type == 'cuda':
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
        else:
            memory_usage.append(0)
    
    return {
        'latency_mean': np.mean(latencies),
        'latency_std': np.std(latencies),
        'memory_mean': np.mean(memory_usage),
        'memory_std': np.std(memory_usage)
    }


def benchmark_decoder_only(model, batch_size, seq_len, vocab_size, device, warmup=1, reps=3):
    """Benchmark de Decoder-Only."""
    model.eval()
    model.to(device)
    
    # Datos sintéticos
    x = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    
    # Benchmark
    latencies = []
    memory_usage = []
    
    for _ in range(reps):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start = time.time()
        
        with torch.no_grad():
            _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end = time.time()
        latencies.append((end - start) * 1000)
        
        if device.type == 'cuda':
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)
        else:
            memory_usage.append(0)
    
    return {
        'latency_mean': np.mean(latencies),
        'latency_std': np.std(latencies),
        'memory_mean': np.mean(memory_usage),
        'memory_std': np.std(memory_usage)
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark de modelos seq2seq')
    parser.add_argument('--model-ed', type=str, required=True, help='Modelo Encoder-Decoder')
    parser.add_argument('--model-do', type=str, required=True, help='Modelo Decoder-Only')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulario')
    parser.add_argument('--n', type=int, default=128, help='Longitud de secuencia')
    parser.add_argument('--batch', type=int, default=8, help='Tamaño de batch')
    parser.add_argument('--warmup', type=int, default=1, help='Iteraciones de warmup')
    parser.add_argument('--reps', type=int, default=3, help='Repeticiones')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--output', type=str, default='out/bench.csv', help='Archivo de salida')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar vocabulario
    vocab = {}
    with open(args.vocab, 'r') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            vocab[token] = int(idx)
    
    vocab_size = len(vocab)
    
    results = []
    
    # Benchmark Encoder-Decoder
    print("\n=== Benchmarking Encoder-Decoder ===")
    
    with tarfile.open(args.model_ed, 'r:gz') as tar:
        tar.extractall('/tmp/bench_ed')
    
    with open('/tmp/bench_ed/config.pkl', 'rb') as f:
        config_ed = pickle.load(f)
    
    # Cargar checkpoint para obtener el estado del modelo
    checkpoint_ed = torch.load('/tmp/bench_ed/model.pt', map_location='cpu')
    state_dict_ed = checkpoint_ed['model_state_dict']
    
    # Inferir parámetros desde state_dict
    vocab_size_ed = state_dict_ed['src_embedding.weight'].shape[0]
    d_model_ed = state_dict_ed['src_embedding.weight'].shape[1]
    max_len_ed = state_dict_ed['pos_encoding.pe'].shape[1]
    
    model_ed = EncoderDecoderTransformer(
        vocab_size=vocab_size_ed,
        d_model=d_model_ed,
        num_heads=config_ed.get('num_heads', 4),
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=d_model_ed * 4,
        max_len=max_len_ed,
        dropout=0.1,
        pad_idx=0
    )
    
    model_ed.load_state_dict(state_dict_ed)
    
    bench_ed = benchmark_encoder_decoder(model_ed, args.batch, args.n, vocab_size_ed, device, args.warmup, args.reps)
    
    print(f"Latencia: {bench_ed['latency_mean']:.2f} ± {bench_ed['latency_std']:.2f} ms")
    print(f"Memoria: {bench_ed['memory_mean']:.2f} ± {bench_ed['memory_std']:.2f} MB")
    
    results.append({
        'model': 'Encoder-Decoder',
        'seq_len': args.n,
        'batch_size': args.batch,
        'latency_mean': bench_ed['latency_mean'],
        'latency_std': bench_ed['latency_std'],
        'memory_mean': bench_ed['memory_mean'],
        'memory_std': bench_ed['memory_std']
    })
    
    # Benchmark Decoder-Only
    print("\n=== Benchmarking Decoder-Only ===")
    
    with tarfile.open(args.model_do, 'r:gz') as tar:
        tar.extractall('/tmp/bench_do')
    
    with open('/tmp/bench_do/config.pkl', 'rb') as f:
        config_do = pickle.load(f)
    
    # Cargar checkpoint para obtener el estado del modelo
    checkpoint_do = torch.load('/tmp/bench_do/model.pt', map_location='cpu')
    state_dict_do = checkpoint_do['model_state_dict']
    
    # Inferir parámetros desde state_dict
    vocab_size_do = state_dict_do['embedding.weight'].shape[0]
    d_model_do = state_dict_do['embedding.weight'].shape[1]
    max_len_do = state_dict_do['pos_encoding.pe'].shape[1]
    
    model_do = DecoderOnlyTransformer(
        vocab_size=vocab_size_do,
        d_model=d_model_do,
        num_heads=config_do.get('num_heads', 4),
        num_layers=4,
        d_ff=d_model_do * 4,
        max_len=max_len_do,
        dropout=0.1,
        pad_idx=0
    )
    
    model_do.load_state_dict(state_dict_do)
    
    bench_do = benchmark_decoder_only(model_do, args.batch, args.n, vocab_size_do, device, args.warmup, args.reps)
    
    print(f"Latencia: {bench_do['latency_mean']:.2f} ± {bench_do['latency_std']:.2f} ms")
    print(f"Memoria: {bench_do['memory_mean']:.2f} ± {bench_do['memory_std']:.2f} MB")
    
    results.append({
        'model': 'Decoder-Only',
        'seq_len': args.n,
        'batch_size': args.batch,
        'latency_mean': bench_do['latency_mean'],
        'latency_std': bench_do['latency_std'],
        'memory_mean': bench_do['memory_mean'],
        'memory_std': bench_do['memory_std']
    })
    
    # Guardar resultados
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Resultados guardados en {args.output}")


if __name__ == '__main__':
    main()
