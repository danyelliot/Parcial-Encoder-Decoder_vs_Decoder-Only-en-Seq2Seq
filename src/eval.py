#!/usr/bin/env python3
"""
Script de evaluación para modelos Encoder-Decoder y Decoder-Only.
"""

import argparse
import json
import tarfile
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import EncoderDecoderTransformer, DecoderOnlyTransformer, load_model
from train import Seq2SeqDataset, collate_fn_encoder_decoder, collate_fn_decoder_only


def evaluate_encoder_decoder(model, data_loader, device, vocab_rev, pad_idx=0, sos_idx=1, eos_idx=2):
    """Evalúa el modelo Encoder-Decoder."""
    model.eval()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    total_loss = 0
    total_tokens = 0
    exact_matches = 0
    total_examples = 0
    token_accuracy = 0
    
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # Teacher forcing evaluation
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            
            # Contar tokens no-padding
            mask = (tgt_output != pad_idx)
            num_tokens = mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Token-level accuracy
            pred_tokens = logits.argmax(dim=-1)
            correct = ((pred_tokens == tgt_output) & mask).sum().item()
            token_accuracy += correct
            
            # Greedy decoding para exact match
            for i in range(src.size(0)):
                src_seq = src[i:i+1]
                generated = greedy_decode_ed(model, src_seq, sos_idx, eos_idx, pad_idx, max_len=50, device=device)
                
                # Comparar con target
                tgt_seq = tgt_output[i].cpu().numpy()
                tgt_seq = [t for t in tgt_seq if t not in [pad_idx, sos_idx, eos_idx]]
                gen_seq = [t for t in generated if t not in [pad_idx, sos_idx, eos_idx]]
                
                if tgt_seq == gen_seq:
                    exact_matches += 1
                
                total_examples += 1
                
                # Guardar ejemplos
                if len(predictions) < 10:
                    src_text = ' '.join([vocab_rev.get(t, '<UNK>') for t in src[i].cpu().numpy() if t != pad_idx and t != eos_idx])
                    tgt_text = ' '.join([vocab_rev.get(t, '<UNK>') for t in tgt_seq])
                    gen_text = ' '.join([vocab_rev.get(t, '<UNK>') for t in gen_seq])
                    
                    predictions.append({
                        'src': src_text,
                        'target': tgt_text,
                        'generated': gen_text,
                        'match': tgt_seq == gen_seq
                    })
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss)
    token_acc = token_accuracy / total_tokens if total_tokens > 0 else 0
    exact_match = exact_matches / total_examples if total_examples > 0 else 0
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'token_accuracy': token_acc,
        'exact_match': exact_match,
        'predictions': predictions
    }


def greedy_decode_ed(model, src, sos_idx, eos_idx, pad_idx, max_len, device):
    """Decodificación greedy para Encoder-Decoder."""
    model.eval()
    
    # Encode
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    enc_output = model.encode(src, src_mask)
    
    # Decode
    tgt = torch.LongTensor([[sos_idx]]).to(device)
    
    for _ in range(max_len):
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=device), diagonal=1)
        tgt_mask = (tgt_mask == 0).unsqueeze(0).unsqueeze(0)
        
        dec_output = model.decode(tgt, enc_output, tgt_mask, src_mask)
        logits = model.output_proj(dec_output)
        
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        
        if next_token.item() == eos_idx:
            break
    
    return tgt.squeeze(0).cpu().numpy().tolist()


def evaluate_decoder_only(model, data_loader, device, vocab_rev, pad_idx=0, sep_idx=4):
    """Evalúa el modelo Decoder-Only."""
    model.eval()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    total_loss = 0
    total_tokens = 0
    exact_matches = 0
    total_examples = 0
    token_accuracy = 0
    
    predictions = []
    
    with torch.no_grad():
        for batch_data in data_loader.dataset:
            # Procesar un ejemplo a la vez para mayor control
            src_ids = batch_data['src_ids']
            tgt_ids = batch_data['tgt_ids']
            
            # Input: src + SEP
            input_seq = src_ids + [sep_idx]
            input_tensor = torch.LongTensor([input_seq]).to(device)
            
            # Target completo
            target_seq = tgt_ids
            
            # Generar
            generated = []
            current_input = input_tensor
            
            for _ in range(50):
                logits = model(current_input)
                next_token = logits[:, -1, :].argmax(dim=-1)
                generated.append(next_token.item())
                
                if next_token.item() == pad_idx:  # Asumimos que PAD es fin
                    break
                
                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
            
            # Comparar
            gen_clean = [t for t in generated if t not in [pad_idx, sep_idx]]
            tgt_clean = [t for t in target_seq if t not in [pad_idx, sep_idx]]
            
            if gen_clean == tgt_clean:
                exact_matches += 1
            
            total_examples += 1
            
            # Token accuracy (solo en la parte generada)
            min_len = min(len(gen_clean), len(tgt_clean))
            if min_len > 0:
                correct = sum(1 for g, t in zip(gen_clean[:min_len], tgt_clean[:min_len]) if g == t)
                token_accuracy += correct
                total_tokens += len(tgt_clean)
            
            # Guardar ejemplos
            if len(predictions) < 10:
                src_text = ' '.join([vocab_rev.get(t, '<UNK>') for t in src_ids])
                tgt_text = ' '.join([vocab_rev.get(t, '<UNK>') for t in tgt_clean])
                gen_text = ' '.join([vocab_rev.get(t, '<UNK>') for t in gen_clean])
                
                predictions.append({
                    'src': src_text,
                    'target': tgt_text,
                    'generated': gen_text,
                    'match': gen_clean == tgt_clean
                })
            
            if total_examples >= 500:  # Evaluar solo primeros 500 ejemplos
                break
    
    token_acc = token_accuracy / total_tokens if total_tokens > 0 else 0
    exact_match = exact_matches / total_examples if total_examples > 0 else 0
    
    return {
        'loss': 0.0,  # No calculamos loss aquí
        'perplexity': 0.0,
        'token_accuracy': token_acc,
        'exact_match': exact_match,
        'predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluación de modelos seq2seq')
    parser.add_argument('model_ed', type=str, help='Modelo Encoder-Decoder (.tar.gz)')
    parser.add_argument('model_do', type=str, help='Modelo Decoder-Only (.tar.gz)')
    parser.add_argument('--vocab', type=str, required=True, help='Archivo de vocabulario')
    parser.add_argument('--output-ed', type=str, default='out/metrics_ed.json')
    parser.add_argument('--output-do', type=str, default='out/metrics_do.json')
    parser.add_argument('--output-ablation', type=str, default='out/ablation.md')
    parser.add_argument('--data', type=str, default='out/tokens.jsonl')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar vocabulario
    vocab = {}
    vocab_rev = {}
    with open(args.vocab, 'r') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            idx = int(idx)
            vocab[token] = idx
            vocab_rev[idx] = token
    
    if '<SEP>' not in vocab:
        vocab['<SEP>'] = len(vocab)
        vocab_rev[len(vocab_rev)] = '<SEP>'
    
    vocab_size = len(vocab)
    
    # Cargar datos de test (usamos últimos 10%)
    dataset = Seq2SeqDataset(args.data, max_len=128)
    test_size = int(0.1 * len(dataset))
    _, test_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - test_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Test: {len(test_dataset)} ejemplos")
    
    # Evaluar Encoder-Decoder
    print("\n=== Evaluando Encoder-Decoder ===")
    
    # Extraer y cargar modelo
    with tarfile.open(args.model_ed, 'r:gz') as tar:
        tar.extractall('/tmp/model_ed')
    
    with open('/tmp/model_ed/config.pkl', 'rb') as f:
        config_ed = pickle.load(f)
    
    model_ed = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=config_ed['d_model'],
        num_heads=config_ed['num_heads'],
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=config_ed['d_model'] * 4,
        max_len=128,
        dropout=0.1,
        pad_idx=0
    )
    
    checkpoint = torch.load('/tmp/model_ed/model.pt', map_location=device)
    model_ed.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader_ed = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        collate_fn=collate_fn_encoder_decoder
    )
    
    metrics_ed = evaluate_encoder_decoder(model_ed, test_loader_ed, device, vocab_rev)
    
    print(f"Perplexity: {metrics_ed['perplexity']:.4f}")
    print(f"Token Accuracy: {metrics_ed['token_accuracy']:.4f}")
    print(f"Exact Match: {metrics_ed['exact_match']:.4f}")
    
    # Evaluar Decoder-Only
    print("\n=== Evaluando Decoder-Only ===")
    
    with tarfile.open(args.model_do, 'r:gz') as tar:
        tar.extractall('/tmp/model_do')
    
    with open('/tmp/model_do/config.pkl', 'rb') as f:
        config_do = pickle.load(f)
    
    model_do = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=config_do['d_model'],
        num_heads=config_do['num_heads'],
        num_layers=4,
        d_ff=config_do['d_model'] * 4,
        max_len=256,
        dropout=0.1,
        pad_idx=0
    )
    
    checkpoint = torch.load('/tmp/model_do/model.pt', map_location=device)
    model_do.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader_do = DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    
    metrics_do = evaluate_decoder_only(model_do, test_loader_do, device, vocab_rev, sep_idx=vocab['<SEP>'])
    
    print(f"Token Accuracy: {metrics_do['token_accuracy']:.4f}")
    print(f"Exact Match: {metrics_do['exact_match']:.4f}")
    
    # Guardar métricas
    with open(args.output_ed, 'w') as f:
        json.dump(metrics_ed, f, indent=2)
    
    with open(args.output_do, 'w') as f:
        json.dump(metrics_do, f, indent=2)
    
    # Crear informe de ablación
    with open(args.output_ablation, 'w') as f:
        f.write("# Ablación: Encoder-Decoder vs Decoder-Only\n\n")
        f.write("## Comparación de Métricas\n\n")
        f.write("| Métrica | Encoder-Decoder | Decoder-Only |\n")
        f.write("|---------|-----------------|---------------|\n")
        f.write(f"| Perplexity | {metrics_ed['perplexity']:.4f} | N/A |\n")
        f.write(f"| Token Accuracy | {metrics_ed['token_accuracy']:.4f} | {metrics_do['token_accuracy']:.4f} |\n")
        f.write(f"| Exact Match | {metrics_ed['exact_match']:.4f} | {metrics_do['exact_match']:.4f} |\n")
        f.write("\n## Ejemplos de Predicción\n\n")
        
        f.write("### Encoder-Decoder\n\n")
        for i, pred in enumerate(metrics_ed['predictions'][:5]):
            f.write(f"**Ejemplo {i+1}** {'✓' if pred['match'] else '✗'}\n")
            f.write(f"- Input: `{pred['src']}`\n")
            f.write(f"- Target: `{pred['target']}`\n")
            f.write(f"- Generated: `{pred['generated']}`\n\n")
        
        f.write("### Decoder-Only\n\n")
        for i, pred in enumerate(metrics_do['predictions'][:5]):
            f.write(f"**Ejemplo {i+1}** {'✓' if pred['match'] else '✗'}\n")
            f.write(f"- Input: `{pred['src']}`\n")
            f.write(f"- Target: `{pred['target']}`\n")
            f.write(f"- Generated: `{pred['generated']}`\n\n")
        
        f.write("## Conclusiones\n\n")
        
        if metrics_ed['exact_match'] > metrics_do['exact_match']:
            f.write("El modelo **Encoder-Decoder** obtiene mejor rendimiento en esta tarea de inversión de secuencias. ")
            f.write("Esto es esperado ya que la arquitectura encoder-decoder está específicamente diseñada para tareas seq2seq ")
            f.write("donde hay una clara distinción entre entrada y salida.\n\n")
        else:
            f.write("El modelo **Decoder-Only** muestra rendimiento competitivo. ")
            f.write("Con suficiente capacidad y datos, puede aprender el mapeo seq2seq mediante prompting.\n\n")
        
        f.write("**Ventajas Encoder-Decoder:**\n")
        f.write("- Separación explícita de entrada/salida\n")
        f.write("- Atención cruzada dedicada\n")
        f.write("- Mejor para tareas con estructura clara\n\n")
        
        f.write("**Ventajas Decoder-Only:**\n")
        f.write("- Arquitectura unificada\n")
        f.write("- Más flexible para múltiples tareas\n")
        f.write("- Escalable (como GPT)\n")
    
    print(f"\n✓ Métricas guardadas en {args.output_ed} y {args.output_do}")
    print(f"✓ Ablación guardada en {args.output_ablation}")


if __name__ == '__main__':
    main()
