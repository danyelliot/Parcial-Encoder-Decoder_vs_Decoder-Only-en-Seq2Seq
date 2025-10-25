#!/usr/bin/env python3
"""
Script para generar gráficos de resultados.
"""

import argparse
import json
import csv
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib no disponible, generando tablas en texto", file=sys.stderr)


def plot_benchmark(bench_file, output_dir):
    """Genera gráficos de benchmark."""
    with open(bench_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if not HAS_MATPLOTLIB:
        # Generar tabla en texto
        with open(f"{output_dir}/bench_table.txt", 'w') as f:
            f.write("Benchmark Results\n")
            f.write("=" * 80 + "\n\n")
            for row in data:
                f.write(f"Model: {row['model']}\n")
                f.write(f"  Latency: {row['latency_mean']} ± {row['latency_std']} ms\n")
                f.write(f"  Memory: {row['memory_mean']} ± {row['memory_std']} MB\n\n")
        return
    
    models = [row['model'] for row in data]
    latencies = [float(row['latency_mean']) for row in data]
    latency_stds = [float(row['latency_std']) for row in data]
    memories = [float(row['memory_mean']) for row in data]
    memory_stds = [float(row['memory_std']) for row in data]
    
    # Gráfico de latencia
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(models))
    ax.bar(x, latencies, yerr=latency_stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Latencia (ms)')
    ax.set_title('Comparación de Latencia')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_latencia.png", dpi=150)
    plt.close()
    
    # Gráfico de memoria
    if any(m > 0 for m in memories):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, memories, yerr=memory_stds, capsize=5, alpha=0.7, color='orange')
        ax.set_xlabel('Modelo')
        ax.set_ylabel('Memoria (MB)')
        ax.set_title('Comparación de Uso de Memoria')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plot_memoria.png", dpi=150)
        plt.close()


def plot_metrics(metrics_ed_file, metrics_do_file, output_dir):
    """Genera gráficos de métricas."""
    with open(metrics_ed_file, 'r') as f:
        metrics_ed = json.load(f)
    
    with open(metrics_do_file, 'r') as f:
        metrics_do = json.load(f)
    
    if not HAS_MATPLOTLIB:
        # Tabla en texto
        with open(f"{output_dir}/metrics_comparison.txt", 'w') as f:
            f.write("Metrics Comparison\n")
            f.write("=" * 80 + "\n\n")
            f.write("Encoder-Decoder:\n")
            f.write(f"  Token Accuracy: {metrics_ed['token_accuracy']:.4f}\n")
            f.write(f"  Exact Match: {metrics_ed['exact_match']:.4f}\n")
            if 'perplexity' in metrics_ed:
                f.write(f"  Perplexity: {metrics_ed['perplexity']:.4f}\n")
            f.write("\nDecoder-Only:\n")
            f.write(f"  Token Accuracy: {metrics_do['token_accuracy']:.4f}\n")
            f.write(f"  Exact Match: {metrics_do['exact_match']:.4f}\n")
        return
    
    # Gráfico de comparación
    metrics = ['Token Accuracy', 'Exact Match']
    ed_values = [metrics_ed['token_accuracy'], metrics_ed['exact_match']]
    do_values = [metrics_do['token_accuracy'], metrics_do['exact_match']]
    
    x = range(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], ed_values, width, label='Encoder-Decoder', alpha=0.7)
    ax.bar([i + width/2 for i in x], do_values, width, label='Decoder-Only', alpha=0.7)
    
    ax.set_xlabel('Métrica')
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Métricas de Evaluación')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_metrics.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generar gráficos de resultados')
    parser.add_argument('bench_file', type=str, help='Archivo CSV de benchmark')
    parser.add_argument('metrics_ed_file', type=str, help='Archivo JSON de métricas ED')
    parser.add_argument('metrics_do_file', type=str, help='Archivo JSON de métricas DO')
    parser.add_argument('--output-dir', type=str, default='out/', help='Directorio de salida')
    
    args = parser.parse_args()
    
    print("Generando gráficos...")
    
    try:
        plot_benchmark(args.bench_file, args.output_dir)
        print(f"✓ Gráficos de benchmark guardados en {args.output_dir}")
    except Exception as e:
        print(f"Error generando gráficos de benchmark: {e}", file=sys.stderr)
    
    try:
        plot_metrics(args.metrics_ed_file, args.metrics_do_file, args.output_dir)
        print(f"✓ Gráficos de métricas guardados en {args.output_dir}")
    except Exception as e:
        print(f"Error generando gráficos de métricas: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
