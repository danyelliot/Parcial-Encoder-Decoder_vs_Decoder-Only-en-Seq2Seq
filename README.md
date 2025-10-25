# Proyecto 4: Encoder-Decoder vs Decoder-Only en Seq2Seq

## Descripción

Implementación y comparación experimental de dos arquitecturas Transformer para tareas de secuencia a secuencia: Encoder-Decoder tradicional y Decoder-Only unificado (tipo GPT). La evaluación se realiza sobre una tarea sintética de inversión de secuencias de palabras.

## Estructura del Proyecto

```
├── src/                    # Código fuente
│   ├── attention.py        # Módulos de atención
│   ├── models.py           # Arquitecturas Transformer
│   ├── tokenizer.py        # Tokenizador
│   ├── train.py            # Entrenamiento
│   ├── eval.py             # Evaluación
│   ├── bench.py            # Benchmarking
│   └── plot.py             # Visualizaciones
├── tools/                  # Utilidades
│   └── gen_corpus.sh       # Generador de corpus
├── tests/                  # Pruebas unitarias
├── docs/                   # Documentación
├── out/                    # Resultados
├── dist/                   # Modelos entrenados
└── Makefile                # Automatización
```

## Requisitos

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib (opcional, para gráficos)

## Instalación y Ejecución

```bash
# Pipeline completo
make deps && make build && make data && make train && make eval && make bench

# Empaquetar resultados
make pack
```

## Arquitecturas Implementadas

### Encoder-Decoder
- Encoder: 2 capas con self-attention
- Decoder: 2 capas con self-attention y cross-attention
- Multi-head attention (4 cabezas)
- Positional encoding sinusoidal

### Decoder-Only
- 4 capas decoder con self-attention causal
- Formato: concatenación `[entrada ||| salida]`
- Arquitectura unificada tipo GPT

## Tarea Sintética

Corpus de 5000 pares de secuencias para inversión:
```
Entrada:  w42 w17 w89 w3
Salida:   w3 w89 w17 w42
```

- Longitud: 3-10 tokens
- Vocabulario: 100 palabras (w0-w99)
- Generación determinista vía SHA256

## Métricas

1. **Exact Match**: Secuencias completamente correctas
2. **Token Accuracy**: Tokens individuales correctos
3. **Perplexity**: Calidad de predicción
4. **Latencia**: Tiempo de inferencia (ms)
5. **Uso de Memoria**: Memoria GPU/CPU (MB)

## Reproducibilidad

El corpus y modelos son completamente reproducibles mediante:
- Semilla fija: 42
- Salt hexadecimal: `1a2b3c4d5e6f7890abcdef1234567890`
- Verificación SHA256

```bash
make verify-corpus  # Verificar integridad del corpus
make verify         # Verificar paquete completo
```

## Tests

```bash
make test
```

Suite de pruebas unitarias con cobertura objetivo del 70% en módulos principales.

## Documentación

- `docs/reporte.md`: Reporte técnico completo
- `docs/autoria.md`: Decisiones de diseño e implementación
- `docs/cobertura.md`: Justificación de pruebas
- `docs/bitacora-sprint-1.md`: Desarrollo del proyecto