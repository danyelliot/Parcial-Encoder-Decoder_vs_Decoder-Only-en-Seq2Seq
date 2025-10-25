# Guía Rápida de Uso

Este proyecto implementa la comparación entre arquitecturas Encoder-Decoder y Decoder-Only para tareas de secuencia a secuencia.

## Estructura del Proyecto

```
Parcial-Encoder-Decoder_vs_Decoder-Only-en-Seq2Seq/
├── README.md                 # Documentación principal
├── Makefile                  # Automatización completa
├── .gitattributes            # Normalización EOL/UTF-8
├── .gitignore                # Archivos excluidos de Git
│
├── src/                      # Código fuente (7 módulos)
│   ├── attention.py          # Módulos de atención multi-head
│   ├── models.py             # Encoder-Decoder y Decoder-Only
│   ├── tokenizer.py          # Tokenizador simple
│   ├── train.py              # Loop de entrenamiento
│   ├── eval.py               # Evaluación y métricas
│   ├── bench.py              # Benchmarking latencia/memoria
│   └── plot.py               # Generación de gráficos
│
├── tools/                    # Utilidades
│   └── gen_corpus.sh         # Generador de corpus sintético
│
├── tests/                    # Tests unitarios (20 tests)
│   ├── test_attention.py     # Tests de atención (6 tests)
│   ├── test_models.py        # Tests de modelos (5 tests)
│   └── test_tokenizer.py     # Tests de tokenizador (8 tests)
│
├── docs/                     # Documentación exhaustiva
│   ├── reporte.md            # Reporte final técnico
│   ├── autoria.md            # Decisiones de diseño
│   ├── cobertura.md          # Justificación de cobertura (78%)
│   └── bitacora-sprint-1.md  # Bitácora de desarrollo
│
├── out/                      # Resultados (generados por make)
│   ├── corpus.txt            # Corpus sintético (5000 pares)
│   ├── corpus_sha256.txt     # Hash del corpus
│   ├── vocab.txt             # Vocabulario (104 tokens)
│   ├── tokens.jsonl          # Tokens indexados
│   ├── metrics_ed.json       # Métricas Encoder-Decoder
│   ├── metrics_do.json       # Métricas Decoder-Only
│   ├── ablation.md           # Comparación de modelos
│   ├── bench.csv             # Resultados de benchmark
│   ├── plot_*.png            # Gráficos
│   ├── env.txt               # Entorno capturado
│   └── HASHES.md             # Hash del paquete final
│
└── dist/                     # Artefactos finales
    ├── model_encoder_decoder.tar.gz
    ├── model_decoder_only.tar.gz
    └── proy4-v1.0.0.tar.gz   # Paquete reproducible completo
```

## Uso Rápido

### Pipeline Completo Automático

```bash
# 1. Verificar dependencias
make deps

# 2. Ejecutar pipeline completo
make build && make data && make tokenize && make train && make eval && make bench && make plot

# 3. Empaquetar resultados
make pack

# 4. Verificar reproducibilidad
make verify

# 5. Ver resultados
cat out/ablation.md
```

### Ejecución Paso a Paso

```bash
# Paso 1: Setup
make build

# Paso 2: Generar corpus (5000 ejemplos de inversión)
make data

# Paso 3: Verificar corpus es reproducible
make verify-corpus

# Paso 4: Tokenizar
make tokenize

# Paso 5: Entrenar
make train

# Paso 6: Evaluar
make eval

# Paso 7: Benchmark
make bench

# Paso 8: Gráficos
make plot

# Paso 9: Empaquetar
make pack
```

## Resultados Esperados

### Métricas de Calidad

| Modelo | Exact Match | Token Accuracy | Perplexity |
|--------|-------------|----------------|------------|
| Encoder-Decoder | ~89% | ~96% | ~1.35 |
| Decoder-Only | ~83% | ~94% | N/A |

### Rendimiento

| Modelo | Latencia (ms) | Memoria (MB) | Parámetros |
|--------|---------------|--------------|------------|
| Encoder-Decoder | ~165 | ~45 | 1.2M |
| Decoder-Only | ~132 | ~38 | 1.5M |

## Características Técnicas

### Tarea Sintética
- **Inversión de secuencias**: `[w1, w2, w3]` → `[w3, w2, w1]`
- **Vocabulario**: 100 palabras (w0-w99) + 4 especiales
- **Longitud**: 3-10 tokens por secuencia
- **Tamaño**: 5000 pares (90% train, 10% test)

### Arquitecturas

**Encoder-Decoder:**
- 2 encoder layers + 2 decoder layers
- Cross-attention entre encoder y decoder
- Ideal para tareas con separación clara entrada/salida

**Decoder-Only:**
- 4 layers (comparable a 2+2)
- Formato: concatenación `[input ||| output]`
- Arquitectura unificada como GPT

**Común a ambos:**
- d_model: 128
- Attention heads: 4
- Positional encoding: Sinusoidal
- Dropout: 0.1
- Activation: ReLU

### Tests
- **20 tests unitarios** (pytest)
- **Cobertura**: 78.3% en módulos críticos
- **Tiempo ejecución**: <2 segundos
- **Patrón AAA/RGR**: Todos los tests

## Dependencias

### Requeridas
- Python 3.7+
- NumPy

### Opcionales (pero necesarias para entrenamiento)
- PyTorch 1.8+ (entrenamiento y evaluación)
- matplotlib (gráficos)
- pytest (tests, opcional)

### Instalación

```bash
# Con pip
pip install numpy torch matplotlib pytest pytest-cov

# Con conda
conda install numpy pytorch matplotlib pytest pytest-cov
```

## Verificación de Instalación

```bash
# Verificar Python
python3 --version  # Debe ser 3.7+

# Verificar NumPy
python3 -c "import numpy; print('NumPy:', numpy.__version__)"

# Verificar PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__)"

# Verificar estructura
make build  # Debe crear directorios sin error
```

## Comandos de Verificación

```bash
# Verificar hash del corpus
make verify-corpus

# Verificar hash del paquete final
make verify

# Verificar idempotencia (ejecuta 2 veces y compara)
make test-idem

# Ejecutar tests
make test

# Limpiar temporales
make clean

# Limpiar todo
make distclean
```

## Documentación

Toda la documentación está en el directorio `docs/`:

1. **README.md** - Vista general del proyecto
2. **docs/reporte.md** - Reporte técnico completo
3. **docs/autoria.md** - Decisiones de diseño y justificaciones
4. **docs/cobertura.md** - Justificación de cobertura de tests
5. **docs/bitacora-sprint-1.md** - Bitácora de desarrollo

## Requisitos Cumplidos

- Proyecto 4: Encoder-Decoder vs Decoder-Only
- Ablación: Comparación de arquitecturas
- Corpus sintético: Inversión de secuencias
- Reproducibilidad: SEED + SALT + hash verificado
- Makefile completo con todos los targets
- Tests con cobertura mayor a 70%
- Empaquetado determinista
- Documentación completa

## Licencia

Proyecto académico para CC0C2 - UNI 2025
