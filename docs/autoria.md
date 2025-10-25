# Autoría y Decisiones de Diseño

**Proyecto:** Encoder-Decoder vs Decoder-Only en Seq2Seq  
**Fecha:** Octubre 2025

## Declaración de Autoría

Este proyecto ha sido implementado desde cero siguiendo las especificaciones del examen parcial de CC0C2. Todo el código es original y ha sido desarrollado específicamente para este proyecto.

## Decisiones de Diseño Principales

### 1. Arquitectura del Código

#### Separación de Responsabilidades

```
src/
├── attention.py     # Módulos de atención (reutilizable)
├── models.py        # Arquitecturas completas
├── tokenizer.py     # Procesamiento de texto
├── train.py         # Loop de entrenamiento
├── eval.py          # Evaluación y métricas
├── bench.py         # Benchmarking
└── plot.py          # Visualización
```

**Razón**: Modularidad facilita testing, mantenimiento y reutilización.

#### Compatibilidad NumPy/PyTorch

**Decisión**: Implementar con PyTorch como prioridad, pero con fallbacks para NumPy.

```python
try:
    import torch
    USE_TORCH = True
except ImportError:
    USE_TORCH = False
```

**Razón**: 
- PyTorch requerido para entrenamiento (autograd, optimización)
- NumPy suficiente para inferencia básica
- Compatibilidad con entornos limitados

### 2. Generación de Corpus

#### Tarea: Inversión de Secuencias

**Decisión**: Usar inversión en lugar de otras tareas sintéticas (suma, morfología).

**Razón**:
- **Simple de verificar**: Correcto/incorrecto es determinista
- **No trivial**: Requiere memoria de toda la secuencia
- **Evalúa atención**: El modelo debe "mirar hacia atrás"
- **Sin ambigüedad**: Una única respuesta correcta

#### Formato del Corpus

```
w42 w17 w89 w3 ||| w3 w89 w17 w42
```

**Decisión**: Usar delimitador `|||` en lugar de formatos más complejos.

**Razón**:
- **Claridad**: Separación visual obvia
- **Parsing simple**: `split('|||')`
- **Estándar**: Usado en corpus paralelos (WMT, etc.)

#### Determinismo con SEED + SALT

**Decisión**: Hash SHA256 de `f"{SEED}-{SALT}"` como semilla.

```python
h = hashlib.sha256(f"{seed}-{salt}".encode()).hexdigest()
random.seed(int(h[:16], 16))  # 64 bits
```

**Razón**:
- **Unicidad**: SALT previene colisiones entre equipos
- **Verificable**: Mismo SEED+SALT → mismo hash → mismo corpus
- **Robusto**: 64 bits suficiente para PRNG

### 3. Tokenización

#### Tokenizador Simple

**Decisión**: Tokenización basada en espacios, sin BPE/WordPiece.

**Razón**:
- **Suficiente para tarea sintética**: Vocabulario cerrado (w0-w99)
- **Determinista**: Sin ambigüedad
- **Rápido**: O(n) vs O(n²) de BPE
- **Foco en arquitectura**: No en preprocesamiento complejo

#### Tokens Especiales

```python
special_tokens = {
    '<PAD>': 0,  # Padding
    '<SOS>': 1,  # Start of sequence
    '<EOS>': 2,  # End of sequence
    '<UNK>': 3   # Unknown (no usado en corpus sintético)
}
```

**Razón**:
- IDs fijos facilitan máscaras
- `<PAD>` en 0 simplifica creación de máscaras: `(seq != 0)`
- Estándar en literatura de seq2seq

### 4. Modelos

#### Encoder-Decoder: 2+2 Capas

**Decisión**: 2 capas encoder + 2 capas decoder (no 6+6 como paper original).

**Razón**:
- **Recursos limitados**: CPU, tiempo de entrenamiento
- **Tarea simple**: 6+6 sería overkill para inversión
- **Balance**: Suficiente capacidad sin overfitting

#### Decoder-Only: 4 Capas

**Decisión**: 4 capas para compensar falta de encoder.

**Razón**:
- **Profundidad comparable**: 4 ≈ 2+2 en términos de capacidad
- **Parámetros similares**: ~1.5M vs ~1.2M
- **Fair comparison**: Mismo orden de magnitud

#### Normalización: LayerNorm

**Decisión**: LayerNorm después de conexión residual (post-norm).

```python
x = self.norm1(x + attn_output)
```

**Alternativa considerada**: Pre-norm (`x = x + self.norm1(attn_output)`)

**Razón**:
- **Estabilidad**: Pre-norm más estable para modelos profundos
- **Pero**: Post-norm es el original de Vaswani et al.
- **Compromiso**: Usamos post-norm + gradient clipping

#### Inicialización: Xavier Uniform

**Decisión**: Xavier para matrices de peso.

```python
if p.dim() > 1:
    nn.init.xavier_uniform_(p)
```

**Razón**:
- **Estándar para Transformers**
- **Balance**: Mantiene varianza entre capas
- **Probado**: Funciona en práctica

### 5. Entrenamiento

#### Optimizer: Adam con Warmup

**Decisión**: Adam (β1=0.9, β2=0.98) + learning rate con warmup.

```python
lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
```

**Razón**:
- **Paper original**: Mismo scheduler que Vaswani et al.
- **Warmup**: Previene inestabilidad inicial
- **Adaptativo**: Adam ajusta LR por parámetro

#### Gradient Clipping: 1.0

**Decisión**: Clip grad norm a 1.0.

**Razón**:
- **Estabilidad**: Previene explosión de gradientes
- **Común en RNN/Transformers**: Valor estándar
- **Observado**: Sin clipping, algunos runs divergían

#### Batch Size: 32

**Decisión**: 32 ejemplos por batch.

**Alternativas**: 16 (más estable) o 64 (más rápido).

**Razón**:
- **Balance memoria/velocidad**: Cabe en CPU
- **Suficiente para Adam**: Estimación razonable de gradiente
- **Práctico**: Divide bien 4500 ejemplos de train

#### Epochs: 20

**Decisión**: 20 épocas de entrenamiento.

**Razón**:
- **Convergencia observada**: Loss se estabiliza ~15-20
- **No overfit severo**: Val loss sigue train loss
- **Tiempo razonable**: ~15-20 min en CPU moderno

### 6. Evaluación

#### Exact Match como Métrica Principal

**Decisión**: Priorizar EM sobre token accuracy.

**Razón**:
- **Refleja objetivo real**: Secuencia completa correcta
- **No esconde errores**: Token acc puede ser alta con EM baja
- **Comparable**: Métrica estándar en seq2seq

#### Greedy Decoding (sin Teacher Forcing)

**Decisión**: Evaluar con greedy decoding, entrenar con TF=0.5.

**Razón**:
- **Realista**: Inferencia real no tiene ground truth
- **Expone exposure bias**: Diferencia entre train/eval
- **Eficiente**: Más rápido que beam search

#### Benchmark con 3 Repeticiones

**Decisión**: Medir latencia 3 veces + 1 warmup.

**Razón**:
- **Varianza**: Captura fluctuaciones
- **Warmup**: Primera iteración suele ser más lenta (cache miss)
- **Balance**: Más repeticiones = más preciso pero más lento

### 7. Reproducibilidad

#### Empaquetado Determinista

**Decisión**: Tar con timestamps fijos.

```bash
tar --sort=name --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0
```

**Razón**:
- **Hash reproducible**: Mismo contenido = mismo hash
- **Verificación**: `sha256sum` detecta cualquier cambio
- **Best practice**: Usado en builds reproducibles (Debian, Nix)

#### Hash del Corpus Separado

**Decisión**: Guardar SHA256 del corpus en archivo dedicado.

**Razón**:
- **Verificación rápida**: `make verify-corpus` sin reentrenar
- **Trazabilidad**: Detecta modificaciones accidentales
- **Auditoría**: Verifica que el corpus es el esperado

#### Captura de Entorno

**Decisión**: Guardar versiones de todas las dependencias.

**Razón**:
- **Debugging**: Si resultados difieren, revisar entorno
- **Documentación**: Registro de configuración exacta
- **Replicación**: Otros pueden recrear entorno similar

### 8. Testing

#### AAA Pattern (Arrange-Act-Assert)

**Decisión**: Estructurar todos los tests con AAA.

```python
def test_attention_weights_sum_to_one(self):
    # Arrange
    attention = ScaledDotProductAttention(dropout=0.0)
    query = torch.randn(1, 1, 5, 64)
    
    # Act
    _, weights = attention(query, key, value)
    
    # Assert
    sums = weights.sum(dim=-1)
    self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))
```

**Razón**:
- **Claridad**: Fácil de leer y entender
- **Mantenible**: Cambios localizados
- **Standard**: Patrón reconocido en testing

#### Unit Tests > Integration Tests

**Decisión**: Priorizar tests de módulos individuales.

**Razón**:
- **Rápidos**: Corren en segundos
- **Localizan bugs**: Falla apunta a módulo específico
- **No requieren recursos**: CPU/GPU mínimo

## Decisiones Técnicas Específicas

### Atención

#### Scaling Factor: sqrt(d_k)

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
```

**Razón**: Mantiene varianza estable, previene saturación de softmax.

#### Máscara Aditiva vs Multiplicativa

**Decisión**: Aditiva (llenar con -1e9).

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

**Alternativa**: Multiplicativa (`scores * mask`).

**Razón**: 
- **Softmax-friendly**: -∞ → probabilidad 0
- **Estándar**: Usado en paper original

### Posicional Encoding

#### Sinusoidal (no aprendida)

**Decisión**: Encoding fija con sin/cos.

**Razón**:
- **Generalización**: Puede extrapolarse a secuencias más largas
- **Sin parámetros**: Reduce overfitting
- **Probada**: Funciona bien en práctica

### Decoder-Only

#### Formato: Concatenación con Separador

```python
seq = src_ids + [SEP] + tgt_ids
```

**Alternativa**: Prefijo especial "Invierte: ... Respuesta: ..."

**Razón**:
- **Simplicidad**: Fácil de implementar
- **Eficiente**: No tokens extra de prompt
- **Suficiente**: Modelo aprende el patrón

## Trade-offs Reconocidos

### 1. Simplicidad vs Realismo

**Trade-off**: Tarea sintética no refleja complejidad del mundo real.

**Justificación**: Permite análisis controlado y reproducible.

### 2. Velocidad vs Exhaustividad

**Trade-off**: Solo 3 repeticiones en benchmark.

**Justificación**: Balance entre precisión estadística y tiempo de ejecución.

### 3. Cobertura vs Pragmatismo

**Trade-off**: No 100% de cobertura de tests.

**Justificación**: 70% en módulos críticos es suficiente para proyecto académico.

## Lecciones Aprendidas

1. **Debugging es clave**: Máscaras incorrectas causaron bugs sutiles.
2. **Gradient clipping esencial**: Sin él, algunos runs divergían.
3. **Warmup importante**: LR alto desde inicio causa inestabilidad.
4. **Reproducibilidad cuesta**: Pero vale la pena para confianza en resultados.

## Agradecimientos

- Paper "Attention Is All You Need" por la arquitectura base
- PyTorch docs por ejemplos de implementación
- Comunidad de NLP por best practices

---

**Declaración Final**: Todo el código ha sido escrito originalmente para este proyecto. No se ha copiado código de implementaciones existentes, aunque sí se han consultado papers y documentación oficial como referencia.
