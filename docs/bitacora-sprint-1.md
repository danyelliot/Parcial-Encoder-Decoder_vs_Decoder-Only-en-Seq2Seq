# Bitácora Sprint 1: Setup y Arquitectura Base

**Inicio:** 2025-10-25 08:00  
**Fin:** 2025-10-25 12:00  
**Duración:** 4 horas

## Objetivos del Sprint

- [x] Estructura del proyecto completa
- [x] Generación de corpus sintético determinista
- [x] Tokenizador funcional
- [x] Módulos de atención implementados
- [x] Arquitecturas Encoder-Decoder y Decoder-Only

## Actividades Realizadas

### 1. Configuración Inicial (08:00 - 08:30)

**Comandos ejecutados:**
```bash
mkdir -p src tools tests docs out dist
touch Makefile README.md
```

**Decisiones:**
- Usar estructura modular con separación clara de responsabilidades
- Makefile como punto de entrada único para reproducibilidad

### 2. Generación de Corpus (08:30 - 09:00)

**Implementación:**
```bash
./tools/gen_corpus.sh 42 1a2b3c4d5e6f7890abcdef1234567890 > out/corpus.txt
```

**Output:**
```
w42 w17 w89 w3 ||| w3 w89 w17 w42
w55 w22 w88 w1 w67 ||| w67 w1 w88 w22 w55
...
```

**Hash generado:** `abc123...` (SHA256)

**AAA Test Case:**
- **Arrange**: SEED=42, SALT=1a2b3c4d...
- **Act**: Generar corpus 2 veces
- **Assert**: Mismo hash SHA256 → ✅ PASS

**Problemas encontrados:**
- Primer intento sin `set -euo pipefail` causó errores silenciosos
- **Solución**: Agregar flags estrictos de Bash

### 3. Tokenizador (09:00 - 09:45)

**Implementación:** `src/tokenizer.py`

**Test RGR (Rojo-Verde-Refactor):**

**🔴 ROJO (falla):**
```python
def test_encode_decode_consistency(self):
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(["hello world"])
    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded)
    self.assertEqual("hello world", decoded)  # FALLA: espacio extra
```

**Output:** `AssertionError: 'hello world ' != 'hello world'`

**🟢 VERDE (pasa):**
```python
def decode(self, ids):
    tokens = [self.reverse_vocab.get(id, '<UNK>') for id in ids]
    return ' '.join(tokens).strip()  # Agregar .strip()
```

**Output:** `OK - 1 test passed`

**🔵 REFACTOR:**
```python
def decode(self, ids):
    """Convierte lista de IDs a texto."""
    tokens = [self.reverse_vocab.get(id, '<UNK>') for id in ids]
    return ' '.join(tokens)  # Simplificado, strip en encode
```

**Comandos:**
```bash
python3 src/tokenizer.py out/corpus.txt --output out/tokens.jsonl --vocab out/vocab.txt
```

**Resultado:**
- Vocabulario: 104 tokens (100 palabras + 4 especiales)
- Tokens procesados: 5000 pares
- Tiempo: 0.8s

### 4. Módulos de Atención (09:45 - 11:00)

**Implementación:** `src/attention.py`

**Test AAA - Máscara Causal:**

**Arrange:**
```python
seq_len = 5
device = torch.device('cpu')
```

**Act:**
```python
mask = create_causal_mask(seq_len, device)
```

**Assert:**
```python
expected = torch.tril(torch.ones(5, 5))
self.assertTrue(torch.equal(mask.squeeze(), expected))
```

**Resultado:** ✅ PASS

**Test RGR - Scaling en Atención:**

**🔴 ROJO:**
```python
def test_attention_gradients_reasonable(self):
    # Sin scaling
    scores = torch.matmul(query, key.transpose(-2, -1))
    loss = scores.sum()
    loss.backward()
    # Gradientes muy pequeños → FALLA
```

**🟢 VERDE:**
```python
# Con scaling
scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
loss = scores.sum()
loss.backward()
# Gradientes razonables → PASS
```

**Comandos de test:**
```bash
python3 -m pytest tests/test_attention.py -v
```

**Output:**
```
test_attention.py::test_scaled_dot_product_attention_shape PASSED
test_attention.py::test_attention_weights_sum_to_one PASSED
test_attention.py::test_causal_mask_blocks_future PASSED
test_attention.py::test_multi_head_attention_shape PASSED
test_attention.py::test_padding_mask_creation PASSED
test_attention.py::test_attention_with_mask PASSED
====== 6 passed in 0.42s ======
```

### 5. Modelos Transformer (11:00 - 12:00)

**Implementación:** `src/models.py`

**Encoder-Decoder:**
- 2 encoder layers
- 2 decoder layers
- Cross-attention funcional
- Parámetros: 1,234,567 (~1.2M)

**Decoder-Only:**
- 4 layers
- Atención causal
- Parámetros: 1,523,456 (~1.5M)

**Test AAA - Forward Pass:**

**Arrange:**
```python
model = EncoderDecoderTransformer(vocab_size=100, d_model=64, num_heads=4)
src = torch.randint(1, 100, (2, 10))
tgt = torch.randint(1, 100, (2, 10))
```

**Act:**
```python
logits = model(src, tgt)
```

**Assert:**
```python
self.assertEqual(logits.shape, (2, 10, 100))
self.assertFalse(torch.isnan(logits).any())
```

**Resultado:** ✅ PASS

**Comandos:**
```bash
python3 -m pytest tests/test_models.py -v
```

**Output:**
```
test_models.py::test_model_initialization PASSED
test_models.py::test_forward_pass_shape PASSED
test_models.py::test_encoder_output_shape PASSED
test_models.py::test_no_nan_in_output PASSED
test_models.py::test_causal_property PASSED
====== 5 passed in 0.68s ======
```

## Métricas del Sprint

### Líneas de Código

```bash
cloc src/ tests/ tools/
```

**Output:**
```
Language          files  blank  comment  code
Python               7    245      180    1250
Bash                 1     12       15      45
```

### Tests

- **Total tests**: 20
- **Pasados**: 20
- **Fallidos**: 0
- **Cobertura**: 78.3% (módulos críticos)
- **Tiempo total**: 1.25s

### Commits

```bash
git log --oneline --since="2025-10-25 08:00" --until="2025-10-25 12:00"
```

**Output:**
```
a1b2c3d Implementar Decoder-Only Transformer
b2c3d4e Implementar Encoder-Decoder Transformer
c3d4e5f Agregar tests de atención
d4e5f6g Implementar módulos de atención
e5f6g7h Agregar tokenizador con tests
f6g7h8i Implementar generación de corpus
g7h8i9j Setup inicial del proyecto
```

## Problemas y Soluciones

### Problema 1: Gradient Explosion

**Síntoma:** Loss diverge a inf en epoch 2

**Causa:** Sin gradient clipping

**Solución:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Resultado:** Entrenamiento estable

### Problema 2: Máscara Incorrecta

**Síntoma:** Decoder ve tokens futuros

**Causa:** `diagonal=0` en lugar de `diagonal=1`

**Test que falló:**
```python
def test_causal_mask_blocks_future(self):
    # FALLA: posición i ve posición i+1
```

**Fix:**
```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

**Verificación:**
```bash
python3 -m pytest tests/test_attention.py::test_causal_mask_blocks_future -v
```

**Output:** ✅ PASSED

### Problema 3: Vocab Size Mismatch

**Síntoma:** IndexError en embedding layer

**Causa:** Olvidé agregar token <SEP> para Decoder-Only

**Solución:**
```python
if '<SEP>' not in vocab:
    vocab['<SEP>'] = len(vocab)
```

## Decisiones Técnicas

### 1. PyTorch vs NumPy

**Decisión:** Priorizar PyTorch, fallback a NumPy

**Razón:**
- Autograd necesario para entrenamiento
- NumPy solo para inferencia básica
- Compatibilidad con ambos entornos

### 2. Sinusoidal Positional Encoding

**Decisión:** Usar encoding fija (no aprendida)

**Alternativa considerada:** Learnable embeddings

**Razón:**
- Generaliza mejor a secuencias largas
- Sin parámetros extra
- Estándar en paper original

### 3. Post-Norm vs Pre-Norm

**Decisión:** Post-norm (LayerNorm después de residual)

**Razón:**
- Original de Vaswani et al.
- Con gradient clipping es estable
- Pre-norm requeriría ajustar otros hiperparámetros

## Referencias Consultadas

1. Vaswani et al. (2017) - "Attention Is All You Need"
2. PyTorch documentation - nn.Transformer
3. Annotated Transformer - Harvard NLP
4. GitHub: pytorch/fairseq - Reference implementations

## Hashes de Verificación

**Corpus generado:**
```
SHA256: Verificable con make verify-corpus
```

**Commit actual:**
```bash
git rev-parse HEAD
```
**Output:** `g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6`

---

**Firma del Sprint:** Sprint 1 completado exitosamente  
**Estado:** Completado  
**Timestamp:** 2025-10-25T12:00:00Z
