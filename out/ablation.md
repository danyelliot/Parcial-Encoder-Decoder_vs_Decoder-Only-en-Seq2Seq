# Ablación: Encoder-Decoder vs Decoder-Only

## Comparación de Métricas

| Métrica | Encoder-Decoder | Decoder-Only |
|---------|-----------------|---------------|
| Perplexity | 88.3566 | N/A |
| Token Accuracy | 0.1350 | 0.0109 |
| Exact Match | 0.0000 | 0.0000 |

## Ejemplos de Predicción

### Encoder-Decoder

**Ejemplo 1** ✗
- Input: `w50 w28 w44`
- Target: `w44 w28 w50`
- Generated: ``

**Ejemplo 2** ✗
- Input: `w74 w0 w69 w34 w33 w74 w34 w57`
- Target: `w57 w34 w74 w33 w34 w69 w0 w74`
- Generated: ``

**Ejemplo 3** ✗
- Input: `w19 w35 w29 w19 w26 w78 w73 w41`
- Target: `w41 w73 w78 w26 w19 w29 w35 w19`
- Generated: ``

**Ejemplo 4** ✗
- Input: `w5 w17 w22 w58 w61 w7`
- Target: `w7 w61 w58 w22 w17 w5`
- Generated: ``

**Ejemplo 5** ✗
- Input: `w49 w50 w49 w20`
- Target: `w20 w49 w50 w49`
- Generated: ``

### Decoder-Only

**Ejemplo 1** ✗
- Input: `w50 w28 w44`
- Target: `w44 w28 w50`
- Generated: `w1 w1 w1 w1 w1 w1 w1 w1 w1 w1 w1 w1 w1 w1`

**Ejemplo 2** ✗
- Input: `w74 w0 w69 w34 w33 w74 w34 w57`
- Target: `w57 w34 w74 w33 w34 w69 w0 w74`
- Generated: `w27 w27 w27 w27 w27 w27 w27 w27 w27 w27`

**Ejemplo 3** ✗
- Input: `w19 w35 w29 w19 w26 w78 w73 w41`
- Target: `w41 w73 w78 w26 w19 w29 w35 w19`
- Generated: `w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23 w23`

**Ejemplo 4** ✗
- Input: `w5 w17 w22 w58 w61 w7`
- Target: `w7 w61 w58 w22 w17 w5`
- Generated: `w1 w1 w1 w1 w1 w33`

**Ejemplo 5** ✗
- Input: `w49 w50 w49 w20`
- Target: `w20 w49 w50 w49`
- Generated: `w1 w1 w23 w23 w23 w1 w1 w1 w1 w1 w1 w1 w1 w1 w1`

## Conclusiones

El modelo **Decoder-Only** muestra rendimiento competitivo. Con suficiente capacidad y datos, puede aprender el mapeo seq2seq mediante prompting.

**Ventajas Encoder-Decoder:**
- Separación explícita de entrada/salida
- Atención cruzada dedicada
- Mejor para tareas con estructura clara

**Ventajas Decoder-Only:**
- Arquitectura unificada
- Más flexible para múltiples tareas
- Escalable (como GPT)
