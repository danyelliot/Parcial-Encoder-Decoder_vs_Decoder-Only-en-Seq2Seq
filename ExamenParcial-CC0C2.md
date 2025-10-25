## **Examen Parcial de CC0C2**

**Entrega:** **Sábado 25 de octubre de 2025** (13:59, hora local)  
**Formato:** Repositorio con **Makefile** y **scripts propios**. Corpus **sintético/local** generado por ustedes.  
**Dependencias permitidas:** Python stdlib, `numpy`, y `torch` **solo si ya están preinstaladas** en el entorno. 

**Alcance reducido (obligatorio):**  
* **Proyecto 3** (Mini-Transformer) **obligatorio**.  
* **Un** proyecto complementario a elegir entre **1, 2, 5 o 6**.  
* **Una** ablación mínima (por ejemplo, **RoPE vs sinusoidal** o **top-p vs beam**).  


### Normas de control, reproducibilidad y trazabilidad


* **Empaquetado determinista:** `make pack` -> `dist/<proy>-vX.Y.Z.tar.gz` + `out/HASHES.md`. Usar `SOURCE_DATE_EPOCH=1700000000`, `tar --sort=name --mtime=... --owner=0 --group=0 --numeric-owner`.  
* **Verificación de corpus:** `make verify-corpus` recalcula el hash desde `SEED+SALT` y compara con `out/corpus_sha256.txt`.  
* **Idempotencia:** `make test-idem` corre dos veces y compara **hashes** de `out/` (sin diferencias).  
* **Bitácoras por sprint:** `docs/bitacora-sprint-{1..N}.md` con comandos, salidas recortadas y decisiones (timestamp).  
* **Pruebas y cobertura:** Suite (pytest/Bats) para funciones críticas (tokenización, máscaras, atención, decodificación). **Gate:** Cobertura **≥70%** en módulos no numéricos; exclusiones justificadas en `docs/cobertura.md` (lista blanca de **archivos**). Debe haber al menos **un** caso AAA/RGR (rojo->verde) por módulo implementado.  
* **Contrato de variables:** `README.md` con **tabla variable->efecto** (`SEED`, `SALT`, `CONTEXT`, `LR`, `HEADS`, `DIM`, `TOPK`, `TOPP`, `TEMP`, `BEAM`, `LENGTH_PENALTY`, `QUANT_BITS`, ...).  
* **Etiquetas simuladas:** `make tag` crea `vX.Y.Z` + `CHANGELOG.md` y "firma" en `out/tag_signature.txt`.  
* **Linting (si disponible):** `shellcheck` para Bash y `ruff` para Python (sin instalaciones nuevas).  
* **Límites prácticos:**  
  * Tamaño máximo de **cada archivo**: 10 MB. Tamaño total de `dist/`: 50 MB.  
  * **Prohibido:** Binarios ajenos al proyecto (auditar con `git lfs ls-files` o `git check-attr -a`).  
* **EOL/codificación:** Usar **LF** y **UTF-8** (ver `.gitattributes`).  

**Estructura mínima**

```
src/           # código del proyecto
tools/         # generadores y utilidades CLI (por ejemplo, gen_corpus.sh)
tests/         # pruebas (pytest/Bats)
docs/          # README, reporte, bitácoras, autoría, cobertura.md, video.md
out/           # métricas, trazas, tablas, logs, hashes, gráficos, env.txt
dist/          # paquete reproducible (tar.gz), artefactos finales
Makefile       # deps, build, data, tokenize, train, test, eval, bench, plot, pack, verify, verify-corpus, tag, test-idem, clean, distclean
.gitattributes # normaliza EOL/UTF-8
```

### Lista de proyectos

### 1.  Micro-Tokenizer & embeddings con *Perplexity*

**Contexto.** Antes del modelo, todo empieza por cómo partes el texto. Un buen tokenizador simplifica el problema y reduce casos fuera de vocabulario, uno malo infla el vocabulario y rompe la generalización.

**Objetivo**

* Implementar un tokenizador propio tipo BPE o Unigram.
* Generar un vocabulario y sus reglas/merges o probabilidades.
* Entrenar embeddings sobre corpus sintético y evaluar un modelo muy pequeño (unigrama, bigrama o red mínima) mediante la medida de sorpresa promedio del texto.
* Explorar similitud semántica con vecinos por coseno y analizar cómo se descompone una palabra desconocida en subpalabras.

**Alcance mínimo**

* Pipeline reproducible: limpiar -> tokenizar -> indexar -> entrenar embeddings -> evaluar.
* Guardar trazabilidad: vocabulario, merges o probabilidades, y ejemplos de tokenización.

**Métricas**

* Calidad del tokenizador: proporción de palabras que entran directo al vocabulario frente a las que se fragmentan,  tamaño del vocabulario, longitud promedio de la secuencia tokenizada.
* Calidad del modelo: sorpresa promedio más baja es mejor, ejemplos de similitud entre términos que deberían estar cerca (por ejemplo, sinónimos, variantes morfológicas).
* Eficiencia: tiempo de tokenización por mil líneas; tamaño del vocabulario.

**Pruebas esenciales**

* Tokenización determinista dado el mismo vocabulario.
* Reconstrucción: detokenizar(tokenizar(texto)) debe acercarse al texto original según tus reglas.
* Casos borde: números, puntuación, palabras muy largas y caracteres raros.

**Visualizaciones sugeridas**

* Histograma de longitudes de secuencia antes y después del tokenizador.
* Proyección de embeddings (por ejemplo, dos dimensiones) con etiquetas de palabras ejemplo.
* Tabla de ejemplos OOV y su descomposición en subpalabras.

**Errores comunes**

* No persistir el estado del tokenizador y regenerarlo distinto en cada corrida.
* Mezclar datos de entrenamiento y evaluación sin separar.

**Entregables.** `out/vocab.txt`, `out/tokens.jsonl`, `out/emb.tsv`, `out/metrics.json`.

### 2.  Atención: Scaled Dot-Product, multi-Head y costo O(n²)

**Contexto.** La atención es el núcleo de los transformers. Aquí se estudia su comportamiento, costo y estabilidad frente a la longitud de contexto.

**Objetivo**

* Implementar atención con máscara causal y versión multi-cabecera.
* Añadir una caché de claves y valores para acelerar la generación paso a paso.
* Comparar contra un RNN simple en una tarea sintética de memoria (por ejemplo, balanceo de paréntesis).

**Alcance mínimo**

* Atención que respete la causalidad y sea estable numéricamente.
* Micro KV-cache funcionando en inferencia autoregresiva.
* Benchmarks con diferentes longitudes de contexto.

**Métricas**

* Latencia media por paso y por secuencia; memoria pico utilizada.
* Exactitud en la tarea sintética (porcentaje correcto).
* Beneficio de la caché: reducción de tiempo por token al generar.

**Pruebas esenciales**

* La máscara debe impedir que un token vea el futuro.
* La salida con y sin caché debe coincidir en valores.
* Estabilidad: sin estallar a valores no finitos bajo entradas razonables.

**Visualizaciones sugeridas**

* Curva de latencia frente a longitud de contexto.
* Curva de memoria frente a longitud de contexto.
* Mapa de atención de un ejemplo ilustrativo.

**Errores comunes**

* Olvidar escalar las similitudes antes de la normalización.
* Implementar la caché pero volver a recalcular todo sin usarla de verdad.

**Entregables.** `out/bench.csv`, `out/plot_latencia.png`, `out/plot_memoria.png`.

### 3. Mini-Transformer (Decoder-Only, 1 bloque) con variantes posicionales (obligatorio para todos)

**Contexto.** Un solo bloque bien hecho permite estudiar de manera controlada cómo influyen las codificaciones posicionales en generalización a contextos más largos.

**Objetivo**

* Implementar un bloque completo: atención con máscara, residuales, normalización por capas, MLP con activación moderna.
* Probar al menos dos maneras de codificar posiciones (por ejemplo, sinusoidal y RoPE).
* Entrenar de forma estable con aumento progresivo de la tasa de aprendizaje e imponer un límite a la norma del gradiente.

**Alcance mínimo**

* Entrenamiento reproducible con semillas fijas y guardado de puntos de control.
* Evaluación en el mismo rango de contexto del entrenamiento y en contextos más largos.

**Métricas**

* Sorpresa promedio del texto en conjunto de prueba.
* Degradación cuando el contexto se extiende más allá de lo visto en entrenamiento.
* Estabilidad del entrenamiento: pérdidas que bajan y no muestran estancamientos graves.

**Pruebas esenciales**

* La máscara causal bloquea posiciones futuras.
* Las dos variantes posicionales producen salidas razonables y no se rompen en contextos largos.
* El recorte de gradiente efectivamente limita normas altas.

**Visualizaciones sugeridas.**

* Curva de entrenamiento y validación a lo largo de pasos.
* Tabla comparativa de sorpresa promedio por tipo posicional y por longitud de contexto.
* Gráfico de generalización: rendimiento fuera de rango.

**Errores comunes**

* Mezclar normalización antes y después de capas sin consistencia.
* Olvidar resetear semillas y registrar hiperparámetros en los artefactos.

**Entregables.** `out/perplexity.json`, `out/ctx_generalization.csv`, `dist/modelo-*.tar.gz`.


### 4. Encoder-Decoder vs Decoder-Only en *Seq2Seq* sintética 

**Contexto.** Muchas tareas reales son de transformación de secuencias. Este proyecto compara arquitecturas cuando hay un claro mapeo entrada-salida.

**Objetivo ampliado**

* Definir una tarea sintética con reglas claras (invertir cadenas, suma con acarreo, reglas morfológicas sencillas).
* Entrenar un encoder-decoder con atención y comparar con un decoder-only guiado por instrucciones o ejemplos en el mensaje.

**Alcance mínimo**

* Entrenamiento con refuerzo del profesor y prueba sin él para observar el sesgo de exposición.
* Conjuntos de entrenamiento y prueba bien separados.

**Métricas**

* Coincidencia exacta de la secuencia generada frente a la esperada.
* Exactitud por token cuando lo anterior es muy estricto.
* Si aplica, medida de coincidencia a nivel de n-gramas.

**Pruebas esenciales**

* El modelo no puede ver la salida durante la generación en evaluación.
* Separa muestras vistas y no vistas.

**Visualizaciones sugeridas**

* Tabla de errores típicos con ejemplos.
* Comparativa por longitud de entrada.

**Errores comunes**

* Sobrecargar el decoder-only con demasiada pista en el mensaje.

**Entregables.** `out/metrics_ed.json`, `out/metrics_do.json`, `out/ablation.md`.


### 5.  Técnicas de decodificación: Greedy, Beam, Top-K/Top-P, Temperatura, penalización por longitud

**Contexto.** La misma red puede producir salidas muy distintas según la estrategia de generación. Aquí se estudian compromisos entre calidad y diversidad.

**Objetivo ampliado**

* Implementar varias estrategias de decodificación y exponer parámetros.
* Comparar cómo cambian repetición, diversidad, longitud y sorpresa media de los tokens generados.

**Alcance mínimo**

* Usar el Mini-Transformer ya entrenado y probarlo sobre un corpus sencillo y controlado.
* Ejecutar cada estrategia con al menos dos configuraciones de parámetros.

**Métricas**

* Repetición: proporción de fragmentos repetidos.
* Diversidad: proporción de fragmentos únicos.
* Longitud media de las secuencias generadas.
* Sorpresa media: valores más altos indican salidas menos previsibles.

**Pruebas esenciales**

* Greedy debe ser determinista con la misma semilla.
* Beam con penalización por longitud no debe colapsar a respuestas mínimas.

**Visualizaciones sugeridas**

* Tabla de intercambio calidad-diversidad por estrategia.
* Gráfico de repetición frente a diversidad para distintos parámetros.
* Muestrario de salidas con la misma semilla.

**Errores comunes**

* Comparar estrategias con parámetros no equivalentes en esfuerzo computacional.
* No guardar las muestras de salida y luego no poder auditar.

**Entregables.** `out/samples/*.txt`, `out/metrics_decode.csv`, `out/tabla_tradeoffs.md`.

### 6. Escalado de inferencia y cuantización (Int8/Int4 simulada) con KV-Cache

**Contexto.** Cuando el modelo sirve tráfico real, el costo por token importa. La cuantización y la caché reducen uso de memoria y tiempo, con posibles pérdidas de calidad.

**Objetivo ampliado**

* Integrar caché de claves y valores en inferencia autoregresiva.
* Aplicar cuantización posterior al entrenamiento sobre los pesos (o pesos y activaciones si decides extender) y medir su efecto.

**Alcance mínimo**

* Comparar latencia y memoria con y sin caché.
* Comparar modelo en precisión completa y versiones cuantizadas.

**Métricas**

* Latencia media por token y por secuencia.
* Uso de memoria máximo durante la generación.
* Caída de calidad medida como incremento de la sorpresa promedio o pérdida en un conjunto fijo.

**Pruebas esenciales**

* La salida del modelo cuantizado debe ser razonablemente cercana a la de precisión completa en tareas sencillas.
* La caché debe mejorar la latencia a partir de contextos medios en adelante.

**Visualizaciones sugeridas**

* Curvas de latencia frente a longitud de contexto para cada variante.
* Barras de memoria usada por configuración.
* Tabla con calidad antes y después de cuantizar.

**Errores comunes**

* Aplicar cuantización sin desactivar puntos del grafo que no deben cuantizarse (por ejemplo, normalizaciones).
* Medir latencia sin una pasada de calentamiento ni repetir corridas.

**Entregables.** `out/bench_latency.csv`, `out/memory_usage.csv`, `out/accuracy_drop.json`.


### Presentación del trabajo y video

* **Video final** (máx. **5 min**, link en `docs/video.md`):  
  - `git log --oneline --decorate --graph -n 15`.
  - Ejecución real de `make verify` + **1** experimento clave (por ejemplo, ablación RoPE vs sinusoidal).
  - Mostrar `wc -c dist/proy-v1.0.0.tar.gz` y `sha256sum dist/proy-v1.0.0.tar.gz` para empatar con `out/HASHES.md`.
  - Lectura de métricas (`out/`) y contenido de `dist/`.
  - Conclusiones y próximos pasos  

* **Exposición oral** (7 min + 5 min de preguntas):  
  * README/lámina única.  
  * Modificar **en vivo** un hiperparámetro y explicar su impacto.  
  * Preguntas orales "¿cómo implementaste X?" para verificar autoría.  

#### Rúbrica única (20 pts)

**Gates previos (si falla alguno -> no se evalúa):**  
repo trazable  | corpus único (hash verificado)  | `make verify`  | pruebas + cobertura  | video   

1. **Técnica / correctitud** (2 pts)  
   * Implementación correcta (Mini-Transformer + módulo elegido).  
   * Consistencia numérica (Pérdidas decrecen, máscaras correctas).  
   * Tests numéricos máscara/softmax.  
   * Estabilidad/optimización (grad clipping, warmup, etc.).  

2. **Reproducibilidad y automatización** (2 pts)  
   * `Makefile` idempotente, targets requeridos, caché por timestamps, `pack`/`verify` funcionales, `out/HASHES.md`, `CHANGELOG.md`.  
   * **+1 pt** si el **tar** es determinista (mismo hash tras `distclean && make pack`).  
   * `make test-idem` sin cambios en `out/`.  

3. **Experimentos y métricas** (2 pts)  
   * Métricas adecuadas (perplexity, exact-match/accuracy/BLEU, distinct-n, latencia/memoria, accuracy drop).  
   * Ablaciones claras .  
   * Benchmarks con **σ** o intervalos de confianza.  

4. **Informe y evidencias** (2 pts)  
   * `docs/reporte.md` replicable, tablas/gráficos `out/`, **bitácoras** completas, **tabla variable->efecto**.  

5. **Video de ejecución** ( 6 pts)  
   * Flujo end-to-end real, `make verify`, experimento, `wc -c`/`sha256sum`, métricas y artefactos coherentes.  

6. **Exposición y preguntas** (6 pts)  
   * Dominio conceptual y defensa de decisiones.  

#### Sugerencia de Cronograma (5-7 días)

* **D1-2:** Setup (Makefile, datos sintéticos, tests), baseline del Mini-Transformer.  
* **D3-4:** Módulo complementario + ablación mínima, cobertura.  
* **D5-6:** Benchmarks (3 repeticiones), `pack`/`verify`, documentación.  
* **D7:** Video (5 min), pulir README, practicar exposición.  

**Recordatorio:** `make verify` y `make verify-corpus` deben pasar "limpio". El video debe mostrar exactamente lo empaquetado en `dist/`.  

### Plantillas iniciales a usar

#### 1) `Makefile`

Incluye `verify-corpus`, `test-idem`, empaquetado determinista, captura de entorno, `clean`/`distclean`.

```makefile
# Makefile para el proyecto CC0c2
# Uso: make [target]

.PHONY: deps build data tokenize train eval bench plot pack verify verify-corpus tag test test-idem clean distclean

# Reproducibilidad
SOURCE_DATE_EPOCH ?= 1700000000
SEED ?= 42
SALT ?= 1a2b3c4d5e6f7890abcdef1234567890
SEED_BENCH ?= 42

# Hiperparámetros por defecto
CONTEXT ?= 512
LR ?= 0.001
HEADS ?= 4
DIM ?= 128

deps:
	@echo "Verificando dependencias preinstaladas (stdlib, numpy, torch opcional)"
	python -c "import numpy; try: import torch; except: print('Torch no disponible')" || true

build:
	@echo "Chequeos básicos"
	# shellcheck tools/*.sh || true
	# ruff check src/*.py || true
	mkdir -p out dist

data:
	@echo "Generando corpus sintético"
	./tools/gen_corpus.sh $(SEED) $(SALT) > out/corpus.txt
	echo "Comando: ./tools/gen_corpus.sh $(SEED) $(SALT)" > out/seed.txt
	sha256sum out/corpus.txt | awk '{print $$1}' > out/corpus_sha256.txt

verify-corpus:
	@echo "Verificando hash del corpus"
	HGEN="$$(./tools/gen_corpus.sh $(SEED) $(SALT) | sha256sum | awk '{print $$1}')"; \
	HSAVED="$$(cat out/corpus_sha256.txt)"; test "$$HGEN" = "$$HSAVED"

tokenize: data
	@echo "Tokenizando corpus"
	python src/tokenizer.py out/corpus.txt --output out/tokens.jsonl --vocab out/vocab.txt

train: tokenize
	@echo "Entrenando modelo"
	python src/train.py --lr $(LR) --heads $(HEADS) --dim $(DIM) --input out/tokens.jsonl --output dist/model.tar.gz

eval: train
	@echo "Evaluando métricas"
	python src/eval.py dist/model.tar.gz --output out/metrics.json

bench:
	@echo "Benchmarking (3 repeticiones, reporte de sigma)"
	python src/bench.py --n $(CONTEXT) --seed $(SEED_BENCH) --warmup 1 --reps 3 --output out/bench.csv

plot: bench
	@echo "Generando gráficos"
	python src/plot.py out/bench.csv --output out/plot_latencia.png

test:
	@echo "Ejecutando tests"
	pytest tests/ --cov=src --cov-report=term-missing || bats tests/ || true

test-idem:
	@echo "Verificando idempotencia"
	rm -rf out/tmp && mkdir -p out/tmp
	$(MAKE) test eval bench plot
	rsync -a --delete out/ out/tmp/
	$(MAKE) test eval bench plot
	{ find out -type f ! -path 'out/tmp/*' ! -name 'hashes.txt' -exec sha256sum {} \; | sort > out/hashes.txt; }
	find out/tmp -type f -exec sha256sum {} \; | sort > out/tmp/hashes.txt
	diff -u out/tmp/hashes.txt out/hashes.txt

pack: eval bench plot
	@echo "Capturando entorno"
	{ \
	  echo "DATE=$$(date -u +%FT%TZ)"; \
	  python - <<'PY' || true
import platform, sys
print("PYTHON", sys.version.replace("\n"," "))
try:
    import numpy as np; print("NUMPY", np.__version__)
except Exception: print("NUMPY none")
try:
    import torch; print("TORCH", torch.__version__)
except Exception: print("TORCH none")
print("PLATFORM", platform.platform())
PY
	} > out/env.txt
	@echo "Empaquetando artefactos reproducibles"
	find out -type f -print0 | xargs -0 touch -d "@$(SOURCE_DATE_EPOCH)"
	rm -f dist/proy-v1.0.0.tar.gz
	tar --sort=name --mtime="@$(SOURCE_DATE_EPOCH)" --owner=0 --group=0 --numeric-owner \
	    -czf dist/proy-v1.0.0.tar.gz out/ \
	    --exclude='out/session.typescript' --exclude='out/terminal.cast' --exclude='out/*.png~'
	sha256sum dist/proy-v1.0.0.tar.gz | awk '{print $$1"  "$$2}' > out/HASHES.md

verify:
	@echo "Verificando hash del paquete"
	sha256sum -c out/HASHES.md

tag:
	@echo "Creando tag simulado"
	echo "v1.0.0: Versión inicial" > CHANGELOG.md
	echo "Firma simulada: $(shell date -u +%FT%TZ)" > out/tag_signature.txt

clean:
	rm -rf out/tmp out/hashes.txt

distclean: clean
	rm -rf out/* dist/* CHANGELOG.md
```

#### 2) `tools/gen_corpus.sh`

Semilla robusta con `SEED` (decimal) + `SALT` (hex), `set -euo pipefail`.

```bash
#!/bin/bash
set -euo pipefail
SEED=${1:-}; SALT=${2:-}
if [ -z "${SEED}" ] || [ -z "${SALT}" ]; then
  echo "Uso: $0 <SEED-decimal> <SALT-hex>" >&2; exit 1
fi

python - <<'PY' "$SEED" "$SALT"
import hashlib, sys, random
seed, salt = sys.argv[1], sys.argv[2]
h = hashlib.sha256(f"{seed}-{salt}".encode()).hexdigest()
random.seed(int(h[:16],16))  # 64 bits
N=50000
print(' '.join(f"word{random.randint(1,1000)}" for _ in range(N)))
PY
```

#### 3) `docs/bitacora-sprint-1.md`

Con AAA/RGR y referencias de hash.

```markdown
# Bitácora Sprint 1: Setup y Mini-Transformer
**Inicio:** [YYYY-MM-DD HH:MM]  **Equipo/Miembro:** [Nombres]

## Comandos
- [YYYY-MM-DD HH:MM] `make data` -> Corpus (SHA256 en `out/corpus_sha256.txt`)  
- [YYYY-MM-DD HH:MM] `pytest tests/test_transformer.py` -> 5/5 (cov=75%)

## AAA/RGR (ejemplo)
- **Arrange**: secuencia=128, máscara causal ON  
- **Act**: aplicar atención  
- **Assert**: logits futuras anuladas => (falló) índice corregido => (verde)

## Métricas
- Perplexity baseline: 10.3 (RoPE)

**Fin:** [YYYY-MM-DD HH:MM]
```

#### 4) `README.md`

Tabla variable->efecto y dependencias claras.

```markdown
# Proyecto CC0c2: Mini-Transformer + módulo complementario

## Dependencias
- Permitido: Python stdlib, numpy, torch (si preinstalado).
- Fallback: Usar NumPy para atención/MLP si torch no está disponible; justificar en `docs/reporte.md`.

## Uso rápido
```bash
make deps && make build && make data && make verify-corpus && make train && make eval
```

#### Variables y efectos

| Variable       | Descripción            | Efecto                                     |
|----------------|-----------------------|--------------------------------------------|
| SEED          | Semilla RNG           | Reproducibilidad de corpus/entrenamiento   |
| SALT          | ≥16 hex               | Unicidad del corpus (hash)                 |
| CONTEXT       | Longitud de contexto  | ↑memoria/latencia; +secuencias largas      |
| LR            | Learning rate         | Alto=rápido/inestable; bajo=lento/estable  |
| HEADS         | numero de heads           | +capacidad y costo O(n²)                   |
| DIM           | dimensión             | +capacidad, +costo                         |
| TOPK/TOPP/TEMP/BEAM/LENGTH_PENALTY/QUANT_BITS | Decodificación/cuantización. | Tradeoffs diversidad/calidad/latencia |

Ver `docs/autoría.md` y `docs/cobertura.md`.


#### 5) `docs/reporte.md`

Con σ/IC y ablaciones.

```markdown
# Reporte final

## Introducción
Mini-Transformer (decoder-only) con RoPE; módulo complementario: [indicar]. Ablación: RoPE vs sinusoidal.

## Métricas
- Perplexity (3 rep): 8.5 ± 0.3 (RoPE) vs 9.2 ± 0.4 (sinusoidal)  
- Latencia (ctx=512, 3 rep, warmup=1): 150 ms ± 10 (ver `out/bench.csv`)

## Ablaciones
- RoPE > sinusoidal en extrapolación (ctx>train)  
- Top-p (0.9) vs Beam (5): distinct-2 ↑, exact-match ↓

## Gráficos
![Latencia](out/plot_latencia.png)

## Conclusiones
[Hallazgos, limitaciones, próximos pasos]
```

#### 6) `docs/cobertura.md`

Justificación de exclusiones y acciones.

```markdown
# Cobertura

## Exclusiones (justificadas)
- src/kernels.py: núm. puro, probado indirectamente (tests de atención)

## Cobertura actual
- src/transformer.py: 75%
- src/decoder.py: 80%

## Acciones
- Agregado test AAA para softmax (rojo->verde por escalado)
```

#### 7) `.gitattributes`

Normaliza EOL/UTF-8.

```
* text=auto eol=lf
*.sh text eol=lf
*.py text eol=lf
*.md text eol=lf
```

