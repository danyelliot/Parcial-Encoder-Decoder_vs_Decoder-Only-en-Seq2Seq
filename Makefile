# Makefile para el proyecto CC0C2 - Proyecto 4: Encoder-Decoder vs Decoder-Only
# Uso: make [target]

.PHONY: deps build data tokenize train eval bench plot pack verify verify-corpus tag test test-idem clean distclean

# Reproducibilidad
SOURCE_DATE_EPOCH ?= 1700000000
SEED ?= 42
SALT ?= 1a2b3c4d5e6f7890abcdef1234567890
SEED_BENCH ?= 42

# Hiperparámetros por defecto
CONTEXT ?= 128
LR ?= 0.001
HEADS ?= 4
DIM ?= 128
EPOCHS ?= 20
BATCH_SIZE ?= 32
DROPOUT ?= 0.1

# Parámetros de decodificación
TEACHER_FORCING ?= 0.5
MAX_LENGTH ?= 50

deps:
	@echo "Verificando dependencias preinstaladas (stdlib, numpy, torch opcional)"
	python3 -c "import numpy; print('NumPy:', numpy.__version__)" || { echo "NumPy requerido"; exit 1; }
	python3 -c "try: import torch; print('Torch:', torch.__version__); except: print('Torch no disponible, usando NumPy')" || true

build:
	@echo "Chequeos básicos"
	@command -v shellcheck >/dev/null 2>&1 && shellcheck tools/*.sh || echo "shellcheck no disponible, omitiendo"
	@command -v ruff >/dev/null 2>&1 && ruff check src/*.py || echo "ruff no disponible, omitiendo"
	mkdir -p out dist

data:
	@echo "Generando corpus sintético"
	./tools/gen_corpus.sh $(SEED) $(SALT) > out/corpus.txt
	echo "Comando: ./tools/gen_corpus.sh $(SEED) $(SALT)" > out/seed.txt
	sha256sum out/corpus.txt | awk '{print $$1}' > out/corpus_sha256.txt

verify-corpus:
	@echo "Verificando hash del corpus"
	@HGEN="$$(./tools/gen_corpus.sh $(SEED) $(SALT) | shasum -a 256 | awk '{print $$1}')"; \
	HSAVED="$$(cat out/corpus_sha256.txt)"; \
	if [ "$$HGEN" = "$$HSAVED" ]; then \
		echo "✓ Corpus verificado correctamente"; \
	else \
		echo "✗ Hash del corpus no coincide"; \
		echo "  Generado: $$HGEN"; \
		echo "  Guardado:  $$HSAVED"; \
		exit 1; \
	fi

tokenize: data
	@echo "Tokenizando corpus"
	python3 src/tokenizer.py out/corpus.txt --output out/tokens.jsonl --vocab out/vocab.txt

train: tokenize
	@echo "Entrenando modelos Encoder-Decoder y Decoder-Only"
	python3 src/train.py --lr $(LR) --heads $(HEADS) --dim $(DIM) --context $(CONTEXT) \
		--epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --dropout $(DROPOUT) \
		--teacher-forcing $(TEACHER_FORCING) \
		--input out/tokens.jsonl --vocab out/vocab.txt \
		--output-ed dist/model_encoder_decoder.tar.gz \
		--output-do dist/model_decoder_only.tar.gz

eval: train
	@echo "Evaluando métricas de ambos modelos"
	python3 src/eval.py dist/model_encoder_decoder.tar.gz dist/model_decoder_only.tar.gz \
		--vocab out/vocab.txt \
		--output-ed out/metrics_ed.json \
		--output-do out/metrics_do.json \
		--output-ablation out/ablation.md

bench:
	@echo "Benchmarking (3 repeticiones, reporte de sigma)"
	python3 src/bench.py --n $(CONTEXT) --seed $(SEED_BENCH) --warmup 1 --reps 3 \
		--model-ed dist/model_encoder_decoder.tar.gz \
		--model-do dist/model_decoder_only.tar.gz \
		--vocab out/vocab.txt \
		--output out/bench.csv

plot: bench eval
	@echo "Generando gráficos"
	python3 src/plot.py out/bench.csv out/metrics_ed.json out/metrics_do.json \
		--output-dir out/

test:
	@echo "Ejecutando tests"
	@if command -v pytest >/dev/null 2>&1; then \
		pytest tests/ --cov=src --cov-report=term-missing || true; \
	else \
		echo "pytest no disponible, ejecutando tests con unittest"; \
		python3 -m unittest discover tests/ || true; \
	fi

test-idem:
	@echo "Verificando idempotencia"
	rm -rf out/tmp && mkdir -p out/tmp
	$(MAKE) test eval bench plot
	rsync -a --delete out/ out/tmp/
	$(MAKE) test eval bench plot
	find out -type f ! -path 'out/tmp/*' ! -name 'hashes.txt' -exec shasum -a 256 {} \; | sort > out/hashes.txt
	find out/tmp -type f -exec shasum -a 256 {} \; | sort > out/tmp/hashes.txt
	diff -u out/tmp/hashes.txt out/hashes.txt && echo "✓ Idempotencia verificada" || echo "✗ Diferencias encontradas"

pack: eval bench plot
	@echo "Capturando entorno"
	@date -u +%FT%TZ > out/env.txt
	@python3 -c "import sys; print('PYTHON', sys.version.replace('\n',' '))" >> out/env.txt || true
	@python3 -c "import numpy; print('NUMPY', numpy.__version__)" >> out/env.txt || echo "NUMPY none" >> out/env.txt
	@python3 -c "import torch; print('TORCH', torch.__version__)" >> out/env.txt || echo "TORCH none" >> out/env.txt
	@python3 -c "import platform; print('PLATFORM', platform.platform())" >> out/env.txt || true
	@echo "Empaquetando artefactos reproducibles"
	@find out -type f -print0 | xargs -0 touch -t "$$(date -r $(SOURCE_DATE_EPOCH) +%Y%m%d%H%M.%S)" 2>/dev/null || \
	find out -type f -print0 | xargs -0 touch -d "@$(SOURCE_DATE_EPOCH)"
	rm -f dist/proy4-v1.0.0.tar.gz
	tar --sort=name \
	    -czf dist/proy4-v1.0.0.tar.gz out/ \
	    --exclude='out/session.typescript' --exclude='out/terminal.cast' --exclude='out/*.png~'
	shasum -a 256 dist/proy4-v1.0.0.tar.gz | awk '{print $$1"  "$$2}' > out/HASHES.md
	@echo "✓ Paquete creado: dist/proy4-v1.0.0.tar.gz"

verify:
	@echo "Verificando hash del paquete"
	@cd out && shasum -a 256 -c HASHES.md && echo "✓ Paquete verificado" || echo "✗ Verificación fallida"

tag:
	@echo "Creando tag simulado"
	echo "v1.0.0: Implementación Encoder-Decoder vs Decoder-Only en Seq2Seq" > CHANGELOG.md
	echo "- Tarea sintética: inversión de secuencias" >> CHANGELOG.md
	echo "- Ambos modelos con atención" >> CHANGELOG.md
	echo "- Ablación: con/sin teacher forcing" >> CHANGELOG.md
	echo "Firma simulada: $$(date -u +%FT%TZ)" > out/tag_signature.txt

clean:
	rm -rf out/tmp out/hashes.txt

distclean: clean
	rm -rf out/* dist/* CHANGELOG.md
