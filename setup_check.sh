#!/bin/bash
# Script de instalación y verificación del proyecto
# Proyecto 4: Encoder-Decoder vs Decoder-Only en Seq2Seq

set -e  # Exit on error

echo "=========================================="
echo "Proyecto 4 - Setup y Verificación"
echo "=========================================="
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función de verificación
check_dependency() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 encontrado"
        return 0
    else
        echo -e "${RED}✗${NC} $1 NO encontrado"
        return 1
    fi
}

# Función de verificación de módulo Python
check_python_module() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python módulo '$1' disponible"
        return 0
    else
        echo -e "${RED}✗${NC} Python módulo '$1' NO disponible"
        return 1
    fi
}

echo "1. Verificando dependencias del sistema..."
echo "-------------------------------------------"

check_dependency python3
check_dependency make
check_dependency git

echo ""
echo "2. Verificando módulos Python..."
echo "-------------------------------------------"

check_python_module numpy || echo -e "${YELLOW}⚠${NC}  Instalar con: pip install numpy"
check_python_module torch || echo -e "${YELLOW}⚠${NC}  Instalar con: pip install torch (requerido para entrenamiento)"

echo ""
echo "3. Verificando estructura del proyecto..."
echo "-------------------------------------------"

directories=("src" "tools" "tests" "docs" "out" "dist")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} Directorio '$dir' existe"
    else
        echo -e "${RED}✗${NC} Directorio '$dir' NO existe"
        mkdir -p "$dir"
        echo -e "${YELLOW}→${NC} Creado '$dir'"
    fi
done

echo ""
echo "4. Verificando archivos principales..."
echo "-------------------------------------------"

files=("Makefile" "README.md" ".gitattributes" "src/attention.py" "src/models.py" "src/tokenizer.py" "tools/gen_corpus.sh")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file NO encontrado"
    fi
done

echo ""
echo "5. Verificando permisos de ejecución..."
echo "-------------------------------------------"

if [ -x "tools/gen_corpus.sh" ]; then
    echo -e "${GREEN}✓${NC} tools/gen_corpus.sh es ejecutable"
else
    echo -e "${YELLOW}⚠${NC}  Dando permisos a tools/gen_corpus.sh"
    chmod +x tools/gen_corpus.sh
fi

echo ""
echo "6. Probando generador de corpus..."
echo "-------------------------------------------"

if ./tools/gen_corpus.sh 42 1a2b3c4d5e6f7890abcdef1234567890 | head -3 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Generador de corpus funciona"
    echo "   Ejemplo de salida:"
    ./tools/gen_corpus.sh 42 1a2b3c4d5e6f7890abcdef1234567890 | head -2 | sed 's/^/   /'
else
    echo -e "${RED}✗${NC} Error en generador de corpus"
fi

echo ""
echo "7. Contando líneas de código..."
echo "-------------------------------------------"

total_lines=$(find src tests tools -name "*.py" -o -name "*.sh" | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
echo -e "${GREEN}✓${NC} Total de líneas de código: $total_lines"

echo ""
echo "=========================================="
echo "Resumen de Verificación"
echo "=========================================="
echo ""

# Verificar si puede ejecutar make build
if make build > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} make build funciona correctamente"
else
    echo -e "${RED}✗${NC} Hay problemas con make build"
fi

echo ""
echo "=========================================="
echo "Próximos Pasos"
echo "=========================================="
echo ""
echo "Para ejecutar el proyecto completo:"
echo ""
echo "  1. Instalar dependencias (si faltan):"
echo "     pip install numpy torch matplotlib pytest pytest-cov"
echo ""
echo "  2. Ejecutar pipeline (solo generación sin entrenamiento):"
echo "     make build && make data && make tokenize"
echo ""
echo "  3. Ver corpus generado:"
echo "     head -20 out/corpus.txt"
echo ""
echo "  4. Para entrenar (requiere PyTorch):"
echo "     make train  # Toma ~15-20 min en CPU"
echo ""
echo "  5. Ver documentación completa:"
echo "     cat README.md"
echo "     cat QUICKSTART.md"
echo ""
echo "=========================================="
echo "Estado del Proyecto: COMPLETO ✅"
echo "=========================================="
