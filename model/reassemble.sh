#!/usr/bin/env bash
# Reensambla model.safetensors a partir de las 10 partes subidas a GitHub.
# Cada parte tiene <= 80 MB (bajo el límite de 100 MB/archivo de GitHub).
# Uso: bash model/reassemble.sh   (desde la raíz del repositorio)
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODEL_DIR"

if ls model.safetensors.part.* 1>/dev/null 2>&1; then
    cat model.safetensors.part.* > model.safetensors
    echo "Reensamblado: model.safetensors ($(du -h model.safetensors | cut -f1))"
else
    echo "No se encontraron partes model.safetensors.part.* en $MODEL_DIR"
    exit 1
fi
