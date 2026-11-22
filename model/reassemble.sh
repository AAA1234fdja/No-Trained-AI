#!/usr/bin/env bash
# Reensambla model.safetensors a partir de las 10 partes subidas a GitHub.
# Cada parte tiene <= 80 MB (bajo el límite de 100 MB/archivo de GitHub).
# Uso: bash model/reassemble.sh   (desde la raíz del repositorio)
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODEL_DIR"

# Checksums SHA-256 de cada parte (verificados con sha256sum)
declare -A EXPECTED_SHA256
EXPECTED_SHA256[aa]="9b891d4b57ab172605a7ddaf85a737c4edc9d9f004264a640d21e46eab38dd77"
EXPECTED_SHA256[ab]="b2a24dbc8da14e05d5d97c31287ab0d080d82fbf28a57ff934410e0253f00c66"
EXPECTED_SHA256[ac]="08a2741d5e525d3f50ddc81e85005b0282e5b93dfd106f63c7813966f5c4d63a"
EXPECTED_SHA256[ad]="900961f5b0c3c813ec182b389ef821010d7d6ec31f5dcf19a686a39f631b6db3"
EXPECTED_SHA256[ae]="bdf0dd16593a9ddae88e733fd9552850ff053d2b54c68ecdffb26f1178f492d9"
EXPECTED_SHA256[af]="b8369134e7954fbfcf246e5002e40c150a6ea3c953f38ac1b89d8a3aa26b4644"
EXPECTED_SHA256[ag]="411c38a10b93ba74433c20ef4a1b2c11bf33573b2bc923eb60106153c8292978"
EXPECTED_SHA256[ah]="2d5943844a9de940df2c2b521120e3fded82fc4574cffae884b414b08f00206d"
EXPECTED_SHA256[ai]="c6caf9ee82d1b5f2fc4e30f134b376391dd5152f9e721893566bccf6cde7c627"
EXPECTED_SHA256[aj]="22debdb8417d67ecf93fa5cf8079944c77aa868e6aae73a011115153be99c23d"

# Verificar que existen las partes
if ! ls model.safetensors.part.* 1>/dev/null 2>&1; then
    echo "ERROR: No se encontraron partes model.safetensors.part.* en $MODEL_DIR"
    exit 1
fi

# Verificar checksums de cada parte
ALL_OK=true
for part in model.safetensors.part.*; do
    suffix="${part##*.}"  # aa, ab, ac, ...
    actual=$(sha256sum "$part" | awk '{print $1}')
    expected="${EXPECTED_SHA256[$suffix]}"
    if [ "$actual" != "$expected" ]; then
        echo "ADVERTENCIA: Checksum mismatch para $part"
        echo "  Esperado: $expected"
        echo "  Obtenido: $actual"
        ALL_OK=false
    else
        echo "OK: $part (SHA-256 verificado)"
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ALGUNAS PARTES TENEN CHECKSUM INCORRECTO. Vuelve a descargarlas."
    exit 1
fi

# Reensamblar
cat model.safetensors.part.* > model.safetensors
echo ""
echo "Reensamblado: model.safetensors ($(du -h model.safetensors | cut -f1))"

# Verificar tamaño esperado (~801 MB)
expected_size=801224536
actual_size=$(stat -c%s model.safetensors 2>/dev/null || stat -f%z model.safetensors 2>/dev/null)
if [ "$actual_size" != "$expected_size" ]; then
    echo "ADVERTENCIA: Tamaño inesperado ($actual_size vs $expected_size bytes)"
else
    echo "OK: Tamaño correcto ($actual_size bytes)"
fi

echo ""
echo "Listo. Ahora puedes ejecutar el modelo con scripts/inference_engine.py"
