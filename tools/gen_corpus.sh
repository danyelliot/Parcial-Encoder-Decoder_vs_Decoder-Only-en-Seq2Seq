#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
SALT="${2:-deadbeef}"
VOCAB_SIZE=100
MIN_LEN=3
MAX_LEN=10
NUM_SAMPLES=5000

for i in $(seq 1 $NUM_SAMPLES); do
  hash=$(echo -n "${SEED}-${SALT}-${i}" | sha256sum | awk '{print $1}')
  len=$(( (0x${hash:0:2} % (MAX_LEN - MIN_LEN + 1)) + MIN_LEN ))

  tokens=()
  for j in $(seq 0 $((len - 1))); do
    offset=$(( j * 2 + 2 ))
    tok=$(( 0x${hash:$offset:2} % VOCAB_SIZE ))
    tokens+=($tok)
  done

  src=""
  for tok in "${tokens[@]}"; do
    src="${src}w${tok} "
  done
  src="${src% }"

  tgt=""
  for ((k=${#tokens[@]}-1; k>=0; k--)); do
    tgt="${tgt}w${tokens[k]} "
  done
  tgt="${tgt% }"

  echo "${src} ||| ${tgt}"
done
