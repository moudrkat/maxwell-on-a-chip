#!/bin/sh
# From the books to a model the chip runs: fetch, clean, tokenize, train, export to model/.
# Needs a GPU for the training (a few minutes); torch, numpy and tokenizers.
set -e
cd "$(dirname "$0")"
./fetch.sh
python clean.py
python prepare.py --vocab 2048
PYTHONPATH=train python train/train.py --arm ple --vocab 2048 --data-dir data --d-model 192 --n-layers 5 --n-heads 6 \
  --ple-dim 32 --fixed-ffn 512 --seq-len 128 --batch-size 64 --steps 12000 --lr 1e-3 --warmup 200 --eval-every 500 \
  --dropout 0.3 --tag mb --save-best
PYTHONPATH=train python train/export.py ple-mb-s0-best --tokenizer data/vocab-2048/tokenizer.json --out model
