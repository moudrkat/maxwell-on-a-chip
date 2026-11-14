"""Tokenize data/maxwell.txt for the esp32-ai trainer: data/vocab-<N>/{tokenizer.json, train.bin, val.bin, manifest.json}.
ByteLevel BPE with one special token (end of text between paragraphs), the recipe the device encoder and decoder
expect; 3% of paragraphs held out for validation.

    python prepare.py --vocab 1024
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
DATA = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vocab", type=int, default=1024)
    args = ap.parse_args()
    paras = [p for p in (DATA / "maxwell.txt").read_text().splitlines() if p.strip()]
    uniq = sorted(set(paras))
    random.Random(0).shuffle(uniq)
    held = set(uniq[:max(100, len(uniq) * 3 // 100)])           # by paragraph, so a repeated one stays on one side
    splits = {"val": [p for p in uniq if p in held], "train": [p for p in paras if p not in held]}
    random.Random(1).shuffle(splits["train"])
    out = DATA / f"vocab-{args.vocab}"
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(splits["train"], trainer=trainers.BpeTrainer(
        vocab_size=args.vocab, special_tokens=[EOT], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    cfg = json.loads(tok.to_str()); cfg["added_tokens"] = []
    (out / "tokenizer.json").write_text(json.dumps(cfg))
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    counts = {}
    for name, part in splits.items():
        ids = []
        for enc in tok.encode_batch(part):
            ids += enc.ids + [eot]
        np.array(ids, dtype=np.uint16).tofile(out / f"{name}.bin")
        counts[name] = len(ids)
    manifest = {"source": "James Clerk Maxwell, Theory of Heat (1899) and The Scientific Papers (1890), archive.org OCR, clean.py",
                "paragraphs": len(paras), "tokens": counts, "vocab": args.vocab, "eot_id": eot,
                "tokenizer_sha256": hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(counts), "->", out)


if __name__ == "__main__":
    main()
