"""Let Maxwell write at a few temperatures, on the int4 weights the chip runs: the same start, softmax at T.

    PYTHONPATH=<esp32-ai>/src:<esp32-ai> python sample.py <ckpt> --data-dir data --temps 0.2 0.7 1.0 1.5
"""
import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from research.oled.inside import load_int4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--start", default="The temperature of a body is")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.2, 0.7, 1.0, 1.5])
    ap.add_argument("--tokens", type=int, default=60)
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(args.data_dir) / f"vocab-{args.vocab}" / "tokenizer.json"))
    model, cfg = load_int4(args.ckpt, "cpu")
    for T in args.temps:
        torch.manual_seed(0)
        ids = tok.encode(args.start).ids
        with torch.no_grad():
            for _ in range(args.tokens):
                logits, _ = model(torch.tensor([ids[-cfg["seq_len"]:]]))
                p = torch.softmax(logits[0, -1] / T, -1)        # Maxwell-Boltzmann: exp(-E / kT), E = -logit
                ids.append(int(torch.multinomial(p, 1)))
        print(f"T = {T}: {tok.decode(ids)}\n")


if __name__ == "__main__":
    sys.exit(main())
