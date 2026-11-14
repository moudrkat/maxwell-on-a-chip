"""Let them talk at a few temperatures, on the int4 weights the chip runs.

    python sample.py runs/ple-mb-s0-best.pt --temps 0.1 0.7 1.9 --start "The temperature of a body is"
"""
import argparse

import torch
from tokenizers import Tokenizer

from measure import load


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer", default="model/tokenizer.json")
    ap.add_argument("--start", default="The temperature of a body is")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.1, 0.7, 1.9])
    ap.add_argument("--tokens", type=int, default=60)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    model, cfg = load(args.ckpt)
    for T in args.temps:
        torch.manual_seed(0)
        ids = tok.encode(args.start).ids
        with torch.no_grad():
            for _ in range(args.tokens):
                logits = model(torch.tensor([ids[-cfg["seq_len"]:]]))[0][0, -1]
                ids.append(int(torch.multinomial(torch.softmax(logits / T, -1), 1)))   # exp(-E/kT), E = -logit
        print(f"T = {T}: {tok.decode(ids)}\n")


if __name__ == "__main__":
    main()
