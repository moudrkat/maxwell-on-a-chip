"""Measure what the charts in the README show, on the int4 weights the chip runs:
  - the next-word distribution after "The temperature of a body is" at a few temperatures;
  - entropy of each sampled word (bits) against temperature;
  - how much German gets into paragraphs that start in English, against temperature (share of German among the
    common function words; 48 paragraphs per temperature), so a stray "the" in German or "der" in English is what's counted).
  - the entropy the demon (only the 4 likeliest words) would leave at the same step.
Writes charts/measure.json.

    python measure.py runs/ple-mb-s0-best.pt --data data/vocab-2048
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent / "train"))
from export import quant_pack  # noqa: E402
from model import Config, make_model  # noqa: E402

EN = set("the of and to in is that it by be as which this for with are on not or from at we its an".split())
DE = set("der die das und ist in den von zu mit sich des auf dem nicht ein eine als auch wird so dass wir sind bei".split())
STARTS_EN = ["The temperature of a body is", "When a gas is compressed", "It is evident that", "The molecules of a gas",
             "We shall now consider", "Heat is"]


def load(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    base = Config(**{k: v for k, v in cfg.items() if k in Config.__dataclass_fields__})
    model = make_model("ple", 0, base, verbose=False, fixed_ffn=cfg.get("ffn_hidden"))
    sd = ck["state"]
    for k, v in sd.items():
        if v.dim() == 2 and "norm" not in k:
            sd[k] = torch.as_tensor(quant_pack(v)[2]).reshape(v.shape).to(v.dtype)
    model.load_state_dict(sd)
    return model.eval(), cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="data/vocab-2048")
    ap.add_argument("--out", default="charts/measure.json")
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(args.data) / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    model, cfg = load(args.ckpt)
    out = {"dist": {}, "sweep": []}
    with torch.no_grad():
        ids = tok.encode(STARTS_EN[0]).ids
        logits = model(torch.tensor([ids]))[0][0, -1]
        for T in (0.1, 0.7, 1.9):
            p = torch.softmax(logits / T, -1)
            top = torch.topk(p, 12)
            out["dist"][str(T)] = [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(top.values, top.indices)]
        rng = random.Random(0)
        for T in [0.05] + [round(0.1 * k, 1) for k in range(1, 21)]:
            bits, dbits, en, de = [], [], 0, 0
            for r in range(48):
                torch.manual_seed(1000 * r + int(T * 100))
                ids = tok.encode(STARTS_EN[r % len(STARTS_EN)]).ids
                start = len(ids)
                for _ in range(60):
                    lg = model(torch.tensor([ids[-cfg["seq_len"]:]]))[0][0, -1] / T
                    p = torch.softmax(lg, -1)
                    bits.append(float(-(p * torch.log2(p.clamp_min(1e-12))).sum()))
                    q = torch.softmax(torch.topk(lg, 4).values, -1)      # what Maxwell's demon would let through
                    dbits.append(float(-(q * torch.log2(q.clamp_min(1e-12))).sum()))
                    nxt = int(torch.multinomial(p, 1))
                    if nxt == eot:
                        break
                    ids.append(nxt)
                words = tok.decode(ids[start:]).lower().split()
                en += sum(w in EN for w in words); de += sum(w in DE for w in words)
            out["sweep"].append({"T": T, "entropy_bits": round(sum(bits) / len(bits), 3),
                                 "demon_bits": round(sum(dbits) / len(dbits), 3),
                                 "german_share": round(de / max(1, en + de), 3)})
            print(out["sweep"][-1], flush=True)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
