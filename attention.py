"""The temperature in the other softmax: attention logits / tau, on this model. Prints the mean entropy of an attention
row (last token, all heads and layers) over eight paragraphs per tau; the output is sampled at T = 0.7 throughout.

    python attention.py runs/ple-mb-s0-best.pt
"""
import math, sys, torch
import torch.nn.functional as F
sys.path.insert(0, "train")
import model as M
from measure import load
from tokenizers import Tokenizer

TAU = {"v": 1.0}
ENT = []
def forward(self, x, cos, sin):
    B, T, C = x.shape
    H, Dh = self.cfg.n_heads, self.cfg.head_dim
    q, k, v = self.qkv(x).split(C, dim=2)
    q = q.view(B, T, H, Dh).transpose(1, 2); k = k.view(B, T, H, Dh).transpose(1, 2); v = v.view(B, T, H, Dh).transpose(1, 2)
    q, k = M.apply_rope(q, cos, sin), M.apply_rope(k, cos, sin)
    s = (q @ k.transpose(-1, -2)) / (math.sqrt(Dh) * TAU["v"])           # sqrt(d) is the temperature; tau multiplies it
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    a = torch.softmax(s, -1)
    row = a[0, :, -1]                                                     # the last token's row, each head
    ENT.append(float(-(row * torch.log2(row.clamp_min(1e-12))).sum(-1).mean()))
    return self.proj((a @ v).transpose(1, 2).contiguous().view(B, T, C))
M.Attention.forward = forward

tok = Tokenizer.from_file("model/tokenizer.json")
model, cfg = load(sys.argv[1])
STARTS = ["The temperature of a body is", "When a gas is compressed", "It is evident that", "The molecules of a gas",
          "We shall now consider", "Heat is", "Die Temperatur eines Gases ist", "Man denke sich"]
import json
out = []
for tau in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    TAU["v"] = tau; ENT.clear(); top = []
    for si, st in enumerate(STARTS):
        torch.manual_seed(si)
        ids = tok.encode(st).ids
        with torch.no_grad():
            for _ in range(60):
                lg = model(torch.tensor([ids]))[0][0, -1]
                ids.append(int(torch.multinomial(torch.softmax(lg / 0.7, -1), 1)))
    out.append({"tau": tau, "attn_bits": round(sum(ENT) / len(ENT), 3)})
    print(out[-1], flush=True)
json.dump(out, open("charts/attention.json", "w"))
