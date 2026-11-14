"""Prose from the archive.org OCR of the books listed in books.txt (as shipped: Maxwell in English, Boltzmann in his
German original, all public domain). Umlauts become ae/oe/ue/ss, since the chip is ASCII.

Keeps running prose and drops what OCR makes of the rest: the publisher's adverts, page headers and numbers,
equations and tables (lines that are mostly symbols), footnote markers. Rejoins words split across lines
("con- posed" -> "composed") and collapses OCR spacing. Writes data/maxwell.txt, one paragraph per line.

    python clean.py
"""
import re
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data" / "maxwell.txt"


def prose(line):
    """A line of running text: mostly letters, enough words, not shouting (headers are in capitals)."""
    s = line.strip()
    if len(s) < 25:
        return False
    letters = sum(c.isalpha() for c in s)
    if letters / len(s) < 0.75:
        return False
    words = s.split()
    if len(words) < 5:
        return False
    if sum(w.isupper() and len(w) > 1 for w in words) > len(words) / 3:
        return False
    if re.search(r"[=<>{}\\|^~]", s):
        return False
    return True


def clean(text, start=None):
    if start:
        i = text.find(start)
        text = text[i:] if i >= 0 else text
    text = re.sub(r"[ \t]+", " ", text)
    paras, cur = [], []
    for line in text.splitlines():
        if prose(line):
            cur.append(line.strip())
        elif cur:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    out = []
    for p in paras:
        p = re.sub(r"(\w)- (\w)", r"\1\2", p)            # words split across lines
        p = re.sub(r"\s+([,.;:?!])", r"\1", p)
        p = re.sub(r"\(\d+\)|\[\d+\]|\*|†|‡", "", p)
        p = re.sub(r"[^\x20-\x7e]", "", p)               # the device encoder takes ASCII
        p = re.sub(r"\s+", " ", p).strip()
        if len(p.split()) >= 12:
            out.append(p)
    return out


def german_ascii(text):
    """The chip and its screen are ASCII: umlauts the way an old typewriter wrote them."""
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("Ä", "Ae"), ("Ö", "Oe"), ("Ü", "Ue"), ("ß", "ss")):
        text = text.replace(a, b)
    return text


def books():
    """books.txt: (identifier, times, start marker or None)."""
    out = []
    for line in (Path(__file__).parent / "books.txt").read_text().splitlines():
        f = line.split("#")[0].split()
        if f:
            out.append((f[0], int(f[1]) if len(f) > 1 else 1, f[2] if len(f) > 2 else None))
    return out


def main():
    once, again = [], []
    for ident, times, start in books():
        ps = clean(german_ascii((RAW / f"{ident}.txt").read_text(errors="ignore")), start=start)
        print(f"{ident}: {len(ps):,} paragraphs" + (f", taken {times} times" if times > 1 else ""))
        once += ps
        again += ps * (times - 1)
    paras = once + again
    OUT.write_text("\n".join(paras) + "\n")
    words = sum(len(p.split()) for p in paras)
    print(f"{len(paras):,} paragraphs, {words:,} words -> {OUT}")


if __name__ == "__main__":
    main()
