"""Maxwell's prose from the archive.org OCR of his books (Theory of Heat, The Scientific Papers, the Treatise on Electricity
and Magnetism, Matter and Motion, the Elementary Treatise), and Boltzmann in his German original (Vorlesungen ueber
Gastheorie I-II, Populaere Schriften), all public domain. Umlauts become ae/oe/ue/ss, since the chip is ASCII.

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


def main():
    paras = []
    paras += clean((RAW / "theoryofheat00maxwuoft.txt").read_text(errors="ignore"), start="INTRODUCTION")
    for v in ("scientificpapers01maxwuoft.txt", "scientificpapers02maxwuoft.txt", "treatiseonelectr01maxwrich.txt",
              "treatiseonelectr02maxw.txt", "mattermotion00maxwuoft.txt", "elementarytreati00maxwuoft.txt"):
        paras += clean((RAW / v).read_text(errors="ignore"))
    n_max = len(paras)
    for v in ("vorlesungenber01bolt.txt", "vorlesungenber02bolt.txt", "populreschrifte00boltgoog.txt"):   # in German
        paras += clean(german_ascii((RAW / v).read_text(errors="ignore")))
    n_bol = len(paras) - n_max
    # two gases in more even amounts: Boltzmann is a sixth of the text, so his paragraphs go in twice (the mix ratio is
    # a choice the model's distribution follows; README says it)
    paras += paras[n_max:]
    print(f"Maxwell {n_max:,} paragraphs (English), Boltzmann {n_bol:,} (German, original), Boltzmann taken twice")
    OUT.write_text("\n".join(paras) + "\n")
    words = sum(len(p.split()) for p in paras)
    print(f"{len(paras):,} paragraphs, {words:,} words -> {OUT}")


if __name__ == "__main__":
    main()
