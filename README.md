# Maxwell on a chip

In my PhD interview, someone asked me to write down the attention equation.

I started writing Boltzmann.

I wasn't wrong. Attention weights are a Boltzmann distribution: `exp(score) / sum(exp(score))`, with `sqrt(d)` sitting in
the place of the temperature. Same equation, different handwriting.

So I did the only reasonable thing. I trained a tiny language model on the two people who gave us that equation,
James Clerk Maxwell and Ludwig Boltzmann, put it on a chip, and let the room set its temperature.

Cold room: they get stuck. *"The temperature of a body is proportional to the pressure of the pressure of the pressure
of the..."* Warm it with your hand: they start to ramble. Somewhere in between, they almost make sense, which is more
than can be said for most of us at room temperature.

## What it is

- A ~3M-parameter transformer trained on Maxwell's books (in English) and Boltzmann's lectures on gas theory (in his
  German original). One model, two physicists, two languages.
- It runs on an ESP32-S3 (the runtime from [esp32-ai](https://github.com/moudrkat/esp32-ai)). It samples every word with
  `softmax(logits / T)`: the Maxwell–Boltzmann distribution, applied to Maxwell and Boltzmann.
- `T` comes from a real thermometer (SHT40) next to the chip.
- Cold, each of them sticks to his own language. Hot, the languages start to mix, like two gases. That is the other
  thing Boltzmann is famous for.

## The small print

- The texts are archive.org OCR, cleaned by `clean.py`: equations and tables are dropped (OCR turns them into confetti),
  so this Maxwell never writes a formula. He just talks about them.
- Boltzmann is one sixth of the text, so his paragraphs go in twice, to make the two gases more even. The mix in the
  output follows that choice: the model's distribution is only as fair as its data.
- Umlauts are written ae, oe, ue, ss. The chip is ASCII, like a typewriter from 1896.
- With about a million words, a model this size learns a fair amount by heart. Cold, it loops rather than recites.

## Status

Work in progress. Data and training: done. The chip firmware (temperature from the SHT40) and an interactive page:
coming.

## Run it

```
./fetch.sh                      # the books, from archive.org
python clean.py                 # -> data/maxwell.txt
python prepare.py --vocab 2048  # -> data/vocab-2048/ for the esp32-ai trainer
```

Training uses `research/tinystories/train.py` from esp32-ai; `sample.py` lets them talk at a few temperatures.

MIT licence. The texts are in the public domain; Maxwell died in 1879, Boltzmann in 1906, and neither has complained.
