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

- A 3M-parameter transformer trained on Maxwell's books (in English) and Boltzmann's lectures on gas theory (in his
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

## Build it

**Parts** (about the price of a nice lunch):

| part | why |
|---|---|
| ESP32-S3 dev board, N16R8 (16 MB flash, 8 MB PSRAM) | the two physicists live here; they need the PSRAM |
| 128x64 I2C OLED, 0.96" (SSD1306) | where they talk |
| SHT40 temperature sensor | the thermostat on their personalities |
| 4 dupont wires, or a μŠup (JST-SH 4-pin) cable if your OLED and SHT40 both have the little white connector | |
| USB cable | |

**Wiring.** The display and the thermometer share one I2C bus, so they go on the same four pins:

| OLED and SHT40 | ESP32-S3 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SDA | GPIO 18 |
| SCL | GPIO 8 |

If both modules have a μŠup connector (LaskaKit's do), click a μŠup cable from the SHT40 into the OLED and wire only
the OLED to the board. Never 5 V on these pins: Maxwell runs at 3.3 V.

**Flash.** From a clone of [esp32-ai](https://github.com/moudrkat/esp32-ai), with this repo's model exported into
`artifacts/maxwell/` (`model.bin`, `tokenizer.json`):

```
ROOM=1 ARTIFACTS=artifacts/maxwell scripts/deploy.sh tinystories
```

**Use.** Nothing. It starts writing. To change their mood:

- warm the SHT40 in your hand (it takes half a minute; thermometers are not in a hurry): they ramble, then switch languages;
- hold it near a cold window: they get stuck in loops;
- type a line over serial (`scripts/oled_talk.py` in esp32-ai) and they continue your sentence instead.

**If it doesn't:**

- screen dark: SDA and SCL swapped, or the OLED is an SH1106 (set `OLED_CONTROLLER` in `display.h`);
- the serial log says "no SHT40": the sensor isn't answering, so they assume a polite 22 C forever. Check the cable;
- flashing fails: hold BOOT, press RESET, let go of BOOT, try again.

## On the chip

The model is 3M parameters, 1.6 MB in int4. It writes paragraph after paragraph on its own, every word sampled at the temperature of the room
(`T = 0.1 + 0.15 * (degrees C - 18)`: 22 C gives 0.7, a warm hand around 30 C gives about 1.9). The reading heads
each paragraph on the screen. Type a line over serial and it continues that instead.

What they said on the bench, same start, int4 weights:

| T | Maxwell |
|---|---|
| 0.1 | The temperature of a body is the same as that of the body, and that the temperature of the body is the same as that of the body, and that... |
| 0.6 | The temperature of a body is therefore at the temperature T will be the same as that of the body, and the temperature of the air becomes the same as that of the hotter evaporation... |
| 1.6 | The temperature of a body is therefore at maciprocal period dannric will haben place whose continuity following tange... |

Cold Maxwell discovers a tautology and can't get out. Hot Maxwell starts speaking German. Boltzmann, warmed up,
returns the favour: *"...auf die Volumeneinheit abgeleitet werden kann. Direction of g; ..."*

## Status

Data, training, the chip firmware: done, not yet tried on a real thermometer. An interactive page: coming.

## Run it

```
./fetch.sh                      # the books, from archive.org
python clean.py                 # -> data/maxwell.txt
python prepare.py --vocab 2048  # -> data/vocab-2048/ for the esp32-ai trainer
```

Training uses `research/tinystories/train.py` from esp32-ai; `sample.py` lets them talk at a few temperatures.

MIT licence. The texts are in the public domain; Maxwell died in 1879, Boltzmann in 1906, and neither has complained.
