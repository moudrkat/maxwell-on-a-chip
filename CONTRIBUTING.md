# Contributing

Small PRs are welcome, and so are issues that say "this didn't work, here is what I saw". I answer within a day.

**Run it before you change it**

```
cc -O2 -o host/maxwell host/maxwell.c -lm && host/maxwell 22     # the chip's code, on your computer
arduino-cli compile --fqbn esp32:esp32:esp32s3:PartitionScheme=custom,PSRAM=opi,FlashSize=16M firmware/maxwell
```

`host/maxwell 19`, `22` and `28` should print about 0.2, 2.8 and 8.8 bits per word. If your change moves those, say so
in the PR.

**Things nobody has done yet**

- run it on a real board with a real SHT40, and send a photo of the screen;
- the SH1106 display path;
- another physicist (`books.txt`), or another language pair;
- an interactive page: the same model in a browser, with a thermometer you drag.

**House rules**

- say what you measured and what you didn't; "compiles, not tried on hardware" is a fine sentence;
- keep the README dry. Maxwell would have wanted it that way.
