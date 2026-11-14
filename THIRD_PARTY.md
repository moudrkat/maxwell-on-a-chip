# Code from other projects

`runtime/`, `firmware/maxwell/` (apart from `room_temp.h` and the ROOM_TEMPERATURE parts of `maxwell.ino`), `train/`
and `tools/` come from [esp32-ai](https://github.com/moudrkat/esp32-ai), which builds on Viacheslav Sierbov's
[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai): the runtime that runs a language model on this chip in the first
place. Without it, Maxwell would have nowhere to live. MIT licence:

```
MIT License

Copyright (c) 2026 Viacheslav Sierbov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
