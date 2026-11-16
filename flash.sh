#!/bin/sh
# Build the firmware and put Maxwell and Boltzmann on the board: the model into its flash partition, then the program.
#   ./flash.sh [port]            (default /dev/ttyACM0)
#   SHT=4,5 ./flash.sh           the thermometer on its own two pins (SDA,SCL) instead of sharing the display's
#   WORD="The temperature of a body is" ./flash.sh    start on one word: the screen shows its distribution, live
# Needs arduino-cli with the esp32 core and the Adafruit SSD1306 + GFX libraries:
#   arduino-cli core install esp32:esp32
#   arduino-cli lib install "Adafruit SSD1306" "Adafruit GFX Library"
set -e
cd "$(dirname "$0")"
PORT=${1:-/dev/ttyACM0}
FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi'
ONE=firmware/maxwell/generated/one_word.h
if [ -n "${WORD:-}" ]; then
  printf '#define ROOM_WORD "%s"\n' "$WORD" > "$ONE"
  echo "starts on one word, after: $WORD"
else
  rm -f "$ONE"
fi
ESPTOOL=$(command -v esptool || command -v esptool.py || ls -d "${ARDUINO_DATA_DIR:-$HOME/.arduino15}"/packages/esp32/tools/esptool_py/*/esptool 2>/dev/null | sort -V | tail -1)
[ -n "$ESPTOOL" ] || { echo "esptool not found: pip install esptool"; exit 1; }
# the pins go into a header, not a compiler flag: a changed flag alone does not make arduino-cli rebuild the sketch
PINS=firmware/maxwell/generated/sht_pins.h
if [ -n "${SHT:-}" ]; then
  printf '#define SHT_SDA %s\n#define SHT_SCL %s\n' "${SHT%,*}" "${SHT#*,}" > "$PINS"
  echo "thermometer on its own pins: SDA ${SHT%,*}, SCL ${SHT#*,}"
else
  rm -f "$PINS"
fi
echo "=== build"
arduino-cli compile --fqbn "$FQBN" --build-path build firmware/maxwell
[ "${BUILD_ONLY:-0}" = 1 ] && exit 0
echo "=== model -> $PORT (partition 'model' at 0x110000)"
"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 write_flash 0x110000 model/model.bin
echo "=== firmware -> $PORT"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" --input-dir build firmware/maxwell
echo "done. They are talking now: open the serial monitor at 115200, or just watch the screen."
