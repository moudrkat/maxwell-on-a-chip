#!/bin/sh
# Build the firmware and put Maxwell and Boltzmann on the board: the model into its flash partition, then the program.
#   ./flash.sh [port]            (default /dev/ttyACM0)
#   SHT=4,5 ./flash.sh           the thermometer on its own two pins (SDA,SCL) instead of sharing the display's
# Needs arduino-cli with the esp32 core and the Adafruit SSD1306 + GFX libraries:
#   arduino-cli core install esp32:esp32
#   arduino-cli lib install "Adafruit SSD1306" "Adafruit GFX Library"
set -e
cd "$(dirname "$0")"
PORT=${1:-/dev/ttyACM0}
FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi'
ESPTOOL=$(command -v esptool || command -v esptool.py || ls -d "${ARDUINO_DATA_DIR:-$HOME/.arduino15}"/packages/esp32/tools/esptool_py/*/esptool 2>/dev/null | sort -V | tail -1)
[ -n "$ESPTOOL" ] || { echo "esptool not found: pip install esptool"; exit 1; }
EXTRA=
[ -n "${SHT:-}" ] && EXTRA="-DSHT_SDA=${SHT%,*} -DSHT_SCL=${SHT#*,}" && echo "thermometer on its own pins: SDA ${SHT%,*}, SCL ${SHT#*,}"
echo "=== build"
arduino-cli compile --fqbn "$FQBN" --build-property "compiler.cpp.extra_flags=$EXTRA" --build-path build firmware/maxwell
[ "${BUILD_ONLY:-0}" = 1 ] && exit 0
echo "=== model -> $PORT (partition 'model' at 0x110000)"
"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 write_flash 0x110000 model/model.bin
echo "=== firmware -> $PORT"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" --input-dir build firmware/maxwell
echo "done. They are talking now: open the serial monitor at 115200, or just watch the screen."
