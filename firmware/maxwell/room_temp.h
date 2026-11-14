// room_temp.h -- the room sets the model's temperature, and the BOOT button lets Maxwell's demon in.
//
// An SHT40 on the display's I2C bus reads the air; every word is sampled at room_to_T(degrees) (runtime/boltzmann.h).
// A cold room makes the model stiff (it loops), a warm hand on the sensor makes it ramble. With no sensor answering,
// it stays at 22 C. The chip writes paragraph after paragraph on its own; a typed line starts the next one.
// The top line of the screen always shows the reading, T and the phase, or DEMON while the demon is at the door.
#ifndef ROOM_TEMP_H
#define ROOM_TEMP_H

#include <Wire.h>
#include "../../runtime/boltzmann.h"

#ifndef PIN_DEMON
#define PIN_DEMON 0                    // the BOOT button on the dev board: press to let the demon in or out
#endif

static float room_c = 22.f;            // last reading, degrees C
static bool room_seen = false;         // has the SHT40 ever answered
static uint8_t room_addr = 0x44;       // SHT40-AD1B answers at 0x44, SHT40-BD1B at 0x45
static unsigned long room_at = 0;
static bool room_demon = false;

static bool sht_try(uint8_t addr) {
  Wire.beginTransmission(addr); Wire.write(0xFD);
  if (Wire.endTransmission() != 0) return false;
  delay(10);
  if (Wire.requestFrom(addr, (uint8_t)6) != 6) return false;
  uint16_t t = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read(); Wire.read(); Wire.read();
  room_c = -45.f + 175.f * t / 65535.f;
  return true;
}
static void room_read() {
  if (room_at && millis() - room_at < 500) return;
  room_at = millis();
  if (sht_try(room_addr)) { room_seen = true; return; }
  if (!room_seen) {                    // not found yet: try the other address
    uint8_t other = room_addr == 0x44 ? 0x45 : 0x44;
    if (sht_try(other)) { room_addr = other; room_seen = true; }
  }
}
static float room_softmax_temp() { room_read(); return room_to_T(room_c); }

// the BOOT button toggles the demon; called between words
static void room_poll_demon() {
  static bool was = false, ready = false;
  if (!ready) { pinMode(PIN_DEMON, INPUT_PULLUP); ready = true; }
  bool down = digitalRead(PIN_DEMON) == LOW;
  if (down && !was) room_demon = !room_demon;
  was = down;
}

// the top line of the screen; display.h calls this whenever it clears the page
static void room_header() {
  char head[24];
  float T = room_softmax_temp();
  snprintf(head, sizeof head, "%.1fC T=%.2f %s", room_c, T, room_demon ? "DEMON" : room_phase(T));
  oled.setCursor(0, 0);
  oled.print(head);
  oled.drawFastHLine(0, CH + 1, SCR_W, OLED_WHITE);
  ox = 0; oy = CH + 4;
}

static int room_next = 0;
static const char *room_start() { return ROOM_STARTS[room_next++ % ROOM_STARTS_N]; }

#endif  // ROOM_TEMP_H
