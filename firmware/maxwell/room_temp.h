// room_temp.h -- the room sets the model's temperature (ROOM_TEMPERATURE, deploy.sh ROOM=1).
//
// An SHT40 on the display's I2C bus reads the air; every token is sampled with softmax(logits / T), where
// T = ROOM_T_AT_18 + (degrees - 18) * ROOM_T_PER_DEGREE, kept within [ROOM_T_MIN, ROOM_T_MAX]. A cold room makes the
// model stiff (it loops), a warm hand on the sensor makes it ramble. With no sensor answering, T stays at the 22 C value.
// The chip then writes paragraph after paragraph on its own, each one freshly sampled; a typed line starts the next one.
#ifndef ROOM_TEMP_H
#define ROOM_TEMP_H

#include <Wire.h>

#ifndef ROOM_T_AT_18
#define ROOM_T_AT_18 0.1f
#endif
#ifndef ROOM_T_PER_DEGREE
#define ROOM_T_PER_DEGREE 0.15f        // 22 C -> 0.7, 26 C -> 1.3, 30 C -> 1.9
#endif
#ifndef ROOM_T_MIN
#define ROOM_T_MIN 0.05f
#endif
#ifndef ROOM_T_MAX
#define ROOM_T_MAX 2.0f
#endif

static float room_c = 22.f;            // last reading, degrees C
static bool room_seen = false;         // has the SHT40 ever answered
static unsigned long room_at = 0;

static void room_read() {
  if (room_at && millis() - room_at < 500) return;
  room_at = millis();
  Wire.beginTransmission(0x44); Wire.write(0xFD);
  if (Wire.endTransmission() != 0) return;
  delay(10);
  if (Wire.requestFrom(0x44, 6) != 6) return;
  uint16_t t = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read(); Wire.read(); Wire.read();
  room_c = -45.f + 175.f * t / 65535.f;
  room_seen = true;
}

static float room_softmax_temp() {
  room_read();
  float T = ROOM_T_AT_18 + (room_c - 18.f) * ROOM_T_PER_DEGREE;
  return T < ROOM_T_MIN ? ROOM_T_MIN : T > ROOM_T_MAX ? ROOM_T_MAX : T;
}

// the phase the text is in at temperature T: frozen in loops, liquid Maxwell, a gas of two languages
static const char *room_phase(float T) { return T < 0.35f ? "SOLID" : T < 1.1f ? "LIQUID" : "GAS"; }

// the first words of each paragraph the chip writes on its own; the rest is the model
static const char *ROOM_STARTS[] = {
  "The temperature of a body is", "Die Temperatur eines Gases ist", "When a gas is compressed",
  "Die Molekuele bewegen sich", "It is evident that", "Man denke sich", "The molecules of a gas",
  "We shall now consider", "Wir wollen nun", "Heat is",
};
static int room_next = 0;
static const char *room_start() {
  const char *s = ROOM_STARTS[room_next % (int)(sizeof ROOM_STARTS / sizeof *ROOM_STARTS)];
  room_next++;
  return s;
}

#endif  // ROOM_TEMP_H
