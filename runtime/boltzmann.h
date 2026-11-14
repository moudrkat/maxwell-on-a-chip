// boltzmann.h -- how the chip picks each word. Plain C: the sketch and the host program (host/maxwell.c) share it.
//
//   p_i = exp(z_i / T) / sum_j exp(z_j / T)          Boltzmann's distribution; a language model calls it softmax
//
// T comes from the room (room_softmax_temp). Maxwell's demon, when let in, opens the door only to the DEMON_K most
// probable words: it lowers the entropy without cooling anything, which is exactly what Maxwell invented it to do.
#ifndef BOLTZMANN_H
#define BOLTZMANN_H

#include <math.h>

#ifndef ROOM_T_NORMAL
#define ROOM_T_NORMAL 0.7f             // the room the chip woke up in is "normal": liquid Maxwell
#endif
#ifndef ROOM_T_PER_DEGREE
#define ROOM_T_PER_DEGREE 0.2f         // 3 C colder -> 0.1 (solid), 6 C warmer -> 1.9 (gas)
#endif
#ifndef ROOM_T_MIN
#define ROOM_T_MIN 0.05f
#endif
#ifndef ROOM_T_MAX
#define ROOM_T_MAX 2.0f
#endif
#ifndef DEMON_K
#define DEMON_K 4
#endif

// degrees Celsius -> sampling temperature, relative to the room at power-up (normal), so it works in any flat:
// what matters is warmer or colder than where they woke up
static float room_to_T(float celsius, float normal) {
  float T = ROOM_T_NORMAL + (celsius - normal) * ROOM_T_PER_DEGREE;
  return T < ROOM_T_MIN ? ROOM_T_MIN : T > ROOM_T_MAX ? ROOM_T_MAX : T;
}

// the phase the text is in at temperature T: frozen stiff, liquid Maxwell, a gas of two languages
static const char *room_phase(float T) { return T < 0.35f ? "SOLID" : T < 1.1f ? "LIQUID" : "GAS"; }

// One word: sample from softmax(z / T) with u uniform in [0, 1). k > 0 lets only the k largest logits through.
// *bits gets the entropy of the distribution sampled from: H = log Z - <x>, in bits.
static int boltzmann_sample(const float *z, int V, float T, int k, float u, float *bits) {
  float zmax = -1e30f, thr = -1e30f;
  for (int v = 0; v < V; v++) if (z[v] > zmax) zmax = z[v];
  if (k > 0 && k < V) {                       // the k-th largest logit: k short passes, V is small
    float prev = 1e30f;
    for (int j = 0; j < k; j++) {
      float m = -1e30f;
      for (int v = 0; v < V; v++) if (z[v] < prev && z[v] > m) m = z[v];
      prev = m;
    }
    thr = prev;
  }
  float sum = 0.f, sumx = 0.f;
  for (int v = 0; v < V; v++) if (z[v] >= thr) {
    float x = (z[v] - zmax) / T, e = expf(x);
    sum += e; sumx += e * x;
  }
  if (bits) *bits = (logf(sum) - sumx / sum) / 0.6931472f;
  float r = u * sum, acc = 0.f;
  int pick = -1;
  for (int v = 0; v < V; v++) if (z[v] >= thr) {
    acc += expf((z[v] - zmax) / T);
    pick = v;
    if (acc >= r) break;
  }
  return pick;
}

// the first words of each paragraph the chip writes on its own; the rest is the model
static const char *ROOM_STARTS[] = {
  "The temperature of a body is", "Die Temperatur eines Gases ist", "When a gas is compressed",
  "Die Molekuele bewegen sich", "It is evident that", "Man denke sich", "The molecules of a gas",
  "We shall now consider", "Wir wollen nun", "Heat is",
};
#define ROOM_STARTS_N ((int)(sizeof ROOM_STARTS / sizeof *ROOM_STARTS))

#endif  // BOLTZMANN_H
