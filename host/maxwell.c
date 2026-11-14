// No chip? The same model file, the same C runtime and the same sampling code (runtime/boltzmann.h) on a computer.
//
//   cc -O2 -o host/maxwell host/maxwell.c -lm
//   host/maxwell 22            # degrees Celsius: one paragraph per start at that room temperature
//   host/maxwell 30 demon      # ... with Maxwell's demon at the door
//   host/maxwell 18 "" 7       # another seed
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define PROGMEM
#include "../runtime/llm.h"
#include "../runtime/bpe_tokenizer.h"
#include "../runtime/boltzmann.h"
#include "../firmware/maxwell/generated/vocab.h"
#include "../firmware/maxwell/generated/tokenizer_encoder.h"

int main(int argc, char **argv) {
  float celsius = argc > 1 ? atof(argv[1]) : 22.f;
  int demon = argc > 2 && !strcmp(argv[2], "demon");
  srand(argc > 3 ? atoi(argv[3]) : 1);
  FILE *f = fopen("model/model.bin", "rb");
  if (!f) { perror("model/model.bin (run from the repo root)"); return 1; }
  fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *buf = malloc(n); if (fread(buf, 1, n, f) != n) return 1; fclose(f);
  Model m; if (llm_load(buf, &m)) { fprintf(stderr, "bad model file\n"); return 1; }
  int D = m.c.dim, L = m.c.n_layers, P = m.c.ple_dim, F = m.c.ffn, V = m.out_vocab, S = m.c.seq_len;
  Scratch s;
  s.x = malloc(D * 4); s.h = malloc((F > D ? F : D) * 4); s.qkv = malloc(3 * D * 4); s.att = malloc(D * 4);
  s.g1 = malloc(F * 4); s.g2 = malloc((P > F ? P : F) * 4);
  s.ple = malloc(L * P * 4); s.tmpP = malloc(L * P * 4); s.trow = malloc(L * P * 4);
  s.logits = malloc(V * 4); s.scores = malloc(S * 4);
  s.kcache = malloc((size_t)L * S * D * 4); s.vcache = malloc((size_t)L * S * D * 4);
  BpeTokenizer tok;
  if (bpe_tokenizer_load(TOKENIZER_ENCODER_ASSET, TOKENIZER_ENCODER_ASSET_SIZE, &tok)) { fprintf(stderr, "tokenizer\n"); return 1; }
  int eot = -1;
  for (int v = 0; v < VOCAB_N; v++)
    if (VOCAB_OFF[v + 1] - VOCAB_OFF[v] == 13 && !memcmp(VOCAB_BLOB + VOCAB_OFF[v], "<|endoftext|>", 13)) eot = v;
  float T = room_to_T(celsius);
  printf("%.1f C  T=%.2f  %s%s\n\n", celsius, T, room_phase(T), demon ? "  (demon)" : "");
  double all_bits = 0; int all_n = 0;
  for (int k = 0; k < ROOM_STARTS_N; k++) {
    uint16_t enc[96];
    int np = bpe_encode_ascii(&tok, ROOM_STARTS[k], enc, 96);
    int pos = 0;
    for (int i = 0; i < np; i++) llm_forward(&m, enc[i], pos++, &s);
    printf("%s", ROOM_STARTS[k]);
    double bits_sum = 0; int words = 0;
    while (pos < S) {
      float bits;
      int t = boltzmann_sample(s.logits, V, T, demon ? DEMON_K : 0, rand() / (RAND_MAX + 1.0f), &bits);
      bits_sum += bits; words++;
      if (t == eot) break;
      fwrite(VOCAB_BLOB + VOCAB_OFF[t], 1, VOCAB_OFF[t + 1] - VOCAB_OFF[t], stdout);
      llm_forward(&m, t, pos++, &s);
    }
    printf("\n  S=%.1f bits/word\n\n", bits_sum / words);
    all_bits += bits_sum; all_n += words;
  }
  printf("mean entropy %.2f bits/word over %d words\n", all_bits / all_n, all_n);
  return 0;
}
