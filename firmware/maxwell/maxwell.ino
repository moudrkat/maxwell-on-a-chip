// Maxwell on a chip: esp32-ai's TinyLM sketch (Viacheslav Sierbov's runtime, MIT; see THIRD_PARTY.md), with
// ROOM_TEMPERATURE: an SHT40 sets the sampling temperature, the chip writes on its own, the screen shows the phase
// of the text and the entropy that came out.
// PLE TinyLM inference on the ESP32-S3.
//
// The 28.9M-param model (14.9MB, 4-bit) lives in a flash 'model' partition,
// memory-mapped. Placement follows reads-per-token rather than what happens to
// fit:
//
//   flash   PLE table + token embedding   one row of the 25.2M-parameter
//                                         table per token
//   PSRAM   staged int8 core + head, KV   read once per position
//   SRAM    scratch + norm vectors        touched many times per token
//
// The logits array stays in PSRAM: 25,353 floats is 99 KiB, and the argmax
// reads it once per token.
//
// Same llm.h that is verified against PyTorch on the host; only the platform
// hooks differ here.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

// int8 activations, required by the staged int8 kernel. Not bit-exact against
// the fp32 golden; verify.c must be built without this flag. Validation CE cost
// (runtime/host_verify/ppl.c, 32,768 predictions): 2.4793 -> 2.4796, ppl 11.93 / 11.94.
#define LLM_INT8_ACT 1
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../../runtime/llm.h"
#include "../../runtime/bpe_tokenizer.h"
#include "generated/vocab.h"
#include "generated/tokenizer_encoder.h"
// deploy.sh writes this per model: how a typed line is wrapped before it is
// encoded (the oled model wants "# <request>\n") and the idle text.
#if __has_include("generated/prompt_format.h")
#include "generated/prompt_format.h"
#endif
#ifndef PROMPT_PREFIX
#define PROMPT_PREFIX ""
#endif
#ifndef PROMPT_SUFFIX
#define PROMPT_SUFFIX ""
#endif
#ifndef IDLE_TEXT
#define IDLE_TEXT "ESP32-S3 PLE TinyLM\n28.9M params\n\nwaiting for a\nstory prompt..."
#endif

// Set to 1 once a display is wired up - see display.h.
// Leave 0 to run serial-only (no panel needed).
#define USE_DISPLAY 1
#if USE_DISPLAY
#include "display.h"
#endif
#if defined(ROOM_TEMPERATURE)
#include "room_temp.h"          // the room sets the temperature
#endif

// OLED_PROGRAM (set by deploy.sh for the oled model): the model writes a small program for this screen, in the
// language of runtime/oled_lang.h, and the board runs it. The typed request becomes "# <request>\n", the code
// appears on the panel as it is written, and when the model ends the document the panel runs it.
#if defined(OLED_PROGRAM)
#if !USE_DISPLAY
#error "OLED_PROGRAM needs the display"
#endif
#include "../../runtime/oled_lang.h"
static ol_machine *olm;                 // ~45 KB, in PSRAM
static char program[OL_MAX_SRC];
static int program_len = 0;
static bool collecting = false;         // true after the prompt: only the model's own tokens are the program
#endif

static const int PROMPT_IDS[] = {433, 447, 259, 405}; // "Once upon a time"
#if defined(OLED_PROGRAM)
static const int N_GENERATE = 240;   // a program, up to the context
#else
static const int N_GENERATE = 200;
#endif
static BpeTokenizer tokenizer;   // encodes serial prompts on the device itself
// The end-of-text id, found in the decode table at boot (-1 when absent). A
// story or an answer stops there instead of running into the next document.
static int eot_id = -1;

// --- serial prompt intake ---------------------------------------------------
// Polled between decode steps as well as when idle, so the CDC FIFO is drained
// continuously and a finished line parks in pending_line until loop() takes it.
static char in_buf[256];
static size_t in_len = 0;
static char pending_line[256];
static bool pending_ready = false;

static bool poll_serial_line() {
  if (pending_ready) return true;
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      in_buf[in_len] = 0;
      size_t had = in_len;
      in_len = 0;
      if (had) {
        memcpy(pending_line, in_buf, had + 1);
        pending_ready = true;
        return true;
      }
    } else if (in_len < sizeof(in_buf) - 1) {
      in_buf[in_len++] = ch;
    }
  }
  return false;
}

Model model;
Scratch s;

// ---- allocation ------------------------------------------------------------
// Allocations are strict because memory placement is part of the runtime
// configuration. Allocation failure stops initialization.
static size_t psram_used = 0, sram_used = 0;

// The two LLM_Q8_MAX_INPUT int8 activation buffers (matvec_q8, matvec_par),
// which are static and so absent from the totals above. Together these report
// the managed hot set, not total SRAM usage; the free-SRAM figure printed at
// boot is the overall diagnostic.
#define STATIC_SRAM_BYTES (2 * LLM_Q8_MAX_INPUT)

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (p) psram_used += n;
  return p;
}
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) {
    Serial.printf("FATAL: required PSRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  return p;
}
static void *sram_or_die(size_t n, const char *what) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) {
    Serial.printf("FATAL: required SRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  sram_used += n;
  return p;
}

// ---- dual-core int8 matvec -------------------------------------------------
// Serves both hooks. Below ~128 rows the task notify round trip costs more than
// the split saves, so small tensors run single-core.
static TaskHandle_t worker_h, main_h;
static const QT *job_t;
static const int8_t *job_xq;
static float job_xs;
static float *job_y;
static int job_split;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}

static void matvec_par(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

// Copy RMSNorm weights from mapped flash to internal SRAM.
static void copy_norms_to_sram() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim;
  const float **vecs[3 * 32 + 2];
  int sizes[3 * 32 + 2], n_vec = 0;
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = sram_or_die(bytes, "norm vector");
    memcpy(dst, *vecs[i], bytes);
    *vecs[i] = (const float *)dst;
  }
  Serial.printf("norms  -> SRAM   %d vectors\n", n_vec);
}

static void alloc_scratch() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, S = c->seq_len;
  // hot working set -> internal SRAM
  s.x     = (float *)sram_or_die(D * 4, "x");
  s.h     = (float *)sram_or_die((F > D ? F : D) * 4, "h");
  s.qkv   = (float *)sram_or_die(3 * D * 4, "qkv");
  s.att   = (float *)sram_or_die(D * 4, "att");
  s.g1    = (float *)sram_or_die(F * 4, "g1");
  s.g2    = (float *)sram_or_die((P > F ? P : F) * 4, "g2");
  s.ple   = (float *)sram_or_die(L * P * 4, "ple");
  s.tmpP  = (float *)sram_or_die(L * P * 4, "tmpP");
  s.trow  = (float *)sram_or_die(L * P * 4, "trow");
  s.scores = (float *)sram_or_die(S * 4, "scores");
  // logits: out_vocab floats, 99 KiB here, read once per token. Left in PSRAM
  // rather than spend a fifth of internal SRAM on it.
  s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
  // KV cache: 1.1MB, read once per position rather than per matvec.
  s.kcache = (float *)ps_or_die((size_t)L * S * D * 4, "kcache");
  s.vcache = (float *)ps_or_die((size_t)L * S * D * 4, "vcache");
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

// Emit one token to every active output (serial always; panel when enabled).
static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  // Non-blocking: when no host is draining the USB-CDC buffer (running as a
  // standalone gadget on the display), skip the write instead of stalling the
  // whole generation once the TX buffer fills.
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);
#if USE_DISPLAY
#if defined(OLED_PROGRAM)
  if (collecting)                       // the request is drawn as a header instead, see request_header()
#endif
  display_puts(bytes, len);
#endif
#if defined(OLED_PROGRAM)
  if (collecting && program_len + len < OL_MAX_SRC) { memcpy(program + program_len, bytes, len); program_len += len; }
#endif
}

#if defined(OLED_PROGRAM)
// Drawing for the interpreter: Adafruit GFX on the panel, white on black, 6x8 text cells.
static void g_clear(void *) { oled.clearDisplay(); }
static void g_text(void *, int x, int y, const char *s) { oled.setTextSize(1); oled.setCursor(x, y); oled.print(s); }
static void g_pixel(void *, int x, int y) { oled.drawPixel(x, y, OLED_WHITE); }
static void g_line(void *, int x0, int y0, int x1, int y1) { oled.drawLine(x0, y0, x1, y1, OLED_WHITE); }
static void g_rect(void *, int x, int y, int w, int h) { if (w > 0 && h > 0) oled.drawRect(x, y, w, h, OLED_WHITE); }
static void g_fill(void *, int x, int y, int w, int h) { if (w > 0 && h > 0) oled.fillRect(x, y, w, h, OLED_WHITE); }
static void g_circle(void *, int x, int y, int r) { if (r >= 0) oled.drawCircle(x, y, r, OLED_WHITE); }
static bool poll_serial_line();
static void g_wait(void *, int ms) {
  oled.display();
  // pause in short slices so a new request interrupts a long animation within ~20 ms
  for (int left = ms; left > 0; left -= 20) {
    delay(left < 20 ? left : 20);
    if (poll_serial_line()) { olm->stop = 1; return; }
  }
}
static void g_done(void *) { oled.display(); }

// What you typed, black on a white bar at the top, so the request and what the board writes below it never
// look alike. Up to two lines of 21 characters.
#ifndef OLED_BLACK
#define OLED_BLACK 0                    // black is 0 on both SSD1306 and SH1106
#endif
static char oled_request[64];
static void request_header() {
  int n = strlen(oled_request), lines = n > 21 ? 2 : 1;
  oled.fillRect(0, 0, SCR_W, lines * CH + 2, OLED_WHITE);
  oled.setTextSize(1);
  oled.setTextColor(OLED_BLACK);
  for (int i = 0; i < lines; i++) {
    char row[22];
    snprintf(row, sizeof(row), "%.21s", oled_request + 21 * i);
    oled.setCursor(1, 1 + i * CH);
    oled.print(row);
  }
  oled.setTextColor(OLED_WHITE);
  ox = 0; oy = lines * CH + 4;
  oled.display();
}

static void run_program() {
  program[program_len] = 0;
  ol_io io = {nullptr, g_clear, g_text, g_pixel, g_line, g_rect, g_fill, g_circle, g_wait, g_done};
  delay(2000);                             // the written code stays long enough to be read before it runs
  oled.clearDisplay();
  int64_t t0 = esp_timer_get_time();
  int rc = ol_parse(olm, program);
  if (rc == 0) rc = ol_run(olm, &io);
  else oled.display();
  if (rc != 0) {
    Serial.printf("\n[program refused: %s]\n", olm->err);
    if (strcmp(olm->err, "interrupted")) {  // a refused program says why, in place of the picture
      display_home();
      const char head[] = "can't run this:\n";
      display_puts((const unsigned char *)head, sizeof(head) - 1);
      display_puts((const unsigned char *)olm->err, strlen(olm->err));
      oled.display();
    }
  } else {
    Serial.printf("\n[program ran: %ld steps, %.1f s]\n", olm->steps, (esp_timer_get_time() - t0) / 1e6);
  }
}
#endif

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 PLE TinyLM ===");

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, model.out_vocab, c->dim, c->n_layers, c->n_heads,
                c->ffn, c->ple_dim, part->size / 1e6);

#if USE_DISPLAY
  display_begin();
#if defined(ROOM_TEMPERATURE)
  display_bottom = ROOM_BARS_TOP - 3;   // the rows below belong to the distribution
#endif
#endif

  // The model header states how many logits it produces; vocab.h carries the
  // decode table. If they disagree, every emitted token would be decoded
  // against the wrong table.
  if (VOCAB_N != model.out_vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: vocab.h %d, model %d\n",
                  VOCAB_N, model.out_vocab);
    return;
  }

  alloc_scratch();
  copy_norms_to_sram();
  Serial.printf("hot set-> SRAM   %u B dynamic + %u B static = %u B managed\n",
                (unsigned)sram_used, (unsigned)STATIC_SRAM_BYTES,
                (unsigned)(sram_used + STATIC_SRAM_BYTES));

  // Stage every per-position tensor to int8 in PSRAM.
  int want = llm_core_stage_count(&model);
  int staged = llm_stage_core_int8_alloc(&model, ps);
  if (staged != want) {
    Serial.printf("FATAL: staged %d/%d core tensors\n", staged, want);
    while (1) delay(1000);
  }
  // The tied head is read per token, not per position, so the core helper does
  // not walk it. Stage it too: it is 85%% of the dense MACs.
  {
    void *b = ps_or_die(llm_stage_int8_bytes(&model.out_head), "staged head");
    llm_stage_int8(&model.out_head, b);
    ++staged;
  }
  Serial.printf("weights-> PSRAM  %d tensors int8, %.2f MB allocated\n",
                staged, psram_used / 1048576.0);

  main_h = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(worker_main, "mv", 4096, NULL, 2, &worker_h, 0) == pdPASS) {
    // After the worker exists: matvec_par notifies worker_h.
    model.layer_matvec = matvec_par;
    model.head_matvec  = matvec_par;
  } else {
    Serial.println("dual-core worker failed; running single core");
  }

  // FNV-1a over the mapped image. scripts/deploy.sh prints the same value for
  // the file it flashed; the two must agree.
  {
    const uint8_t *img = (const uint8_t *)base;
    uint32_t fp = 2166136261u;
    for (size_t i = 0; i < model.image_bytes; i++) { fp ^= img[i]; fp *= 16777619u; }
    Serial.printf("build: bytes=%u fp=%08x sram=%uB psram=%.2fMB\n",
                  (unsigned)model.image_bytes, fp,
                  (unsigned)(sram_used + STATIC_SRAM_BYTES),
                  psram_used / 1048576.0);
  }
  Serial.printf("free: sram %.0f KB | psram %.2f MB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);

#if defined(OLED_PROGRAM)
  olm = (ol_machine *)ps_or_die(sizeof(ol_machine), "oled interpreter");
#endif
  if (bpe_tokenizer_load(TOKENIZER_ENCODER_ASSET, TOKENIZER_ENCODER_ASSET_SIZE,
                         &tokenizer)) {
    Serial.println("tokenizer load failed; serial prompts disabled");
  }

  // Quiet start: a title card, then wait for a prompt over serial. The box
  // speaks only when spoken to.
#if USE_DISPLAY
  {
    const char idle[] = IDLE_TEXT;
    display_puts((const unsigned char *)idle, (int)sizeof(idle) - 1);
  }
#endif
  {
    static const char eot[] = "<|endoftext|>";
    const int n = (int)sizeof(eot) - 1;
    for (int v = 0; v < VOCAB_N; v++)
      if (VOCAB_OFF[v + 1] - VOCAB_OFF[v] == n &&
          memcmp(VOCAB_BLOB + VOCAB_OFF[v], eot, (size_t)n) == 0) { eot_id = v; break; }
    Serial.printf("eot id: %d\n", eot_id);
  }
#if defined(OLED_PROGRAM)
  Serial.println("ready; type what to draw, then newline");
#else
  Serial.println("ready; send a story opening followed by newline");
#endif
}

// Tell one story: prime the prompt, decode until N_GENERATE or the context
// fills. temp=0 decodes greedily - deterministic, and therefore in lockstep
// with the verified brainscope twin fed the same prompt; temp>0 samples the
// softmax at that temperature, so the standalone box keeps telling a fresh
// story every round instead of the same one forever.
static void run_story(const int *prompt_ids, int n_prompt, float temp) {
  Serial.print(">>> ");
#if USE_DISPLAY
  display_home();
#endif
#if defined(OLED_PROGRAM)
  request_header();
#endif
  int pos = 0, tok = 0;
  int64_t decode_us = 0;
  int decoded = 0;

  for (int i = 0; i < n_prompt; i++) {  // prime with the prompt
    tok = prompt_ids[i];
    emit(tok);
    llm_forward(&model, tok, pos++, &s);
  }

  llm_profile_reset(&s);
#if defined(OLED_PROGRAM)
  program_len = 0;
  collecting = true;
#endif

  int64_t t_start = esp_timer_get_time();
#if defined(ROOM_TEMPERATURE)
  Serial.printf("[%s, %.1f C%s] ", room_seen ? "SHT40" : "no SHT40, 22 C assumed", room_c, room_demon ? ", demon" : "");
  double entropy_bits = 0; int entropy_n = 0;
#endif
  for (int step = 0; step < N_GENERATE && pos < model.c.seq_len; step++) {
#if defined(ROOM_TEMPERATURE)
    temp = room_softmax_temp();         // every word at the temperature of the room right now
#endif
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < model.out_vocab; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
#if defined(ROOM_TEMPERATURE)
    {                                   // Boltzmann's distribution at the room's temperature (runtime/boltzmann.h)
      float bits;
      room_poll_demon();
      best = boltzmann_sample(s.logits, model.out_vocab, temp, room_demon ? DEMON_K : 0,
                              esp_random() / 4294967296.0f, &bits);
      entropy_bits += bits; entropy_n++;
#if USE_DISPLAY
      room_rank = boltzmann_top(s.logits, model.out_vocab, temp, room_demon ? DEMON_K : 0, best, room_p, ROOM_BARS);
      display_hold = true;              // the word and its distribution go to the screen in one frame
#endif
    }
#else
    if (temp > 0.0f) {
      // Softmax sample at `temp`: two extra passes over the logits (~10 ms)
      // against ~95 ms/token of compute. The argmax above doubles as the
      // numeric stabilizer.
      float sum = 0.f;
      for (int v = 0; v < model.out_vocab; v++)
        sum += expf((s.logits[v] - bv) / temp);
      float r = (esp_random() / 4294967296.0f) * sum;
      float acc = 0.f;
      int pick = model.out_vocab - 1;
      for (int v = 0; v < model.out_vocab; v++) {
        acc += expf((s.logits[v] - bv) / temp);
        if (acc >= r) { pick = v; break; }
      }
      best = pick;
    }
#endif
    tok = best;
    if (tok == eot_id) break;       // the document ended; do not start another
    emit(tok);
#if defined(ROOM_TEMPERATURE) && USE_DISPLAY
    room_bars(room_p, room_rank);       // under the word, the distribution it was drawn from
    display_hold = false;
    oled.display();
#endif
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);  // feed the task WDT ~every 8 tokens
    if (poll_serial_line()) {       // a new prompt interrupts this story
      Serial.println("\n(interrupted by a new prompt)");
      blink(0);
#if defined(OLED_PROGRAM)
      collecting = false;
#endif
      return;
    }
  }
  int64_t total_us = esp_timer_get_time() - t_start;
#if USE_DISPLAY
  display_hold = false;
#endif

  Serial.printf("\n\n--- %d tokens in %.2f s ---\n", decoded, total_us / 1e6);
  Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
#if defined(OLED_PROGRAM)
  collecting = false;
  blink(0);
  run_program();                    // the picture stays until the next request
  return;
#endif
#if USE_DISPLAY
#if defined(ROOM_TEMPERATURE)
  {                                     // temperature in, entropy out
    char tail[40];
    int n = snprintf(tail, sizeof tail, "\nS=%.1f bits/word", entropy_n ? entropy_bits / entropy_n : 0.0);
    Serial.printf("%s\n", tail);
#if USE_DISPLAY
    display_puts((const unsigned char *)tail, n);
#endif
  }
#else
  display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
#endif
  blink(0);
}

// A line of ASCII arriving over serial becomes the next story's opening,
// generated greedily ON THIS CHIP - and greedy plus the verified twin means
// brainscope fed the same prompt shows the internals of the very story the
// panel is writing. Room is reserved for the answer: a long prompt cannot
// eat the whole context.
static void prompt_story(const char *text) {
#if defined(OLED_PROGRAM)
  snprintf(oled_request, sizeof(oled_request), "%s", text);
#endif
  char wrapped[320];
  snprintf(wrapped, sizeof(wrapped), "%s%s%s", PROMPT_PREFIX, text, PROMPT_SUFFIX);
  uint16_t enc[96];
  int n = bpe_encode_ascii(&tokenizer, wrapped, enc, (int)(sizeof(enc) / sizeof(enc[0])));
  if (n <= 0) { Serial.println("(ascii only, or prompt too long)"); return; }
  int ids[96];
  for (int i = 0; i < n; i++) ids[i] = enc[i];
  run_story(ids, n, 0.0f);
}

#if __has_include("generated/one_word.h")
#include "generated/one_word.h"         // flash.sh writes it for WORD="..."
#endif
#if defined(ROOM_TEMPERATURE) && USE_DISPLAY
// One word, held still. A typed line that starts with "?" ("?The temperature of a body is") is not continued: the
// screen shows the distribution of the next word only, the way a softmax is always drawn: the 16 most probable words
// along x, probability up y (the full height is 50%, marked; a taller bar is cut there and its number still says how
// much), redrawn a few times a second at the temperature the thermometer reads right now. Cold, one bar towers; in a
// warm hand they sink into a flat row. The three likeliest words are
// named, the tick under the axis marks the word just drawn, and the last line reads the sentence with it.
// Any other typed line ends it.
static void word_label(int tok, char *out, int n) {     // the token's text without its leading space
  const unsigned char *b = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok], k = 0;
  for (int i = 0; i < len && k < n - 1; i++) {
    char c = (char)b[i];
    if (k == 0 && c == ' ') continue;
    if (c >= 33 && c < 127) out[k++] = c;
  }
  out[k] = 0;
}
static void one_word(const char *text) {
  uint16_t enc[96];
  int n = bpe_encode_ascii(&tokenizer, text, enc, (int)(sizeof(enc) / sizeof(enc[0])));
  if (n <= 0) { Serial.println("(ascii only, or prompt too long)"); return; }
  for (int i = 0; i < n; i++) llm_forward(&model, enc[i], i, &s);
  const int V = model.out_vocab, NTOP = 16, NNAMED = 3;
  const int Y0 = CH + 4, YAXIS = 53, HMAX = YAXIS - Y0;   // the plot: rows 12..52, the axis on row 53
  int top[NTOP];
  for (int j = 0; j < NTOP; j++) {                       // the largest logits; the temperature never reorders them
    int b = -1;
    for (int v = 0; v < V; v++) {
      bool taken = false;
      for (int q = 0; q < j; q++) if (top[q] == v) taken = true;
      if (!taken && (b < 0 || s.logits[v] > s.logits[b])) b = v;
    }
    top[j] = b;
  }
  char label[NNAMED][13];
  for (int j = 0; j < NNAMED; j++) word_label(top[j], label[j], sizeof label[j]);
  const char *tail = text + strlen(text);                // the last words of the sentence that fit beside the drawn one
  while (tail > text && strlen(tail) < 9) { const char *p = tail - 1; while (p > text && p[-1] != ' ') p--; if (strlen(p) > 9) break; tail = p; }
  Serial.printf("\none word after \"%s\"\n", text);
  unsigned long said = 0;
  while (!poll_serial_line()) {
    float T = room_softmax_temp();
    float zmax = s.logits[top[0]], sum = 0.f, p[NTOP], rest = 1.f;
    for (int v = 0; v < V; v++) sum += expf((s.logits[v] - zmax) / T);
    for (int j = 0; j < NTOP; j++) { p[j] = expf((s.logits[top[j]] - zmax) / T) / sum; rest -= p[j]; }
    if (rest < 0.f) rest = 0.f;
    int pick = boltzmann_sample(s.logits, V, T, 0, esp_random() / 4294967296.0f, nullptr);
    char drawn[13]; word_label(pick, drawn, sizeof drawn);
    oled.clearDisplay();
    room_header();                                       // the reading, T and the phase
    char row[24];
    for (int j = 0; j < NTOP; j++) {                     // the bars, 6 wide, 2 apart
      int h = (int)(p[j] / 0.5f * HMAX + 0.5f);          // the full height is 50%
      if (h < 1) h = 1;
      if (h > HMAX) h = HMAX;
      oled.fillRect(j * 8, YAXIS - h, 6, h, OLED_WHITE);
    }
    oled.drawFastHLine(0, YAXIS, SCR_W, OLED_WHITE);
    for (int j = 0; j < NTOP; j++) if (top[j] == pick) oled.fillRect(j * 8 + 1, YAXIS + 1, 4, 2, OLED_WHITE);
    oled.setCursor(SCR_W - 3 * CW, Y0); oled.print("50%");  // what the top of the plot stands for
    for (int j = 0; j < NNAMED; j++) {                   // the three likeliest, named, in the empty corner of the plot
      int nrow = snprintf(row, sizeof row, "%s %d%%", label[j], (int)(p[j] * 100 + 0.5f));
      oled.setCursor(SCR_W - nrow * CW, Y0 + 2 + (j + 1) * CH); oled.print(row);
    }
    snprintf(row, sizeof row, "%s %s", tail, drawn);
    oled.setCursor(0, 56); oled.print(row);
    oled.display();
    if (millis() - said > 2000) {
      said = millis();
      Serial.printf("%.1f C T=%.2f:", room_c, T);   // the whole picture as text, for whoever is logging it
      for (int j = 0; j < NTOP; j++) { char w[13]; word_label(top[j], w, sizeof w); Serial.printf(" %s=%.3f", w, p[j]); }
      Serial.printf(" rest=%.3f -> %s\n", rest, drawn);
    }
    delay(400);
  }
}
#endif

// Standalone gadget mode with an ear: poll for a typed prompt continuously -
// including BETWEEN decode steps, so a prompt interrupts the current story
// within a token instead of sitting in the FIFO until it ends. Idle ~20 s
// with no prompt -> tell a fresh sampled story of our own.
void loop() {
  if (poll_serial_line()) {
    pending_ready = false;
#if defined(ROOM_TEMPERATURE) && USE_DISPLAY
    if (pending_line[0] == '?') one_word(pending_line + 1); else
#endif
    prompt_story(pending_line);
  }
#if defined(ROOM_WORD) && USE_DISPLAY
  static bool word_shown = false;       // flashed with WORD="...": start on that one word; a typed line ends it
  if (!word_shown) { word_shown = true; one_word(ROOM_WORD); return; }
#endif
#if defined(ROOM_TEMPERATURE)
  // nothing typed: the next paragraph, a few seconds after the last one ended
  static unsigned long room_resume = 0;
  if (!pending_ready && millis() > room_resume) {
    prompt_story(room_start());
    room_resume = millis() + 4000;
  }
#endif
  delay(20);
}
