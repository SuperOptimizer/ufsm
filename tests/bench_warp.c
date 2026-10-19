/* the sampler's geometry-augmentation warps on synthetic P^3 windows (CPU only): bench_warp [P] [threads] [reps]
   per window: write_spatial_tm in the trainer's form (--gpu-input --pack-targets, hard targets, fp16 CT, the --ct-aug LUT plan; and
   without a plan), the stored phase (warp_phase), the winding mod 4 (warp_w4) and both in one pass (warp_phase_w4, what draw() runs
   for --task unwrap2), each with the reference loops (_ref) beside them. Every thread warps its own windows (rotation 10 degrees,
   elastic, zoom 0.7) from shared sources; ms per window = wall time / windows per thread. */
#include "../src/sample.c"
#ifdef BENCH_BASE   /* built against the sources before the vector warps (-DBENCH_BASE): the old loops under the new names */
#define warp_phase_ref warp_phase
#define warp_w4_ref warp_w4
static void warp_w4_pack(const uint8_t *w4, const uint8_t *cp, uint16_t *wq, size_t n) { for (size_t k = 0; k < n; k++) wq[k] = (uint16_t)(w4[k] | cp[k] << 8); }
static void warp_phase_w4(const uint8_t *ph, const uint16_t *wq, uint8_t *dph, uint8_t *dw, uint8_t *dc, int P, sym sy, const spatial_aug *a) {
    const size_t p3 = (size_t)P * P * P; uint8_t *w4 = malloc(p3), *cp = malloc(p3);
    for (size_t k = 0; k < p3; k++) { w4[k] = (uint8_t)wq[k]; cp[k] = (uint8_t)(wq[k] >> 8); }
    warp_w4(w4, cp, dw, dc, P, sy, a); warp_phase(ph, dph, P, sy, a); free(w4); free(cp);
}
#endif
#include <assert.h>
void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }

static int P, REPS;
static uint8_t *ct, *tgt, *msk, *ph, *w4, *wc; static uint16_t *wq;
static float cy[2048], cx[2048];
typedef struct { int id; double t[8]; } targ;
enum { OP_TM_LUT, OP_TM_NONE, OP_PH, OP_W4, OP_FUSED, OP_PH_REF, OP_W4_REF, OP_N };
static const char *names[OP_N] = {"write_spatial_tm (ct-aug LUT)", "write_spatial_tm (no plan)", "warp_phase", "warp_w4", "warp_phase_w4 (fused)", "warp_phase_ref", "warp_w4_ref"};
static int ops_on[OP_N];
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *run(void *arg) {
    targ *a = arg; const size_t p3 = (size_t)P * P * P;
    uint8_t *TM = malloc(p3), *o1 = malloc(p3), *o2 = malloc(p3), *o3 = malloc(p3); uint16_t *H = malloc(p3 * 2);
    float *app = malloc(p3 * 4), *nrow = malloc((size_t)P * 4);
    const int64_t org[3] = {0, 0, 0};
    for (int rep = 0; rep < REPS; rep++) {
        spatial_aug s; spatial_aug_make(&s, 1000 + 77 * a->id + rep, P, 10, 1, 1, 1); spatial_aug_zoom(&s, .7);
        sym sy = sym_of((a->id * 7 + rep) % 48);
        for (int op = 0; op < OP_N; op++) {
            if (!ops_on[op]) continue;
            rng r; rseed(&r, rep);
            ct_aug_plan ap; ct_aug_make(&ap, 5, 1, 90, 30, P); ap.filter_mode = 0; ap.noise_rho = 0;
            const double t0 = now();
            switch (op) {
            case OP_TM_LUT: case OP_TM_NONE:
                write_spatial_tm(ct, tgt, msk, nullptr, nullptr, P, org, 90, 1.f / 30, 1, cy, cx, sy, &s, 1.05f, .01f, .03f, &r, nrow, 1, nullptr, H,
                                 op == OP_TM_LUT ? &ap : nullptr, nullptr, nullptr, nullptr, nullptr, app, 1, 1, TM); break;
            case OP_PH: warp_phase(ph, o1, P, sy, &s); break;
            case OP_W4: warp_w4(w4, wc, o2, o3, P, sy, &s); break;
            case OP_FUSED: warp_phase_w4(ph, wq, o1, o2, o3, P, sy, &s); break;
            case OP_PH_REF: warp_phase_ref(ph, o1, P, sy, &s); break;
            case OP_W4_REF: warp_w4_ref(w4, wc, o2, o3, P, sy, &s); break;
            }
            a->t[op] += now() - t0;
        }
    }
    free(TM); free(o1); free(o2); free(o3); free(H); free(app); free(nrow); tp_drain();
    return nullptr;
}
int main(int argc, char **argv) {
    P = argc > 1 ? atoi(argv[1]) : 256; const int NT = argc > 2 ? atoi(argv[2]) : 1; REPS = argc > 3 ? atoi(argv[3]) : 2;
    const char *sel = getenv("BENCH_OPS");   /* comma list of op numbers (default: all but the references) */
    for (int k = 0; k < OP_N; k++) ops_on[k] = sel ? 0 : k < OP_PH_REF;
    if (sel) for (const char *q = sel; *q; q++) if (*q >= '0' && *q < '0' + OP_N) ops_on[*q - '0'] = 1;
    const size_t p3 = (size_t)P * P * P;
    ct = malloc(p3); tgt = malloc(2 * p3 + 64); msk = malloc(p3); ph = malloc(p3 + 64); w4 = malloc(p3 + 64); wc = malloc(p3 + 64);
    for (int z = 0; z < P; z++) { cy[z] = P * .5f + .1f * z - 300; cx[z] = P * .4f - 200; }
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {   /* a spiral-ish sheet stack around an axis outside the window */
        const size_t k = ((size_t)z * P + y) * P + x;
        const double r = hypot(y - cy[z], x - cx[z]), q = r / 14.0 + .002 * z;
        const double f = q - floor(q);
        ct[k] = (uint8_t)(f < .35 ? 140 + (k * 2654435761u >> 28) : (x + y) % 97 == 0 ? 0 : 40 + (k * 40503u >> 29));
        tgt[k] = f < .2 ? 255 : 0; tgt[p3 + k] = f >= .2 && f < .35 ? 255 : 0; msk[k] = ct[k] != 0;
        ph[k] = (x * 7 + y * 3) % 211 == 0 ? PHASE_UNKNOWN : (uint8_t)((int)(f * PHASE_PERIOD) % PHASE_PERIOD);
    }
    { uint8_t *pw = malloc(p3); for (size_t k = 0; k < p3; k++) pw[k] = ct[k] ? ph[k] : SIDE_UNKNOWN;
      const int wt = (P + 5) / 6; if (band_wind4_tiled(pw, P, P, P, wt, 80, 512, 1, w4, wc, nullptr) < 0) return 1; free(pw); }
    wq = malloc(p3 * 2); warp_w4_pack(w4, wc, wq, p3);
    pthread_t th[256]; targ ta[256];
    const double t0 = now();
    for (int i = 0; i < NT; i++) { memset(&ta[i], 0, sizeof ta[i]); ta[i].id = i; pthread_create(&th[i], nullptr, run, &ta[i]); }
    for (int i = 0; i < NT; i++) pthread_join(th[i], nullptr);
    const double wall = now() - t0;
    printf("P %d, %d threads, %d windows each (%.1f s)\n", P, NT, REPS, wall);
    double tot = 0;
    for (int op = 0; op < OP_N; op++) {
        if (!ops_on[op]) continue;
        double s = 0; for (int i = 0; i < NT; i++) s += ta[i].t[op];
        const double ms = 1e3 * s / NT / REPS; tot += op <= OP_FUSED && op != OP_TM_NONE && op != OP_PH && op != OP_W4 ? ms : 0;
        printf("  %-32s %9.1f ms per window per thread  (%.2f windows/s over %d threads)\n", names[op], ms, NT * 1e3 / ms, NT);
    }
    printf("  trainer warp path (tm LUT + fused phase/w4): %.1f ms per window per thread\n", tot);
    return 0;
}
