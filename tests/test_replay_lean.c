/* The lean slot layouts and replay together, on the CPU: a --task unwrap2 --rv sampler with --gpu-aug 2 byte planes (the plain path)
   against the same seed with ink bits (batch.inkb: the papyrus plane as 1 bit + ct8), w4 runs (batch.w4r: no w4 / wcomp planes, the
   phase on the source grid; its CPU runs path: no device here) and --replay 2. Every new window (batch.use 0) must hash the same, plane
   by plane, as the plain path's window of the same index (the lean planes rebuilt on the host as the device builds them:
   sampler_batch_hash); each replay (use 1) must keep its window's CT bytes and be warped. Usage: test_replay_lean [windows 12] */
#define WIND_FIXTURE_NO_MAIN
#include "make_wind_fixture.c"
#include "sample.h"
#include "sources.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

typedef struct { int n; const char *nm[32]; uint64_t h[32]; uint64_t ct8; int use, warped; } rec;
static uint64_t hash_of(const rec *r, const char *nm) { for (int q = 0; q < r->n; q++) if (!strcmp(r->nm[q], nm)) return r->h[q]; return 0; }

int main(int argc, char **argv) {
    const int NW = argc > 1 ? atoi(argv[1]) : 12;
    setenv("UFSM_SAMPLER_NOPIN", "1", 1);   /* (pageable slots: no CUDA) */
    char tmpd[] = "/tmp/ufsm-replaylean-XXXXXX"; const char *dir = mkdtemp(tmpd);
    CHECK(dir && !make_wind_fixture(dir, 256, 2.f), "wind fixture");
    char path[1400]; snprintf(path, sizeof path, "%s/sources.json", dir);
    sources *S = sources_load(path); CHECK(S != nullptr, "sources");
    if (!S) return 1;
    sample_cfg c = sample_cfg_default();
    c.P = 64; c.B = 1; c.nworkers = 2; c.nbuf = 5; c.deterministic = 1; c.seed = 11; c.xfmt = 1; c.gpu_input = 1; c.side = 2; c.wind4 = 1; c.pack_tm = 1; c.pmask = 1;
    c.slot_skip = 3; c.gpu_aug = 2; c.ink_plane = 1; c.rv_iso = 100;   /* (as train sets them for --task unwrap2 --rv 1 --gpu-aug 2, B 1) */
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1; c.min_fg = 0;
    CHECK(!sample_aug_preset(&c, "all"), "preset");
    c.augment = 3; c.symmetry_p = .5f; c.rotate_p = .5f; c.zoom_p = .3f; c.elastic_p = .3f; c.zoom_out_p = 0; c.wind4_tile = 16;
    memset(c.scan.p, 0, sizeof c.scan.p); c.scan.strength = 1;
    const int sops[] = {SA_BLUR, SA_NOISE, SA_TONE, SA_BIAS, SA_SLICE, SA_CHUNK, SA_CUTOUT, SA_DROPSLICE};
    for (size_t k = 0; k < sizeof sops / sizeof *sops; k++) c.scan.p[sops[k]] = .15f;
    sample_cfg cl = c; cl.ink_bits = 1; cl.w4_runs = 1; cl.w4_dev = 1; cl.replay = 2; cl.replay_gap = 3;
    rec *A = calloc((size_t)NW, sizeof *A), *L = calloc((size_t)3 * NW, sizeof *L); int nl = 0;
    for (int run = 0; run < 2; run++) {
        sampler *sp = sampler_start(S, run ? &cl : &c); CHECK(sp != nullptr, "sampler %d", run);
        if (!sp) return 1;
        const int want = run ? 2 * NW : NW;
        for (int i = 0; i < want; i++) {
            batch *b = sampler_next(sp); if (!b) { CHECK(0, "run %d: sampler stopped at %d", run, i); break; }
            if (run) CHECK(b->inkb && !b->ink && b->w4r && !b->w4 && !b->wcomp, "lean layout (ink bits, w4 runs)");
            rec *r = run ? &L[nl++] : &A[i];
            r->n = sampler_batch_hash(sp, b, r->nm, r->h, 32); r->use = b->use;
            int32_t h[8]; memcpy(h, b->rad, sizeof h); r->warped = h[6] & 1; r->ct8 = hash_of(r, "ct8");
            sampler_release(sp, b);
        }
        sampler_stop(sp);
    }
    int nnew = 0, nrep = 0;
    for (int i = 0; i < nl; i++) {
        const rec *r = &L[i];
        if (r->use == 0) {
            if (nnew >= NW) continue;
            const rec *a = &A[nnew];
            for (int q = 0; q < a->n; q++) {
                int found = 0; for (int t = 0; t < r->n; t++) if (!strcmp(r->nm[t], a->nm[q])) { found = 1; CHECK(r->h[t] == a->h[q], "window %d: plane %s differs from the plain path", nnew, a->nm[q]); }
                CHECK(found, "window %d: plane %s missing", nnew, a->nm[q]);
            }
            nnew++;
        } else {
            int orig = 0; for (int t = 0; t < i; t++) orig |= L[t].use == 0 && L[t].ct8 == r->ct8;
            CHECK(orig, "replay %d: no earlier window with its CT bytes", i);
            CHECK(r->warped, "replay %d: not warped", i);
            nrep++;
        }
    }
    CHECK(nnew >= NW / 2 && nrep > 0, "windows compared %d, replays %d", nnew, nrep);
    if (!fails) { char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {} }
    if (fails) { fprintf(stderr, "test_replay_lean: %d FAILED\n", fails); return 1; }
    printf("test_replay_lean: ok (%d new windows byte-identical to the plain path with ink bits + w4 runs + --replay 2, %d replays)\n", nnew, nrep);
    return 0;
}
